#include "topk_select.cuh"
#include "common.cuh"
#include "reduce.cuh"

namespace {
constexpr int kThreads = 256;
constexpr int kBisectionIters = 24;  // fp32 precision: range halves 24x, plenty for realistic activation magnitudes
}  // namespace

__global__ void topk_threshold_select_kernel(const float* __restrict__ vals, int n, int k,
                                              int* __restrict__ out_indices, int* __restrict__ out_count) {
    extern __shared__ float sdata[];  // (blockDim.x / 32) floats, for block_reduce_sum/max
    __shared__ float s_lo, s_hi, s_mid;
    __shared__ int s_counter;

    const int tid = threadIdx.x;

    // Phase 1: block-wide max, to bound the initial search range. vals are
    // assumed non-negative (magnitudes), so 0 is a safe lower bound.
    float local_max = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, vals[i]);
    }
    const float total_max = block_reduce_max(local_max, sdata);
    if (tid == 0) {
        s_lo = 0.0f;
        s_hi = total_max + 1e-3f;  // strictly above the max, so count(>=hi) starts at 0
    }
    __syncthreads();

    // Phase 2: bisect on threshold tau, maintaining count(>=lo) >= k,
    // count(>=hi) < k. Every thread participates in every iteration's
    // count (a block_reduce_sum, which itself __syncthreads()s), so this
    // needs no separate kernel launch per iteration -- that's the whole
    // point: launch overhead would dominate the microsecond budget task 1
    // targets otherwise.
#pragma unroll 1
    for (int iter = 0; iter < kBisectionIters; ++iter) {
        if (tid == 0) {
            s_mid = 0.5f * (s_lo + s_hi);
        }
        __syncthreads();
        const float mid = s_mid;

        float local_count = 0.0f;
        for (int i = tid; i < n; i += blockDim.x) {
            if (vals[i] >= mid) local_count += 1.0f;
        }
        // n <= a few hundred thousand fits exactly in fp32 (integers up to
        // 2^24 are exact), so accumulating the count as float and reusing
        // block_reduce_sum needs no separate integer reduction primitive.
        const float total_count = block_reduce_sum(local_count, sdata);
        __syncthreads();  // all threads must read total_count's inputs before lo/hi mutate
        if (tid == 0) {
            if (total_count >= static_cast<float>(k)) {
                s_lo = mid;
            } else {
                s_hi = mid;
            }
        }
        __syncthreads();
    }

    // Phase 3: compact every index clearing the converged threshold.
    // atomicAdd-assigned output slots -- unordered, which is fine (see
    // topk_select.cuh's header comment).
    if (tid == 0) {
        s_counter = 0;
    }
    __syncthreads();
    const float tau = s_lo;
    for (int i = tid; i < n; i += blockDim.x) {
        if (vals[i] >= tau) {
            const int pos = atomicAdd(&s_counter, 1);
            if (pos < k) {
                out_indices[pos] = i;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        *out_count = min(s_counter, k);
    }
}

void launch_topk_threshold_select(const float* abs_g, int n, int k, int* out_indices, int* out_count) {
    const size_t shmem = ceil_div(kThreads, 32) * sizeof(float);
    topk_threshold_select_kernel<<<1, kThreads, shmem>>>(abs_g, n, k, out_indices, out_count);
    CUDA_CHECK(cudaGetLastError());
}
