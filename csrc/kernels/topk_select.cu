#include "topk_select.cuh"
#include "common.cuh"
#include "reduce.cuh"

namespace {
constexpr int kThreads = 1024;  // max threads/block on Turing -- more parallelism within the single block
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

    // Phase 3: gather every index STRICTLY above tau first -- these are
    // unambiguously in the top-k regardless of output slot, since nothing
    // at the boundary can displace them, so atomicAdd race order among
    // them is harmless (see topk_select.cuh's header comment).
    if (tid == 0) {
        s_counter = 0;
    }
    __syncthreads();
    const float tau = s_lo;
    for (int i = tid; i < n; i += blockDim.x) {
        if (vals[i] > tau) {
            const int pos = atomicAdd(&s_counter, 1);
            if (pos < k) {
                out_indices[pos] = i;
            }
        }
    }
    __syncthreads();
    // Any remaining slots come from indices tied exactly AT tau. Ties are
    // NOT measure-zero here: |gate| comes from dequantized activations, so
    // exact float equality across channels is common (unlike the
    // continuous-float assumption this kernel started with). Filling them
    // by atomicAdd race order made the returned SET depend on GPU thread
    // scheduling -- invisible on M7's original continuous-activation
    // tests, but non-deterministic once M8's calibration pass fed it real
    // quantized data. Fill by ascending index instead: single-threaded and
    // O(n), but only runs the `remaining` iterations that matter and ties
    // are the rare case, so this doesn't reopen the >1 SM performance
    // question this kernel already has open.
    if (tid == 0) {
        const int base = min(s_counter, k);
        int remaining = k - base;
        int filled = 0;
        for (int i = 0; i < n && filled < remaining; ++i) {
            if (vals[i] == tau) {
                out_indices[base + filled] = i;
                ++filled;
            }
        }
        *out_count = base + filled;
    }
}

void launch_topk_threshold_select(const float* abs_g, int n, int k, int* out_indices, int* out_count) {
    const size_t shmem = ceil_div(kThreads, 32) * sizeof(float);
    topk_threshold_select_kernel<<<1, kThreads, shmem>>>(abs_g, n, k, out_indices, out_count);
    CUDA_CHECK(cudaGetLastError());
}
