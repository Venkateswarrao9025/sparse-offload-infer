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

    // Phase 3: write every qualifying index (> tau unconditionally, == tau
    // until k total slots are filled) in ASCENDING INDEX ORDER -- a single
    // deterministic scan, not just for which indices tie at the boundary
    // (an earlier version of this kernel only fixed non-determinism THERE,
    // reasoning that atomicAdd race order among the strictly-greater
    // majority was "harmless" since it can't change which indices are
    // selected). That reasoning covered SET correctness but missed a real
    // consequence: out_indices' ORDER feeds gemv_w4a16_sparse_accumulate's
    // sequential float accumulation downstream, which is NOT
    // order-independent -- summing the identical k values in a different
    // (atomicAdd-race-dependent) order produces a different rounded
    // result. Invisible to integer-count-based determinism tests (M8
    // calibration's histograms don't care about order), but real: two
    // back-to-back calls on the SAME frozen input returned the IDENTICAL
    // SET (confirmed empirically, symmetric difference zero) yet
    // DIFFERENT perplexity when fed through the real 28-layer decode
    // loop -- caught building M8's ablation table, on real hardware, not
    // by any test. A single-threaded O(n) scan trades some of this
    // kernel's already-known-suboptimal parallelism (documented follow-up:
    // radix-select/multi-block redesign, not started) for a stronger and
    // actually-necessary guarantee: bit-for-bit reproducible output, not
    // just a reproducible SET.
    const float tau = s_lo;
    if (tid == 0) {
        int write_pos = 0;
        for (int i = 0; i < n && write_pos < k; ++i) {
            if (vals[i] >= tau) {
                out_indices[write_pos] = i;
                ++write_pos;
            }
        }
        *out_count = write_pos;
    }
}

void launch_topk_threshold_select(const float* abs_g, int n, int k, int* out_indices, int* out_count) {
    const size_t shmem = ceil_div(kThreads, 32) * sizeof(float);
    topk_threshold_select_kernel<<<1, kThreads, shmem>>>(abs_g, n, k, out_indices, out_count);
    CUDA_CHECK(cudaGetLastError());
}
