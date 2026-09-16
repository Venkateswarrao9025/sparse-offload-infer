#include "gemv_w8a16.cuh"
#include "common.cuh"
#include "reduce.cuh"

namespace {
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
}  // namespace

// One warp per output row. K is split into 32 CONTIGUOUS per-lane chunks
// (rounded up to a multiple of 4, one uint32 load's worth of int8 weights)
// rather than the usual lane-strided split, so that each lane's chunk stays
// inside as few scale groups as possible -- the group scale is then loaded
// once per group and kept in a register (`s`) across every element of that
// group, instead of re-reading it from global memory per element.
__global__ void gemv_w8a16_kernel(const int8_t* __restrict__ Wq, const float* __restrict__ scale,
                                   const half* __restrict__ x, half* __restrict__ y, int N, int K,
                                   int group_size, int num_groups, int scale_row_stride) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * kWarpsPerBlock + warp_id;
    if (row >= N) return;

    const int8_t* row_ptr = Wq + static_cast<size_t>(row) * K;
    const float* srow = scale + static_cast<size_t>(row) * scale_row_stride;

    const int words_total = ceil_div(K, 4);
    const int words_per_lane = ceil_div(words_total, 32);
    const int k_start = lane * words_per_lane * 4;
    const int k_end = min(k_start + words_per_lane * 4, K);

    float acc = 0.0f;
    int last_group = -1;
    float s = 0.0f;
    for (int k = k_start; k < k_end; k += 4) {
        const int group = k / group_size;
        if (group != last_group) {
            s = srow[group];
            last_group = group;
        }
        const int remaining = K - k;
        if (remaining >= 4) {
            // Vectorized path: one uint32 load = 4 packed int8 weights.
            const uint32_t packed = *reinterpret_cast<const uint32_t*>(row_ptr + k);
            const int8_t v0 = static_cast<int8_t>(packed & 0xFFu);
            const int8_t v1 = static_cast<int8_t>((packed >> 8) & 0xFFu);
            const int8_t v2 = static_cast<int8_t>((packed >> 16) & 0xFFu);
            const int8_t v3 = static_cast<int8_t>((packed >> 24) & 0xFFu);
            acc += s * (static_cast<float>(v0) * __half2float(x[k + 0]) +
                        static_cast<float>(v1) * __half2float(x[k + 1]) +
                        static_cast<float>(v2) * __half2float(x[k + 2]) +
                        static_cast<float>(v3) * __half2float(x[k + 3]));
        } else {
            // Tail: K not a multiple of 4. row_ptr only has K valid bytes,
            // never a full padded uint32, so fall back to scalar reads.
#pragma unroll
            for (int i = 0; i < remaining; ++i) {
                acc += s * static_cast<float>(row_ptr[k + i]) * __half2float(x[k + i]);
            }
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        y[row] = __float2half(acc);
    }
}

void launch_gemv_w8a16(const int8_t* Wq, const float* scale, const half* x, half* y, int N, int K,
                        int group_size, int num_groups, int scale_row_stride) {
    const int blocks = ceil_div(N, kWarpsPerBlock);
    gemv_w8a16_kernel<<<blocks, kThreadsPerBlock>>>(Wq, scale, x, y, N, K, group_size, num_groups,
                                                      scale_row_stride);
    CUDA_CHECK(cudaGetLastError());
}
