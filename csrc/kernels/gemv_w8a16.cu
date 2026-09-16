#include "gemv_w8a16.cuh"
#include "common.cuh"
#include "reduce.cuh"

namespace {
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
}  // namespace

// One warp per output row, lane-strided over K -- same access pattern as
// gemv_fp16_v2/v3 (lane l reads word l+32*iter, so all 32 lanes' reads in one
// iteration are 32 CONSECUTIVE uint32 words, i.e. one coalesced 128-byte
// transaction). An earlier version instead gave each lane a private
// contiguous chunk of K so the group scale could sit in a register across a
// whole group; that broke coalescing (consecutive lanes read addresses
// hundreds of bytes apart) and measured 10-20x SLOWER than gemv_fp16_v3
// despite moving 4x fewer bytes (see docs/LEARNING_NOTES.md's M4 entry).
// Coalescing dominates: the scale here is instead loaded once per WORD
// (amortized over 4 elements) rather than cached across a whole group --
// still nowhere near "reload it per element" (PROJECT_SPEC.md M4 task 2's
// warning), just not maximally amortized, and it keeps every load coalesced.
__global__ void gemv_w8a16_kernel(const int8_t* __restrict__ Wq, const float* __restrict__ scale,
                                   const half* __restrict__ x, half* __restrict__ y, int N, int K,
                                   int group_size, int num_groups, int scale_row_stride) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * kWarpsPerBlock + warp_id;
    if (row >= N) return;

    const int8_t* row_ptr = Wq + static_cast<size_t>(row) * K;
    const float* srow = scale + static_cast<size_t>(row) * scale_row_stride;
    const uint32_t* row4 = reinterpret_cast<const uint32_t*>(row_ptr);
    const int K4 = K / 4;

    float acc = 0.0f;
    for (int k4 = lane; k4 < K4; k4 += 32) {
        const int k = k4 * 4;
        const float s = srow[k / group_size];
        const uint32_t packed = row4[k4];
        const int8_t v0 = static_cast<int8_t>(packed & 0xFFu);
        const int8_t v1 = static_cast<int8_t>((packed >> 8) & 0xFFu);
        const int8_t v2 = static_cast<int8_t>((packed >> 16) & 0xFFu);
        const int8_t v3 = static_cast<int8_t>((packed >> 24) & 0xFFu);
        acc += s * (static_cast<float>(v0) * __half2float(x[k + 0]) +
                    static_cast<float>(v1) * __half2float(x[k + 1]) +
                    static_cast<float>(v2) * __half2float(x[k + 2]) +
                    static_cast<float>(v3) * __half2float(x[k + 3]));
    }
    // Tail: K not a multiple of 4. At most 3 elements, so a single lane
    // handling them serially (folded into its own `acc` before the warp
    // reduce) is cheap and avoids any alignment assumptions.
    if (lane == 0) {
        for (int k = K4 * 4; k < K; ++k) {
            const float s = srow[k / group_size];
            acc += s * static_cast<float>(row_ptr[k]) * __half2float(x[k]);
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
