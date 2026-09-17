#include "gemv_sparse_accumulate.cuh"
#include "common.cuh"
#include "dequant.cuh"

namespace {
constexpr int kThreadsPerBlock = 256;
constexpr int kInt4Group = 8;  // values packed per uint32, per layout.h
}  // namespace

__global__ void gemv_w4a16_sparse_accumulate_kernel(const uint8_t* __restrict__ Wq_selected,
                                                      const float* __restrict__ scale_selected,
                                                      const half* __restrict__ h_selected, half* __restrict__ y,
                                                      int H, int k, int group_size, int num_groups) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= H) return;

    const int words_per_row = ceil_div(H, kInt4Group);
    const int word_idx = c / kInt4Group;
    const int nibble_idx = c % kInt4Group;
    const int pair_idx = nibble_idx / 2;
    const bool high_half = (nibble_idx % 2) != 0;
    const int group_idx = c / group_size;

    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        const uint32_t* row_words =
            reinterpret_cast<const uint32_t*>(Wq_selected + static_cast<size_t>(i) * words_per_row * 4);
        const uint32_t packed = row_words[word_idx];
        half2 dq[4];
        dequant_int4x8_awq(packed, dq);
        const half val_h = high_half ? __high2half(dq[pair_idx]) : __low2half(dq[pair_idx]);

        const float s = scale_selected[static_cast<size_t>(i) * num_groups + group_idx];
        const float h_i = __half2float(h_selected[i]);
        acc += h_i * s * __half2float(val_h);
    }
    y[c] = __float2half(acc);
}

void launch_gemv_w4a16_sparse_accumulate(const uint8_t* Wq_selected, const float* scale_selected,
                                          const half* h_selected, half* y, int H, int k, int group_size,
                                          int num_groups) {
    const int blocks = ceil_div(H, kThreadsPerBlock);
    gemv_w4a16_sparse_accumulate_kernel<<<blocks, kThreadsPerBlock>>>(Wq_selected, scale_selected, h_selected, y, H,
                                                                       k, group_size, num_groups);
    CUDA_CHECK(cudaGetLastError());
}
