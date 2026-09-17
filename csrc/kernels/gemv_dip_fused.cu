#include "gemv_dip_fused.cuh"
#include "common.cuh"
#include "dequant.cuh"
#include "reduce.cuh"

namespace {
constexpr int kBuildThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = 4;
constexpr int kGemvThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kColumnThreadsPerBlock = 256;
constexpr int kInt4Group = 8;  // values packed per uint32, per layout.h

__device__ __forceinline__ bool desc_from_staging(int32_t desc) { return (desc >> 31) != 0; }
__device__ __forceinline__ int32_t desc_offset(int32_t desc) { return desc & 0x7fffffff; }
}  // namespace

__global__ void build_dip_descriptors_kernel(const int32_t* __restrict__ selected_indices, int k,
                                              const int32_t* __restrict__ slot_of,
                                              int32_t* __restrict__ out_descriptors,
                                              int32_t* __restrict__ out_miss_channels,
                                              int32_t* __restrict__ out_miss_count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= k) return;
    const int32_t channel = selected_indices[tid];
    const int32_t slot = slot_of[channel];
    if (slot >= 0) {
        out_descriptors[tid] = slot;  // top bit 0: cache
    } else {
        const int32_t miss_pos = atomicAdd(out_miss_count, 1);
        out_miss_channels[miss_pos] = channel;
        out_descriptors[tid] = miss_pos | (1 << 31);  // top bit 1: staging
    }
}

void launch_build_dip_descriptors(const int32_t* selected_indices, int k, const int32_t* slot_of,
                                   int32_t* out_descriptors, int32_t* out_miss_channels, int32_t* out_miss_count) {
    CUDA_CHECK(cudaMemsetAsync(out_miss_count, 0, sizeof(int32_t)));
    const int blocks = ceil_div(k, kBuildThreadsPerBlock);
    build_dip_descriptors_kernel<<<blocks, kBuildThreadsPerBlock>>>(selected_indices, k, slot_of, out_descriptors,
                                                                     out_miss_channels, out_miss_count);
    CUDA_CHECK(cudaGetLastError());
}

// --- "up" direction: one warp per selected row, same lane-strided/coalesced
// access and LOP3 dequant as gemv_w4a16_group_lop3 (M4), except the row's
// packed-weight/scale base pointer comes from its descriptor instead of a
// single fixed buffer.
__global__ void gemv_dip_fused_up_kernel(const uint8_t* __restrict__ cache_Wq, const float* __restrict__ cache_scale,
                                          const uint8_t* __restrict__ staging_Wq,
                                          const float* __restrict__ staging_scale,
                                          const int32_t* __restrict__ descriptors, const half* __restrict__ x,
                                          half* __restrict__ y, int k, int K, int group_size, int num_groups) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * kWarpsPerBlock + warp_id;
    if (row >= k) return;

    const int32_t desc = descriptors[row];
    const bool from_staging = desc_from_staging(desc);
    const int32_t offset = desc_offset(desc);

    const int words_per_row = ceil_div(K, kInt4Group);
    const uint8_t* row_bytes = from_staging ? (staging_Wq + static_cast<size_t>(offset) * words_per_row * 4)
                                             : (cache_Wq + static_cast<size_t>(offset) * words_per_row * 4);
    const float* srow = from_staging ? (staging_scale + static_cast<size_t>(offset) * num_groups)
                                      : (cache_scale + static_cast<size_t>(offset) * num_groups);
    const uint32_t* row_words = reinterpret_cast<const uint32_t*>(row_bytes);
    const float4* x4 = reinterpret_cast<const float4*>(x);

    float acc = 0.0f;
    for (int w = lane; w < words_per_row; w += 32) {
        const int kk = w * kInt4Group;
        const float s = srow[kk / group_size];
        const uint32_t packed = row_words[w];
        half2 dq[4];
        dequant_int4x8_awq(packed, dq);

        const int remaining = K - kk;
        if (remaining >= kInt4Group) {
            const float4 xv = x4[w];
            const half2* xh = reinterpret_cast<const half2*>(&xv);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 dqf = __half22float2(dq[j]);
                const float2 xf = __half22float2(xh[j]);
                acc += s * (dqf.x * xf.x + dqf.y * xf.y);
            }
        } else {
            const half* dq_half = reinterpret_cast<half*>(dq);
#pragma unroll
            for (int i = 0; i < kInt4Group; ++i) {
                if (i < remaining) {
                    acc += s * __half2float(dq_half[i]) * __half2float(x[kk + i]);
                }
            }
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        y[row] = __float2half(acc);
    }
}

void launch_gemv_dip_fused_up(const uint8_t* cache_Wq, const float* cache_scale, const uint8_t* staging_Wq,
                               const float* staging_scale, const int32_t* descriptors, const half* x, half* y, int k,
                               int K, int group_size, int num_groups) {
    const int blocks = ceil_div(k, kWarpsPerBlock);
    gemv_dip_fused_up_kernel<<<blocks, kGemvThreadsPerBlock>>>(cache_Wq, cache_scale, staging_Wq, staging_scale,
                                                                descriptors, x, y, k, K, group_size, num_groups);
    CUDA_CHECK(cudaGetLastError());
}

// --- "down" direction: one thread per output column, looping over the k
// selected rows (same parallelization as gemv_w4a16_sparse_accumulate, M7
// task 3), each row's source resolved via its descriptor.
__global__ void gemv_dip_fused_down_kernel(const uint8_t* __restrict__ cache_Wq,
                                            const float* __restrict__ cache_scale,
                                            const uint8_t* __restrict__ staging_Wq,
                                            const float* __restrict__ staging_scale,
                                            const int32_t* __restrict__ descriptors,
                                            const half* __restrict__ h_selected, half* __restrict__ y, int H, int k,
                                            int group_size, int num_groups) {
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
        const int32_t desc = descriptors[i];
        const bool from_staging = desc_from_staging(desc);
        const int32_t offset = desc_offset(desc);
        const uint8_t* row_bytes = from_staging ? (staging_Wq + static_cast<size_t>(offset) * words_per_row * 4)
                                                 : (cache_Wq + static_cast<size_t>(offset) * words_per_row * 4);
        const float* srow = from_staging ? (staging_scale + static_cast<size_t>(offset) * num_groups)
                                          : (cache_scale + static_cast<size_t>(offset) * num_groups);
        const uint32_t* row_words = reinterpret_cast<const uint32_t*>(row_bytes);

        const uint32_t packed = row_words[word_idx];
        half2 dq[4];
        dequant_int4x8_awq(packed, dq);
        const half val_h = high_half ? __high2half(dq[pair_idx]) : __low2half(dq[pair_idx]);

        const float s = srow[group_idx];
        const float h_i = __half2float(h_selected[i]);
        acc += h_i * s * __half2float(val_h);
    }
    y[c] = __float2half(acc);
}

void launch_gemv_dip_fused_down(const uint8_t* cache_Wq, const float* cache_scale, const uint8_t* staging_Wq,
                                 const float* staging_scale, const int32_t* descriptors, const half* h_selected,
                                 half* y, int H, int k, int group_size, int num_groups) {
    const int blocks = ceil_div(H, kColumnThreadsPerBlock);
    gemv_dip_fused_down_kernel<<<blocks, kColumnThreadsPerBlock>>>(cache_Wq, cache_scale, staging_Wq, staging_scale,
                                                                    descriptors, h_selected, y, H, k, group_size,
                                                                    num_groups);
    CUDA_CHECK(cudaGetLastError());
}
