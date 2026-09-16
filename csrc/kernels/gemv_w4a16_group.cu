#include "gemv_w4a16_group.cuh"
#include "common.cuh"
#include "dequant.cuh"
#include "reduce.cuh"

namespace {
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kInt4Group = 8;  // values packed per uint32, per layout.h
}  // namespace

// --- naive: scalar nibble unpack, lane-strided over int4-groups-of-8.
//
// Same access pattern as gemv_fp16_v2/v3 / gemv_w8a16: lane l reads word
// l+32*iter, so all 32 lanes' reads in one iteration are 32 CONSECUTIVE
// uint32 words -- one coalesced transaction. See gemv_w8a16.cu's comment
// (and docs/LEARNING_NOTES.md's M4 entry) for why this replaced an earlier
// contiguous-per-lane-chunk design that broke coalescing for a marginal
// scale-caching win and measured far slower despite moving 1/4 the bytes.
// The int4-packed buffer is already padded per-row to a whole number of
// uint32 words (see layout.h), so unlike gemv_w8a16 there's no separate
// scalar tail loop needed -- the last word's own `remaining` guard covers it.
__global__ void gemv_w4a16_group_kernel(const uint8_t* __restrict__ Wq, const float* __restrict__ scale,
                                         const half* __restrict__ x, half* __restrict__ y, int N, int K,
                                         int group_size, int num_groups) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * kWarpsPerBlock + warp_id;
    if (row >= N) return;

    const int words_per_row = ceil_div(K, kInt4Group);
    const uint32_t* row_words = reinterpret_cast<const uint32_t*>(Wq) + static_cast<size_t>(row) * words_per_row;
    const float* srow = scale + static_cast<size_t>(row) * num_groups;

    float acc = 0.0f;
    for (int w = lane; w < words_per_row; w += 32) {
        const int k = w * kInt4Group;
        const float s = srow[k / group_size];
        const uint32_t packed = row_words[w];
        int v[kInt4Group];
        v[0] = (packed >> 0) & 0xF;
        v[2] = (packed >> 4) & 0xF;
        v[4] = (packed >> 8) & 0xF;
        v[6] = (packed >> 12) & 0xF;
        v[1] = (packed >> 16) & 0xF;
        v[3] = (packed >> 20) & 0xF;
        v[5] = (packed >> 24) & 0xF;
        v[7] = (packed >> 28) & 0xF;
#pragma unroll
        for (int i = 0; i < kInt4Group; ++i) {
            if (v[i] >= 8) v[i] -= 16;  // sign-extend
        }
        const int remaining = min(kInt4Group, K - k);
#pragma unroll
        for (int i = 0; i < remaining; ++i) {
            acc += s * static_cast<float>(v[i]) * __half2float(x[k + i]);
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        y[row] = __float2half(acc);
    }
}

void launch_gemv_w4a16_group(const uint8_t* Wq, const float* scale, const half* x, half* y, int N, int K,
                              int group_size, int num_groups) {
    const int blocks = ceil_div(N, kWarpsPerBlock);
    gemv_w4a16_group_kernel<<<blocks, kThreadsPerBlock>>>(Wq, scale, x, y, N, K, group_size, num_groups);
    CUDA_CHECK(cudaGetLastError());
}

// --- lop3: same lane-strided/coalesced access, but dequantizes a whole
// int4-group (8 values) to half2x4 via dequant_int4x8_awq (dequant.cuh)
// instead of scalar shift/mask/sign-extend, and loads x with one float4
// (8-half) vectorized load instead of 8 scalar __half2float calls -- lane l's
// x4[w] read is likewise 32-lane coalesced (l and l+1 read adjacent float4s).
__global__ void gemv_w4a16_group_lop3_kernel(const uint8_t* __restrict__ Wq, const float* __restrict__ scale,
                                              const half* __restrict__ x, half* __restrict__ y, int N, int K,
                                              int group_size, int num_groups) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * kWarpsPerBlock + warp_id;
    if (row >= N) return;

    const int words_per_row = ceil_div(K, kInt4Group);
    const uint32_t* row_words = reinterpret_cast<const uint32_t*>(Wq) + static_cast<size_t>(row) * words_per_row;
    const float* srow = scale + static_cast<size_t>(row) * num_groups;
    const float4* x4 = reinterpret_cast<const float4*>(x);  // 8 halfs per float4

    float acc = 0.0f;
    for (int w = lane; w < words_per_row; w += 32) {
        const int k = w * kInt4Group;
        const float s = srow[k / group_size];
        const uint32_t packed = row_words[w];
        half2 dq[4];
        dequant_int4x8_awq(packed, dq);

        const int remaining = K - k;
        if (remaining >= kInt4Group) {
            // Fast path: full group, x is 8-half aligned (K is padded to a
            // multiple of 8 in the packed buffer, and callers guarantee x's
            // base pointer is 8-half aligned via a contiguous CUDA tensor).
            const float4 xv = x4[w];
            const half2* xh = reinterpret_cast<const half2*>(&xv);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 dqf = __half22float2(dq[j]);
                const float2 xf = __half22float2(xh[j]);
                acc += s * (dqf.x * xf.x + dqf.y * xf.y);
            }
        } else {
            // Tail group: K not a multiple of 8. Fall back to scalar reads
            // so we never touch x past index K-1.
            const half* dq_half = reinterpret_cast<half*>(dq);
#pragma unroll
            for (int i = 0; i < kInt4Group; ++i) {
                if (i < remaining) {
                    acc += s * __half2float(dq_half[i]) * __half2float(x[k + i]);
                }
            }
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        y[row] = __float2half(acc);
    }
}

void launch_gemv_w4a16_group_lop3(const uint8_t* Wq, const float* scale, const half* x, half* y, int N, int K,
                                   int group_size, int num_groups) {
    const int blocks = ceil_div(N, kWarpsPerBlock);
    gemv_w4a16_group_lop3_kernel<<<blocks, kThreadsPerBlock>>>(Wq, scale, x, y, N, K, group_size, num_groups);
    CUDA_CHECK(cudaGetLastError());
}
