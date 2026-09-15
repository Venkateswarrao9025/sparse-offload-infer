#include "gemv_fp16.cuh"
#include "common.cuh"
#include "reduce.cuh"

namespace {
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
}

// --- v1: one thread per row, naive sequential dot product ------------------

__global__ void gemv_v1_kernel(const half* __restrict__ W, const half* __restrict__ x, half* __restrict__ y,
                                int N, int K) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const half* row_ptr = W + static_cast<size_t>(row) * K;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += __half2float(row_ptr[k]) * __half2float(x[k]);
    }
    y[row] = __float2half(acc);
}

void launch_gemv_fp16_v1(const half* W, const half* x, half* y, int N, int K) {
    const int threads = 256;
    const int blocks = ceil_div(N, threads);
    gemv_v1_kernel<<<blocks, threads>>>(W, x, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// --- v2: one warp per row, strided dot product + warp-shuffle reduce -------

__global__ void gemv_v2_kernel(const half* __restrict__ W, const half* __restrict__ x, half* __restrict__ y,
                                int N, int K) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * kWarpsPerBlock + warp_id;
    if (row >= N) return;

    const half* row_ptr = W + static_cast<size_t>(row) * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) {
        acc += __half2float(row_ptr[k]) * __half2float(x[k]);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        y[row] = __float2half(acc);
    }
}

void launch_gemv_fp16_v2(const half* W, const half* x, half* y, int N, int K) {
    const int blocks = ceil_div(N, kWarpsPerBlock);
    gemv_v2_kernel<<<blocks, kThreadsPerBlock>>>(W, x, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// --- v3: v2 + float4-vectorized loads (8 half elements/iteration) ----------

__global__ void gemv_v3_kernel(const half* __restrict__ W, const half* __restrict__ x, half* __restrict__ y,
                                int N, int K) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * kWarpsPerBlock + warp_id;
    if (row >= N) return;

    const half* row_ptr = W + static_cast<size_t>(row) * K;
    const float4* row4 = reinterpret_cast<const float4*>(row_ptr);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int K8 = K / 8;

    float acc = 0.0f;
    for (int k = lane; k < K8; k += 32) {
        const float4 wv = row4[k];
        const float4 xv = x4[k];
        const half2* wh = reinterpret_cast<const half2*>(&wv);
        const half2* xh = reinterpret_cast<const half2*>(&xv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 wf = __half22float2(wh[j]);
            const float2 xf = __half22float2(xh[j]);
            acc += wf.x * xf.x + wf.y * xf.y;
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        y[row] = __float2half(acc);
    }
}

void launch_gemv_fp16_v3(const half* W, const half* x, half* y, int N, int K) {
    const int blocks = ceil_div(N, kWarpsPerBlock);
    gemv_v3_kernel<<<blocks, kThreadsPerBlock>>>(W, x, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// --- v4: split-K, one warp per (row, K-chunk), atomicAdd into fp32 accum ---

__global__ void gemv_v4_splitk_kernel(const half* __restrict__ W, const half* __restrict__ x,
                                       float* __restrict__ y_accum, int N, int K, int split) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * kWarpsPerBlock + warp_id;
    const int chunk = blockIdx.y;
    if (row >= N || chunk >= split) return;

    const int chunk_size = ceil_div(K, split);
    const int k_start = chunk * chunk_size;
    const int k_end = min(k_start + chunk_size, K);
    if (k_start >= k_end) return;

    const half* row_ptr = W + static_cast<size_t>(row) * K;
    float acc = 0.0f;
    for (int k = k_start + lane; k < k_end; k += 32) {
        acc += __half2float(row_ptr[k]) * __half2float(x[k]);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        atomicAdd(&y_accum[row], acc);
    }
}

void launch_gemv_fp16_v4_splitk(const half* W, const half* x, float* y_accum, int N, int K, int split) {
    dim3 block(kThreadsPerBlock);
    dim3 grid(ceil_div(N, kWarpsPerBlock), split);
    gemv_v4_splitk_kernel<<<grid, block>>>(W, x, y_accum, N, K, split);
    CUDA_CHECK(cudaGetLastError());
}
