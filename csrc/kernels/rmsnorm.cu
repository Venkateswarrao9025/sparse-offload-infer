#include "rmsnorm.cuh"
#include "common.cuh"
#include "reduce.cuh"

namespace {
constexpr int kThreads = 256;
}

__global__ void rmsnorm_kernel(const half* __restrict__ x, const half* __restrict__ weight,
                                half* __restrict__ out, int hidden, float eps) {
    extern __shared__ float sdata[];  // (blockDim.x / 32) floats, for block_reduce_sum

    const int row = blockIdx.x;
    const int hidden2 = hidden / 2;
    const half2* x_row = reinterpret_cast<const half2*>(x + static_cast<size_t>(row) * hidden);
    half2* out_row = reinterpret_cast<half2*>(out + static_cast<size_t>(row) * hidden);
    const half2* w2 = reinterpret_cast<const half2*>(weight);

    float local_sumsq = 0.0f;
    for (int i = threadIdx.x; i < hidden2; i += blockDim.x) {
        const float2 v = __half22float2(x_row[i]);
        local_sumsq += v.x * v.x + v.y * v.y;
    }

    const float total = block_reduce_sum(local_sumsq, sdata);

    __shared__ float rms_inv;
    if (threadIdx.x == 0) {
        const float mean_sq = total / static_cast<float>(hidden);
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hidden2; i += blockDim.x) {
        const float2 v = __half22float2(x_row[i]);
        const float2 w = __half22float2(w2[i]);
        float2 r;
        r.x = v.x * rms_inv * w.x;
        r.y = v.y * rms_inv * w.y;
        out_row[i] = __float22half2_rn(r);
    }
}

void launch_rmsnorm(const half* x, const half* weight, half* out, int rows, int hidden, float eps) {
    const size_t shmem = (kThreads / 32) * sizeof(float);
    rmsnorm_kernel<<<rows, kThreads, shmem>>>(x, weight, out, hidden, eps);
    CUDA_CHECK(cudaGetLastError());
}
