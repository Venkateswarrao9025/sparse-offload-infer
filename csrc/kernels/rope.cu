#include "rope.cuh"
#include "common.cuh"

// One block per head, one thread per (i, i+head_dim/2) pair.
__global__ void rope_apply_kernel(half* __restrict__ x, const float* __restrict__ cos_vals,
                                   const float* __restrict__ sin_vals, int head_dim) {
    const int h = blockIdx.x;
    const int half_dim = head_dim / 2;
    const int i = threadIdx.x;
    if (i >= half_dim) return;

    half* row = x + static_cast<size_t>(h) * head_dim;
    const float c = cos_vals[i];
    const float s = sin_vals[i];
    const float x1 = __half2float(row[i]);
    const float x2 = __half2float(row[i + half_dim]);
    row[i] = __float2half(x1 * c - x2 * s);
    row[i + half_dim] = __float2half(x2 * c + x1 * s);
}

void launch_rope_apply(half* x, const float* cos_vals, const float* sin_vals, int num_heads, int head_dim) {
    const int half_dim = head_dim / 2;
    rope_apply_kernel<<<num_heads, half_dim>>>(x, cos_vals, sin_vals, head_dim);
    CUDA_CHECK(cudaGetLastError());
}
