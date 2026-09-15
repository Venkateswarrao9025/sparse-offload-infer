#include "strided_copy.cuh"
#include "common.cuh"

__global__ void strided_copy_kernel(const float* __restrict__ in, float* __restrict__ out, int64_t n,
                                     int64_t stride) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i * stride];
    }
}

void launch_strided_copy(const float* in, float* out, int64_t n, int64_t stride) {
    const int threads = 256;
    const int64_t blocks = ceil_div(n, static_cast<int64_t>(threads));
    strided_copy_kernel<<<static_cast<unsigned int>(blocks), threads>>>(in, out, n, stride);
    CUDA_CHECK(cudaGetLastError());
}
