#include "elementwise.cuh"
#include "common.cuh"

__global__ void add_one_kernel(const float* __restrict__ in, float* __restrict__ out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] + 1.0f;
    }
}

void launch_add_one(const float* in, float* out, int64_t n) {
    const int threads = 256;
    const int64_t blocks = ceil_div(n, static_cast<int64_t>(threads));
    add_one_kernel<<<static_cast<unsigned int>(blocks), threads>>>(in, out, n);
    CUDA_CHECK(cudaGetLastError());
}
