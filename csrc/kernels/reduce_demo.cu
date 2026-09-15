#include "reduce_demo.cuh"
#include "common.cuh"
#include "reduce.cuh"

namespace {
constexpr int kThreads = 256;
}

// --- v1: naive atomic ------------------------------------------------------

__global__ void reduce_naive_atomic_kernel(const float* __restrict__ in, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(out, in[i]);
    }
}

void launch_reduce_naive_atomic(const float* in, float* out, int64_t n) {
    const int64_t blocks = ceil_div(n, static_cast<int64_t>(kThreads));
    reduce_naive_atomic_kernel<<<static_cast<unsigned int>(blocks), kThreads>>>(in, out, n);
    CUDA_CHECK(cudaGetLastError());
}

// --- v2: shared-memory tree -------------------------------------------------

__global__ void reduce_shared_tree_kernel(const float* __restrict__ in, float* out, int64_t n) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

void launch_reduce_shared_tree(const float* in, float* out, int64_t n) {
    const int64_t blocks = ceil_div(n, static_cast<int64_t>(kThreads));
    const size_t shmem = kThreads * sizeof(float);
    reduce_shared_tree_kernel<<<static_cast<unsigned int>(blocks), kThreads, shmem>>>(in, out, n);
    CUDA_CHECK(cudaGetLastError());
}

// --- v3: warp-shuffle --------------------------------------------------------

__global__ void reduce_warp_shuffle_kernel(const float* __restrict__ in, float* out, int64_t n) {
    extern __shared__ float sdata[];  // (blockDim.x / 32) floats
    const int tid = threadIdx.x;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    float val = (i < n) ? in[i] : 0.0f;
    val = block_reduce_sum(val, sdata);
    if (tid == 0) {
        atomicAdd(out, val);
    }
}

void launch_reduce_warp_shuffle(const float* in, float* out, int64_t n) {
    const int64_t blocks = ceil_div(n, static_cast<int64_t>(kThreads));
    const size_t shmem = (kThreads / 32) * sizeof(float);
    reduce_warp_shuffle_kernel<<<static_cast<unsigned int>(blocks), kThreads, shmem>>>(in, out, n);
    CUDA_CHECK(cudaGetLastError());
}

// --- v4: vectorized float4 + warp-shuffle ------------------------------------

__global__ void reduce_vectorized_kernel(const float4* __restrict__ in4, float* out, int64_t n4) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    float val = 0.0f;
    if (i < n4) {
        const float4 v = in4[i];
        val = v.x + v.y + v.z + v.w;
    }
    val = block_reduce_sum(val, sdata);
    if (tid == 0) {
        atomicAdd(out, val);
    }
}

void launch_reduce_vectorized(const float* in, float* out, int64_t n) {
    const int64_t n4 = n / 4;
    const int64_t blocks = ceil_div(n4, static_cast<int64_t>(kThreads));
    const size_t shmem = (kThreads / 32) * sizeof(float);
    reduce_vectorized_kernel<<<static_cast<unsigned int>(blocks), kThreads, shmem>>>(
        reinterpret_cast<const float4*>(in), out, n4);
    CUDA_CHECK(cudaGetLastError());
}
