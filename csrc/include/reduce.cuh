#pragma once
// Shared reduction primitives, reused by M1's reduction demo and M2's
// RMSNorm/softmax.

#include <cuda_runtime.h>

// Sums `val` across a warp via shuffle. Only lane 0's return value is the
// correct total; other lanes hold partial/garbage sums.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

// Sums `val` across every thread in the block. Every thread must call this
// (it contains a __syncthreads()). `shared` must point to at least
// ceil(blockDim.x / 32) floats of shared memory. Only thread 0's return
// value is the block total.
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}
