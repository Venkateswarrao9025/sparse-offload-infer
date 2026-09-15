#include "transpose.cuh"
#include "common.cuh"

namespace {
constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;  // each thread handles kTileDim/kBlockRows rows of the tile
}

// --- naive --------------------------------------------------------------

__global__ void transpose_naive_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        out[col * n + row] = in[row * n + col];
    }
}

void launch_transpose_naive(const float* in, float* out, int n) {
    dim3 block(32, 32);
    dim3 grid(ceil_div(n, 32), ceil_div(n, 32));
    transpose_naive_kernel<<<grid, block>>>(in, out, n);
    CUDA_CHECK(cudaGetLastError());
}

// --- tiled, unpadded (bank-conflicted) -----------------------------------

__global__ void transpose_unpadded_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    __shared__ float tile[kTileDim][kTileDim];

    int x = blockIdx.x * kTileDim + threadIdx.x;
    int y = blockIdx.y * kTileDim + threadIdx.y;
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        if (x < n && (y + j) < n) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * kTileDim + threadIdx.x;
    y = blockIdx.x * kTileDim + threadIdx.y;
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        if (x < n && (y + j) < n) {
            out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void launch_transpose_unpadded(const float* in, float* out, int n) {
    dim3 block(kTileDim, kBlockRows);
    dim3 grid(ceil_div(n, kTileDim), ceil_div(n, kTileDim));
    transpose_unpadded_kernel<<<grid, block>>>(in, out, n);
    CUDA_CHECK(cudaGetLastError());
}

// --- tiled, padded (bank-conflict-free) -----------------------------------

__global__ void transpose_padded_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    __shared__ float tile[kTileDim][kTileDim + 1];  // +1 padding breaks the bank-conflict pattern

    int x = blockIdx.x * kTileDim + threadIdx.x;
    int y = blockIdx.y * kTileDim + threadIdx.y;
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        if (x < n && (y + j) < n) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * kTileDim + threadIdx.x;
    y = blockIdx.x * kTileDim + threadIdx.y;
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        if (x < n && (y + j) < n) {
            out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void launch_transpose_padded(const float* in, float* out, int n) {
    dim3 block(kTileDim, kBlockRows);
    dim3 grid(ceil_div(n, kTileDim), ceil_div(n, kTileDim));
    transpose_padded_kernel<<<grid, block>>>(in, out, n);
    CUDA_CHECK(cudaGetLastError());
}
