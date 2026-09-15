#pragma once
// Shared helpers for all soinfer kernels: error checking, arch guards, ceil_div.
//
// Target GPU is NVIDIA T4 (Turing, sm_75): no native bf16, no cp.async.
// Any Ampere+-only code path MUST be gated like this:
//
//   #if __CUDA_ARCH__ >= 800
//       ... bf16 / cp.async path ...
//   #else
//       ... fp16 fallback (the sm_75 path) ...
//   #endif

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t _soinfer_err = (expr);                                               \
        if (_soinfer_err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__,   \
                    cudaGetErrorString(_soinfer_err));                                   \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

template <typename T>
__host__ __device__ inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}
