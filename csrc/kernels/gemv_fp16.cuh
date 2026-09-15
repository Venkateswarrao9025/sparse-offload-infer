#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// y = W @ x. W: [N, K] row-major half, x: [K] half, y: [N] half (v1-v3) or
// [N] fp32 accumulator (v4, caller casts to half). All accumulate in fp32.

// v1: one thread per output row, sequential dot product. Baseline.
void launch_gemv_fp16_v1(const half* W, const half* x, half* y, int N, int K);

// v2: one warp per output row, strided dot product + warp-shuffle reduce.
void launch_gemv_fp16_v2(const half* W, const half* x, half* y, int N, int K);

// v3: same as v2 but with float4-vectorized loads (8 half elements per
// iteration instead of 1). Requires K % 8 == 0.
void launch_gemv_fp16_v3(const half* W, const half* x, half* y, int N, int K);

// v4: split-K. Each block computes a partial dot product over one K-chunk
// for one row, then atomicAdds into y_accum (fp32, caller must zero it
// first). Adds parallelism when N is too small to saturate the GPU on its
// own (tall-skinny: small N, huge K).
void launch_gemv_fp16_v4_splitk(const half* W, const half* x, float* y_accum, int N, int K, int split);
