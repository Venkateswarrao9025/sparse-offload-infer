#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// RMSNorm over the last dim: out[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2) + eps) * weight.
// x: [rows, hidden], weight: [hidden], out: [rows, hidden], all half.
// hidden must be even (half2 vectorization). Sum-of-squares is accumulated in
// fp32 to avoid the accumulation error a pure-fp16 reduction would pick up.
void launch_rmsnorm(const half* x, const half* weight, half* out, int rows, int hidden, float eps);
