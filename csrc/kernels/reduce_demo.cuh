#pragma once
#include <cstdint>

// Four versions of sum-reduction over `in` (length n), each progressively
// less naive. *out must point to a single float that the caller has already
// zero-initialized -- every variant here accumulates into it via atomicAdd.

// v1: every thread atomically adds its own element straight to *out.
// Maximum contention; the baseline everything else is measured against.
void launch_reduce_naive_atomic(const float* in, float* out, int64_t n);

// v2: per-block shared-memory tree reduction, one atomicAdd per block.
void launch_reduce_shared_tree(const float* in, float* out, int64_t n);

// v3: per-block warp-shuffle reduction (see reduce.cuh), one atomicAdd per block.
void launch_reduce_warp_shuffle(const float* in, float* out, int64_t n);

// v4: same as v3 but each thread loads a float4 (4 elements) per iteration
// before reducing. Requires n % 4 == 0 (caller's responsibility -- this is a
// bandwidth demo, not a general-purpose kernel).
void launch_reduce_vectorized(const float* in, float* out, int64_t n);
