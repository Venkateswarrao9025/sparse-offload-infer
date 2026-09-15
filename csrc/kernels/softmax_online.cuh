#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// Row-wise softmax over the last dim. x, out: [rows, cols], half.
// Accumulates in fp32 internally.

// v1: naive two-pass statistics -- one pass for the row max, a second pass
// for sum(exp(x - max)), then a third pass to write the output.
void launch_softmax_twopass(const half* x, half* out, int rows, int cols);

// v2: online single-pass statistics -- max and sum are computed together via
// the FlashAttention rescaling recurrence, so only one pass over the row is
// needed to get both, before the (unavoidable) output-writing pass.
void launch_softmax_online(const half* x, half* out, int rows, int cols);
