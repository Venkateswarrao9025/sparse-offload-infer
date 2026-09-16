#pragma once
#include <cstdint>
#include <cuda_fp16.h>

// y = W @ x, W stored as symmetric grouped INT4, AWQ-interleaved packing (see
// csrc/include/layout.h). Wq: packed uint8, row length ceil(K/8)*4 bytes ==
// ceil(K/8) uint32 words. x: [K] half, y: [N] half. scale: [N, num_groups]
// fp32, scale[row, k/group_size] is the dequant scale for element k. K is the
// true (unpadded) row length; the packed buffer's zero-padding never
// contributes since padded qweight values are exactly 0. All accumulation is
// fp32.
void launch_gemv_w4a16_group(const uint8_t* Wq, const float* scale, const half* x, half* y, int N, int K,
                              int group_size, int num_groups);

// Same numerics as launch_gemv_w4a16_group, but the int4->fp16 nibble
// extraction uses the LOP3.LUT bit-manipulation trick (construct the fp16 bit
// pattern directly instead of an int->float conversion instruction) --
// PROJECT_SPEC.md M4 task 3. Benchmark against the naive version above.
void launch_gemv_w4a16_group_lop3(const uint8_t* Wq, const float* scale, const half* x, half* y, int N, int K,
                                   int group_size, int num_groups);
