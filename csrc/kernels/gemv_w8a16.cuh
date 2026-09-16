#pragma once
#include <cstdint>
#include <cuda_fp16.h>

// y = W @ x, W stored as symmetric INT8 (W ~= scale * qweight, see
// csrc/include/layout.h). W: [N, K] int8 row-major, x: [K] half, y: [N] half.
// scale: [num_scale_rows, num_groups] fp32, where num_scale_rows is either N
// (per_channel/group) or 1 (per_tensor -- caller passes scale_row_stride=0 to
// broadcast row 0 to every output row). scale[row, k/group_size] is the
// dequant scale for element k of that row. All accumulation is fp32.
void launch_gemv_w8a16(const int8_t* Wq, const float* scale, const half* x, half* y, int N, int K,
                        int group_size, int num_groups, int scale_row_stride);
