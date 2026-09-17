#pragma once
#include <cstdint>
#include <cuda_fp16.h>

// M7 task 3, "down" direction: y = sum_{i in S} h[i] * W_T[i, :], where
// W_T is down_proj STORED TRANSPOSED ([intermediate_size, hidden_size]
// instead of nn.Linear's usual [hidden_size, intermediate_size]) so that
// "restrict to the selected channel set S" is a ROW selection -- the same
// primitive M7 task 1 (topk_select) and task 2 (gather_rows) already
// produce, reused unchanged for down_proj by storing it transposed.
//
// This is deliberately NOT the same access pattern as gemv_w4a16_group
// (one row per output, reduce ALONG K): here there is no "output row" at
// all, since S's rows collectively define ONE H-length output vector.
// Instead: one thread per OUTPUT COLUMN c, looping over the (already
// row-gathered, compact) k selected rows, accumulating h[i] * W_T[i, c]
// into a register. Consecutive threads (consecutive c) read the SAME
// packed word for i's within one int4-group-of-8, so this is a coalesced
// broadcast-style read, not the lane-strided pattern gemv_w4a16_group
// uses -- the two kernels are transposes of each other's parallelization,
// matching the transposed shape of the actual math they compute.
//
// Wq_selected: packed uint8, k rows already gathered (task 2's
//   gather_rows_staged/naive) from down_proj^T's pinned arena, row length
//   ceil(H/8)*4 bytes (== ceil(H/8) uint32 words), AWQ-order INT4 (see
//   layout.h/dequant.cuh -- same packing as every other W4A16 kernel).
// scale_selected: [k, num_groups] fp32, num_groups == ceil(H/group_size),
//   already gathered/compacted to match Wq_selected's row order.
// h_selected: [k] half -- the (already elementwise silu(gate)*up) values
//   at the k selected intermediate-channel indices, same order as the
//   gathered rows.
// y: [H] half output. Fully overwritten (not accumulated into) -- every
//   thread owns exactly one output column and writes it once.
// "up" (task 3's other GEMV) needs no new kernel: it's the existing
// gemv_w4a16_group_lop3 (csrc/kernels/gemv_w4a16_group.cuh) applied to
// up_proj's k gathered rows with N=k -- up_proj is already row-indexed by
// intermediate channel, so restricting to S is exactly what task 2's
// row gather already produces.
void launch_gemv_w4a16_sparse_accumulate(const uint8_t* Wq_selected, const float* scale_selected,
                                          const half* h_selected, half* y, int H, int k, int group_size,
                                          int num_groups);
