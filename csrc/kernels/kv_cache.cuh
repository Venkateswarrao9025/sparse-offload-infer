#pragma once
#include <cuda_fp16.h>

// Appends one new token's K/V into a preallocated, contiguous per-layer KV
// cache at sequence position `pos`. Cache layout: [num_kv_heads, max_seq_len,
// head_dim] half, row-major -- head_dim is the innermost (contiguous) dim so
// a single head's whole history is one contiguous span, which is what
// decode_attention.cu's per-timestep dot products want. k_new/v_new:
// [num_kv_heads, head_dim] half (this step's projected K/V, already split
// per head). Paged (block-table) layout is a stretch goal noted in
// PROJECT_SPEC.md M5 task 3 -- not implemented here; this is the contiguous
// baseline.
void launch_kv_cache_append(const half* k_new, const half* v_new, half* k_cache, half* v_cache, int num_kv_heads,
                             int max_seq_len, int head_dim, int pos);
