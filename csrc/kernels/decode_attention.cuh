#pragma once
#include <cuda_fp16.h>

// Single-query (decode-time) attention over a KV cache, with GQA head
// mapping and online-softmax statistics (M2's FlashAttention-style
// rescaling recurrence, extended to also accumulate a running weighted-V
// sum instead of just normalizing constants).
//
// q: [num_q_heads, head_dim] half -- this step's query, already RoPE-rotated
//    (and Qwen3-style QK-normed, if applicable) by the caller; this kernel
//    only does the attention math (scores, softmax, weighted V), not
//    positional encoding or any per-head norm.
// k_cache, v_cache: [num_kv_heads, max_seq_len, head_dim] half, see
//    kv_cache.cuh -- must already include this step's token (append before
//    calling this).
// out: [num_q_heads, head_dim] half.
// cur_len: number of valid cached positions (1-indexed count, i.e. includes
//    the just-appended current token) to attend over -- causal masking for
//    decode is automatic since there ARE no future positions in the cache
//    yet, so no explicit mask is needed.
// num_q_heads must be a multiple of num_kv_heads (GQA); query head qh reads
// KV head qh / (num_q_heads / num_kv_heads), matching HF's `repeat_kv`
// grouping order.
void launch_decode_attention(const half* q, const half* k_cache, const half* v_cache, half* out, int num_q_heads,
                              int num_kv_heads, int max_seq_len, int head_dim, int cur_len, float scale);
