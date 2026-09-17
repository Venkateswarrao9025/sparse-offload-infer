#pragma once
#include <cuda_fp16.h>

// Applies RoPE (rotate-half / "NEOX-style" convention, matching HF's
// rotate_half + apply_rotary_pos_emb exactly -- see docs/LEARNING_NOTES.md's
// M5 entry) to x in place. x: [num_heads, head_dim] half. cos_vals,
// sin_vals: [head_dim / 2] float32, precomputed on the host/Python side for
// the current position as cos(pos * inv_freq[i]) / sin(pos * inv_freq[i]),
// inv_freq[i] = 1 / theta^(2i/head_dim). head_dim must be even.
//
// Derivation: HF builds cos/sin as length-head_dim by concatenating the
// length-(head_dim/2) freqs with itself, then computes
//   out = x*cos + rotate_half(x)*sin,  rotate_half(x) = cat(-x2, x1)
// where x1/x2 are x's first/second half. Since cos/sin are identical in
// both halves, this reduces to, for i in [0, head_dim/2):
//   out[i]            = x[i]*c[i] - x[i+head_dim/2]*s[i]
//   out[i+head_dim/2] = x[i+head_dim/2]*c[i] + x[i]*s[i]
// which is what this kernel computes directly (no need to materialize the
// duplicated length-head_dim cos/sin).
void launch_rope_apply(half* x, const float* cos_vals, const float* sin_vals, int num_heads, int head_dim);
