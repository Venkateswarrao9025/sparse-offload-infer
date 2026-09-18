#pragma once
#include <cuda_fp16.h>

// M7 task 1: given |g| (magnitudes of the SwiGLU gate activation, size I,
// non-negative), selects the k indices with the largest values. Two
// variants:
//
// - launch_topk_threshold_select: single-kernel-launch threshold+count+
//   compact. Binary-searches (in fp32, within ONE block -- no host
//   round-trips between iterations, since M7's whole point is that
//   selection must cost less than the transfer it saves, and a
//   host-driven per-iteration launch loop would blow the microsecond
//   budget on launch overhead alone) for a threshold tau such that
//   count(|g| >= tau) >= k, then writes out_indices in a single
//   ASCENDING-INDEX-ORDER scan: every qualifying index (> tau
//   unconditionally, == tau until k slots are filled -- ties are NOT
//   measure-zero here, since M8's calibration pass runs this on
//   dequantized activations where exact float ties across channels are
//   common, unlike continuous floats). The full output is a pure,
//   deterministic function of the input -- both the returned SET and its
//   ORDER. An earlier version compacted the >tau majority via
//   atomicAdd-assigned output slots, reasoning the order among them was
//   "harmless" since row gather only needs the index set, not an
//   ordering -- true for the gather itself, but false for what happens
//   next: gather order becomes summation order in
//   gemv_w4a16_sparse_accumulate downstream, and float summation is not
//   order-independent. Same SET, different order, different rounded
//   result -- invisible to count-based tests, caught building M8's
//   ablation table on real hardware (two calls on a frozen input gave the
//   identical selected set but different final perplexity through a real
//   28-layer decode loop). Single block, and Phase 3 is now
//   single-threaded on top of that: uses one SM, not the whole GPU, and a
//   known further-suboptimal step within it -- simplicity/correctness-first
//   over performance, matching this project's v1-then-optimize pattern (a
//   grid-wide multi-block redesign is the natural follow-up, already
//   flagged as not started before this fix and unaffected by it).
//
// out_indices must have room for at least k ints. *out_count (device
// int) is set to the number actually written (== k, unless k > n).
void launch_topk_threshold_select(const float* abs_g, int n, int k, int* out_indices, int* out_count);
