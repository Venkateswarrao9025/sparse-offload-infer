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
//   count(|g| >= tau) >= k, then compacts into out_indices: every index
//   STRICTLY above tau first (via atomicAdd-assigned output slots -- the
//   OUTPUT ORDER among these is unspecified, which is fine: row gather
//   only needs the index SET, not an ordering), then any remaining slots
//   filled from indices tied exactly AT tau, by ascending index. The tie
//   handling is NOT measure-zero here (M8's calibration pass runs this on
//   dequantized activations, where exact float ties across channels are
//   common, unlike continuous floats) -- earlier atomicAdd-order tie
//   resolution made the returned SET depend on GPU thread-scheduling
//   order; ascending-index tie-break makes it a pure function of the
//   input again. Single block: uses one SM, not the whole GPU -- simplicity/
//   correctness-first, matching this project's v1-then-optimize pattern
//   (a grid-wide multi-block version is the natural follow-up once this
//   is verified).
//
// out_indices must have room for at least k ints. *out_count (device
// int) is set to the number actually written (== k, unless k > n).
void launch_topk_threshold_select(const float* abs_g, int n, int k, int* out_indices, int* out_count);
