#pragma once
#include <cstdint>
#include <cuda_fp16.h>

// M8 task 3, the "centerpiece": for each of the k channels M7 task 1's
// top-k selection picked, resolve whether that channel's row is already
// GPU-resident (a "hot" channel, M8 task 2's cache) or needs to come from
// a freshly host-gathered staging buffer (M7 task 2), and run the GEMV
// against whichever source holds it -- IN ONE KERNEL, no host-side
// branching, no second launch (PROJECT_SPEC.md M8 task 3).
//
// NOT YET HARDWARE-VERIFIED as of the commit that adds this file (no GPU
// access this session -- see docs/LEARNING_NOTES.md's M8 task 3 entry).
// Written and reasoned through carefully, matching M4/M7's existing GEMV
// kernels' numerics exactly, but treat it as unverified until a real
// `make test` run confirms it.
//
// Two-step pipeline:
//
// 1. launch_build_dip_descriptors: given the k selected channel indices
//    (topk_threshold_select's output, M7 task 1) and a per-channel
//    slot_of[I] array (-1 if channel c is not cached, else its resident
//    cache slot -- built once when M8 task 2's HotCache is populated),
//    for each selected channel EITHER encodes its cache slot into a
//    descriptor OR assigns it a compacted position in a MISS list. The
//    caller then only needs to gather_rows_staged (M7 task 2) the MISS
//    channels -- strictly fewer than k whenever the cache has any hits at
//    all, which is the actual byte-savings-per-token M8 task 3/4 adds
//    over M7 task 4's baseline (which gathers all k every time,
//    regardless of caching).
//
//    Descriptor encoding (int32): bit 31 = source (0 = cache, 1 =
//    staging), bits [0:30] = offset within that source's row buffer (a
//    cache slot index, or a compacted position in the miss list).
//
// 2. launch_gemv_dip_fused_up / _down: same numerics as
//    gemv_w4a16_group_lop3 (M4) / gemv_w4a16_sparse_accumulate (M7 task
//    3) respectively, but each row's packed-weight/scale base pointer is
//    resolved from ITS OWN descriptor via a branchless ternary on plain
//    pointer arithmetic (`from_staging ? staging_base + off : cache_base
//    + off`) instead of a fixed base pointer shared by every row --
//    PROJECT_SPEC.md M8 task 3's "consider a predicated pointer select
//    rather than a branch to avoid warp divergence." A ternary this
//    simple (no side effects on either arm) is the standard way to hint
//    nvcc toward a predicated `selp` instead of a divergent branch, but
//    that has NOT been confirmed by inspecting the generated SASS or by
//    an Nsight Compute warp-efficiency profile -- that confirmation is
//    M9 work (Nsight Compute pass on every hot kernel), not done here.
void launch_build_dip_descriptors(const int32_t* selected_indices, int k, const int32_t* slot_of,
                                   int32_t* out_descriptors, int32_t* out_miss_channels, int32_t* out_miss_count);

// "up" direction: y[k] = W_selected @ x, one warp per selected row (same
// parallelization as gemv_w4a16_group_lop3), row source resolved per-row
// via its descriptor. cache_Wq/cache_scale and staging_Wq/staging_scale
// are two separate [*, row_nbytes]/[*, num_groups] buffers; a descriptor
// with source=0 indexes into the cache buffers, source=1 into staging.
void launch_gemv_dip_fused_up(const uint8_t* cache_Wq, const float* cache_scale, const uint8_t* staging_Wq,
                               const float* staging_scale, const int32_t* descriptors, const half* x, half* y, int k,
                               int K, int group_size, int num_groups);

// "down" direction: y[H] = sum_i h[i] * W_T_selected[i,:], one thread per
// output column (same parallelization as gemv_w4a16_sparse_accumulate),
// each iteration's row source resolved via its descriptor.
void launch_gemv_dip_fused_down(const uint8_t* cache_Wq, const float* cache_scale, const uint8_t* staging_Wq,
                                 const float* staging_scale, const int32_t* descriptors, const half* h_selected,
                                 half* y, int H, int k, int group_size, int num_groups);
