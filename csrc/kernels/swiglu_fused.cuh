#pragma once
#include <cuda_fp16.h>

// Fused SwiGLU gate+up projection: h = silu(gate_W @ x) * (up_W @ x).
// gate_W, up_W: [I, H] half row-major (same H, may be different I per row but
// share the row count I here). x: [H] half. h: [I] half (fp32-accumulated).
// Fuses what would otherwise be two separate GEMVs (gate_proj, up_proj) plus
// an elementwise SiLU+multiply pass into one kernel: each warp reads x ONCE
// and reuses it for both dot products (instead of two independent kernels
// each re-reading all of x), and gate/up pre-activations never round-trip to
// global memory as separate [I] buffers -- only the final h is written.
// Requires H % 8 == 0 (float4-vectorized loads, same convention as
// gemv_fp16_v3). The down-projection GEMV (h -> y) is NOT fused here --
// reuse gemv_fp16_v3(down_W, h) for that; down needs the complete h vector
// before any output element can be computed, so fusing it into this same
// kernel would need a full grid-wide sync partway through, which isn't worth
// it for the modest further gain over what's fused here.
void launch_swiglu_gate_up(const half* gate_W, const half* up_W, const half* x, half* h, int I, int H);
