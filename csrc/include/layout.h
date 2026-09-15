#pragma once
// Quantized-weight memory layout -- SHARED SOURCE OF TRUTH between the Python
// quantization library (python/soinfer/quant/) and every CUDA dequant kernel
// from M4 onward. If you change a layout decision, change it here AND in
// python/soinfer/quant/pack.py in the same commit (PROJECT_SPEC.md sec 10).
//
// All quantization in this project is SYMMETRIC (no zero-point): a real
// tensor W is represented as W ~= scale * q, where q is a signed integer in
// [-qmax, qmax] and qmax = 2^(bits-1) - 1 (7 for 4-bit, 127 for 8-bit). The
// full negative range (-8 for 4-bit, -128 for 8-bit) is intentionally never
// produced by the quantizer, so the representable range stays symmetric
// around zero and no zero-point offset is needed.
//
// ---------------------------------------------------------------------------
// INT8 packing: none. One qweight element == one signed int8 byte. Trivial,
// listed here only so the convention is stated explicitly rather than assumed.
//
// ---------------------------------------------------------------------------
// INT4 packing: two elements per byte, in AWQ interleaved order.
//
// int4 values are grouped 8 at a time along the innermost (K) dimension. Call
// them v[0..7] in their original logical order. They are NOT packed
// sequentially (v0,v1 -> byte0; v2,v3 -> byte1; ...) -- they are packed in
// the permuted order
//
//     AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
//
// meaning nibble i (counting from the LSB of a little-endian 4-byte group) of
// the packed group holds v[AWQ_ORDER[i]]:
//
//     byte0 = v0 | (v2 << 4)
//     byte1 = v4 | (v6 << 4)
//     byte2 = v1 | (v3 << 4)
//     byte3 = v5 | (v7 << 4)
//
// Why: read as a single little-endian 32-bit word, the low half-word (bytes
// 0-1) holds the four EVEN-indexed values in original order (v0,v2,v4,v6),
// and the high half-word (bytes 2-3) holds the four ODD-indexed values in
// original order (v1,v3,v5,v7). A dequant kernel can therefore split one
// 32-bit load into two 16-bit masked extractions and convert each to a
// half4-worth of values with no further shuffling before the strided (stride
// -2) accumulation a warp already does in gemv_fp16_v2/v3 -- this is what
// "makes the LOP3 dequant trick work" (PROJECT_SPEC.md M3 task 2). The actual
// LOP3.LUT dequant kernel is written in M4; this layout is fixed now so that
// work doesn't require re-deriving (and re-testing) the packing scheme later.
//
// If K is not a multiple of 8, the last group is padded with zero-valued
// int4 entries (0 quantizes/dequantizes to exactly 0.0 under symmetric
// quantization, so padding never perturbs a real value). The true K is
// stored alongside the packed tensor; unpacking truncates back to it.
//
// ---------------------------------------------------------------------------
// Scale layout (all granularities other than per-tensor):
//
// Scales are stored row-major as a 2-D tensor of shape [N, num_groups], one
// fp32 scale per (output-channel row, group-along-K) pair:
//   - per_channel:  num_groups == 1                    (one scale per row)
//   - group / block32 / mx_e8m0:  num_groups == ceil(K / group_size)
// A kernel reads scale[row, k / group_size] for element k of that row -- keep
// the scale for the whole group in a register across the group rather than
// re-reading it per element (PROJECT_SPEC.md M4 task 2 calls this out as the
// classic performance bug).
//
// mx_e8m0 additionally constrains every stored scale to a power of two
// (OCP-style shared E8M0 micro-exponent): scale = 2^(e8m0_code - 127), with
// e8m0_code stored as an unsigned byte using the same bias-127 convention as
// an IEEE-754 float32 exponent field.

#include <cstdint>

namespace soinfer {

constexpr int kInt4GroupSize = 8;
constexpr int kAwqOrder[kInt4GroupSize] = {0, 2, 4, 6, 1, 3, 5, 7};
constexpr int kE8M0Bias = 127;

}  // namespace soinfer
