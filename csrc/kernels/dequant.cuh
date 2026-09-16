#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// Fast INT4 -> FP16 dequant for one AWQ-order-packed 32-bit word (8 signed
// values, see csrc/include/layout.h), built entirely from FP16 bit-pattern
// construction (mask/OR/half2 arithmetic) instead of int->float conversion
// instructions -- PROJECT_SPEC.md M4 task 3.
//
// This is the standard AWQ/FasterTransformer dequantize_s4_to_fp16x2 trick,
// adapted for OUR packing convention. That published trick treats each
// nibble as UNSIGNED (0..15) and recovers a signed value via a flat `-8`
// (i.e. it assumes a zero-point-8 asymmetric code). Our nibbles are instead
// plain two's-complement signed 4-bit (layout.h): sign_extend(n) = n<8 ? n :
// n-16. The two conventions agree after flipping each nibble's sign bit
// first: sign_extend(n) == (n ^ 0x8) - 8 for all n in [0,15] (check the two
// branches: n=0b1000 -> xor 0b0000 -> 0-8=-8 == sign_extend(8)=8-16=-8; n=0
// -> xor 0b1000=8 -> 8-8=0 == sign_extend(0)=0). So we XOR the whole packed
// word with 0x88888888 (every nibble's bit 3) up front, then run the
// unmodified published bit-trick, then subtract 8 from every lane.
//
// Bit layout worked out from layout.h's AWQ_ORDER = [0,2,4,6,1,3,5,7]:
// nibble n0..n7 (at bit offsets 0,4,...,28 of the packed word) hold
// v0,v2,v4,v6,v1,v3,v5,v7. The trick below groups (n0,n4), (n1,n5), (n2,n6),
// (n3,n7) into four half2 lanes, which -- after AWQ_ORDER's interleaving --
// works out to (v0,v1), (v2,v3), (v4,v5), (v6,v7): four CONSECUTIVE pairs
// along K, needing no further shuffling before pairing with a contiguous
// float4 load of x (this is the payoff layout.h calls out).
__device__ __forceinline__ void dequant_int4x8_awq(uint32_t packed, half2 out[4]) {
    const uint32_t x = packed ^ 0x88888888u;
    constexpr uint32_t kBottomMask = 0x000f000fu;
    constexpr uint32_t kTopMask = 0x00f000f0u;
    constexpr uint32_t kMagic = 0x64006400u;  // half2{1024, 1024}
    const uint32_t top = x >> 8;

    uint32_t h0 = (x & kBottomMask) | kMagic;
    uint32_t h1 = (x & kTopMask) | kMagic;
    uint32_t h2 = (top & kBottomMask) | kMagic;
    uint32_t h3 = (top & kTopMask) | kMagic;

    const half2 magic1024 = __floats2half2_rn(1024.f, 1024.f);
    const half2 sixteenth = __floats2half2_rn(1.f / 16.f, 1.f / 16.f);
    const half2 neg64 = __floats2half2_rn(-64.f, -64.f);
    const half2 eight = __floats2half2_rn(8.f, 8.f);

    // h1/h3 carry their nibble in bits [4:7]/[20:23] of the fp16 mantissa
    // (i.e. scaled by 16 relative to h0/h2), so they need the fma-by-1/16
    // rescale instead of a plain subtract.
    half2 g0 = __hsub2(*reinterpret_cast<half2*>(&h0), magic1024);
    half2 g1 = __hfma2(*reinterpret_cast<half2*>(&h1), sixteenth, neg64);
    half2 g2 = __hsub2(*reinterpret_cast<half2*>(&h2), magic1024);
    half2 g3 = __hfma2(*reinterpret_cast<half2*>(&h3), sixteenth, neg64);

    out[0] = __hsub2(g0, eight);  // {v0, v1} == k+0, k+1
    out[1] = __hsub2(g1, eight);  // {v2, v3} == k+2, k+3
    out[2] = __hsub2(g2, eight);  // {v4, v5} == k+4, k+5
    out[3] = __hsub2(g3, eight);  // {v6, v7} == k+6, k+7
}
