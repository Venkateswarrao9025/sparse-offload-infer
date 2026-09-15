#pragma once
#include <cstdint>

// out[i] = in[i * stride] for i in [0, n). stride=1 is a plain coalesced
// copy; as stride grows, consecutive threads touch increasingly scattered
// cache lines and achieved bandwidth collapses even though the same number
// of "useful" bytes move -- this is the point of the M1 coalescing sweep.
void launch_strided_copy(const float* in, float* out, int64_t n, int64_t stride);
