#pragma once
#include <cstdint>

// M0 toolchain smoke test: out[i] = in[i] + 1 for all i in [0, n).
void launch_add_one(const float* in, float* out, int64_t n);
