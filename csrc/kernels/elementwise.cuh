#pragma once
#include <cstdint>

// M0 toolchain smoke test: out[i] = in[i] + 1 for all i in [0, n).
void launch_add_one(const float* in, float* out, int64_t n);

// M1 memory-bandwidth warm-up: out[i] = a[i] + b[i] for all i in [0, n).
void launch_vector_add(const float* a, const float* b, float* out, int64_t n);
