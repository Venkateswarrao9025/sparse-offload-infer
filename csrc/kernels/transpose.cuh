#pragma once
#include <cstdint>

// Square (n x n) float32 matrix transpose, three ways. M1 only -- rectangular
// shapes are out of scope for this fundamentals demo.

// One thread per element, direct global-memory read/write. Reads are
// coalesced; writes are not (consecutive threads write n floats apart).
void launch_transpose_naive(const float* in, float* out, int n);

// Tiled through shared memory (TILE_DIM x TILE_DIM, unpadded), so both global
// reads and writes are coalesced -- but the shared-memory tile itself is read
// column-wise on the write-out step, which hits the same bank for every
// thread in a warp (32-way bank conflict).
void launch_transpose_unpadded(const float* in, float* out, int n);

// Same tiling, but the shared-memory tile is padded by one column
// (TILE_DIM x (TILE_DIM+1)) so the column-wise read step lands on distinct
// banks -- eliminates the conflict from the unpadded version.
void launch_transpose_padded(const float* in, float* out, int n);
