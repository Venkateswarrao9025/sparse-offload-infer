#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// M7 task 2: given a set of selected row indices (from task 1's top-k
// select), gather those rows of a matrix living in PINNED host memory into
// a DEVICE buffer, two ways to compare:
//
// - launch_gather_rows_staged: memcpy each selected row (host-side, plain
//   memory-to-memory copy -- no PCIe involved yet) into a contiguous PINNED
//   staging buffer, then ONE cudaMemcpyAsync H2D of the whole staged block.
// - launch_gather_rows_naive: one cudaMemcpyAsync per selected row, straight
//   from its (scattered) offset in the pinned arena to its slot in the
//   device buffer.
//
// The point of the comparison: every H2D transfer has fixed per-call
// overhead (descriptor setup, DMA engine queuing) on top of its per-byte
// cost, so k separate small transfers pay that overhead k times where one
// large transfer pays it once. See PROJECT_SPEC.md M7 task 2 ("compare
// against per-row cudaMemcpyAsync -- it will be far worse -- show the
// data").
//
// Both are asynchronous on the given stream -- callers must synchronize (or
// use a CUDA event) before reading gpu_dst.
//
// matrix_base: start of the matrix's rows in pinned host memory (i.e.
//   PinnedWeightStore.matrix_view(handle).data_ptr()), row-major,
//   row_nbytes apart.
// indices: HOST int64 array of k row indices, each in [0, num_rows).
// staging: PINNED host buffer, >= k * row_nbytes bytes (staged variant only).
// gpu_dst: device buffer, >= k * row_nbytes bytes.
void launch_gather_rows_staged(const uint8_t* matrix_base, const int64_t* indices, int64_t k,
                                int64_t row_nbytes, uint8_t* staging, uint8_t* gpu_dst,
                                cudaStream_t stream);

void launch_gather_rows_naive(const uint8_t* matrix_base, const int64_t* indices, int64_t k,
                               int64_t row_nbytes, uint8_t* gpu_dst, cudaStream_t stream);
