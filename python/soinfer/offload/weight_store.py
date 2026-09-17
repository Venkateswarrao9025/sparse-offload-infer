"""M6 task 1: a pinned host arena holding quantized weight matrices, laid out
row-major with rows as the addressable unit.

"Rows as the addressable unit" (PROJECT_SPEC.md M6 task 1) means: every
matrix's row is a single contiguous span of bytes at a fixed, computable
offset, so a future caller (M7's DIP row gather) can copy an arbitrary
*subset* of a matrix's rows without touching the rest -- the arena is one
big pinned allocation, but nothing about its layout assumes whole-matrix
transfers are the only thing that will ever happen to it. Deciding this now
(rather than when M7 needs it) is exactly what the spec calls out.

A "row" here is whatever soinfer.quant.pack already produced for one
output channel: for W8A16, row_nbytes == K (one byte per element, see
pack.pack_int8); for W4A16-group, row_nbytes == ceil(K/8)*4 (see
pack.pack_int4). This module is deliberately format-agnostic -- it stores
opaque byte rows and leaves interpreting them to the GEMV kernels that
already exist (M4).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WeightHandle:
    """Identifies one matrix's rows within a PinnedWeightStore's arena.
    Opaque to callers other than via the store's own methods -- offset is
    an implementation detail, not something to do arithmetic on directly."""

    name: str
    offset: int  # byte offset into the arena
    num_rows: int
    row_nbytes: int

    @property
    def nbytes(self) -> int:
        return self.num_rows * self.row_nbytes


class PinnedWeightStore:
    """A single pinned host allocation holding one or more quantized weight
    matrices, row-major. `total_bytes` must be known upfront (sum of every
    matrix's num_rows * row_nbytes to be registered) -- pinned memory is
    expensive to allocate (it's physically locked, unswappable), so growing
    it dynamically (many small cudaHostAlloc calls) is exactly the pattern
    to avoid; one allocation sized for the whole model is the point.
    """

    def __init__(self, total_bytes: int):
        if total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        self.arena = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
        self._write_offset = 0
        self._handles: dict[str, WeightHandle] = {}

    def register(self, name: str, data: torch.Tensor) -> WeightHandle:
        """Copies `data` ([num_rows, row_nbytes], uint8, any device) into
        the next free span of the arena and returns a handle for later
        row/matrix access. Registration order determines arena layout;
        register everything once at model-load time, not per token."""
        if name in self._handles:
            raise ValueError(f"weight {name!r} already registered")
        if data.dtype != torch.uint8 or data.dim() != 2:
            raise ValueError(f"register({name!r}): data must be 2D uint8 [num_rows, row_nbytes], got {data.shape} {data.dtype}")

        num_rows, row_nbytes = data.shape
        nbytes = num_rows * row_nbytes
        end_offset = self._write_offset + nbytes
        if end_offset > self.arena.numel():
            raise ValueError(
                f"register({name!r}): arena overflow -- {nbytes} bytes requested, "
                f"{self.arena.numel() - self._write_offset} remain"
            )

        dst = self.arena[self._write_offset:end_offset].view(num_rows, row_nbytes)
        dst.copy_(data)
        handle = WeightHandle(name, self._write_offset, num_rows, row_nbytes)
        self._handles[name] = handle
        self._write_offset = end_offset
        return handle

    def handle(self, name: str) -> WeightHandle:
        return self._handles[name]

    def matrix_view(self, handle: WeightHandle) -> torch.Tensor:
        """The full matrix as [num_rows, row_nbytes] uint8, a view into the
        pinned arena (no copy)."""
        return self.arena[handle.offset : handle.offset + handle.nbytes].view(handle.num_rows, handle.row_nbytes)

    def row_view(self, handle: WeightHandle, row: int) -> torch.Tensor:
        """One row as [row_nbytes] uint8, a view into the pinned arena (no
        copy). This is the primitive M7's row gather will call in a loop
        over a selected index set."""
        if not 0 <= row < handle.num_rows:
            raise IndexError(f"row {row} out of range for {handle.name!r} (num_rows={handle.num_rows})")
        start = handle.offset + row * handle.row_nbytes
        return self.arena[start : start + handle.row_nbytes]

    def rows_view(self, handle: WeightHandle, rows: torch.Tensor) -> torch.Tensor:
        """Selected rows, gathered into a NEW contiguous tensor (unlike
        row_view/matrix_view, this one copies -- there's no way to view a
        non-contiguous row subset of the arena as a single tensor without
        one). `rows`: 1D int64 tensor of row indices. This is the naive
        reference gather M7 explicitly wants to benchmark against a
        coalesced version -- see PROJECT_SPEC.md M7 task 2."""
        matrix = self.matrix_view(handle)
        return matrix.index_select(0, rows).contiguous()

    @property
    def bytes_used(self) -> int:
        return self._write_offset

    @property
    def bytes_total(self) -> int:
        return self.arena.numel()
