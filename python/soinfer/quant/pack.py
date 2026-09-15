"""Bit-packing for quantized integer tensors, matching csrc/include/layout.h
byte-for-byte -- if you change the packing scheme here, change it there too
in the same commit.

INT8 needs no packing (pack_int8/unpack_int8 exist only so both element
formats share one API). INT4 packs two values per byte, permuted into AWQ
order so a future CUDA dequant kernel can split one 32-bit load into two
16-bit masked extractions with no further shuffling (see layout.h for why).
"""
from __future__ import annotations

import torch

GROUP = 8
AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)  # nibble i of a packed group <- value at index AWQ_ORDER[i]
_INV_AWQ_ORDER = tuple(AWQ_ORDER.index(j) for j in range(GROUP))


def pack_int8(q: torch.Tensor) -> tuple[torch.Tensor, int]:
    if q.dtype != torch.int8:
        raise ValueError(f"pack_int8 expects int8, got {q.dtype}")
    return q.clone(), q.shape[-1]


def unpack_int8(packed: torch.Tensor, orig_k: int) -> torch.Tensor:
    if packed.dtype != torch.int8:
        raise ValueError(f"unpack_int8 expects int8, got {packed.dtype}")
    return packed[..., :orig_k].clone()


def pack_int4(q: torch.Tensor) -> tuple[torch.Tensor, int]:
    """q: int8, values in [-8, 7], shape [..., K]. Returns (packed uint8
    tensor of shape [..., ceil(K/8)*4], orig_k) -- see layout.h for the byte
    layout. A ragged K is zero-padded (0 packs/unpacks losslessly)."""
    if q.dtype != torch.int8:
        raise ValueError(f"pack_int4 expects int8, got {q.dtype}")
    orig_k = q.shape[-1]
    num_groups = -(-orig_k // GROUP)  # ceil div
    padded_k = num_groups * GROUP
    if padded_k != orig_k:
        pad = torch.zeros(*q.shape[:-1], padded_k - orig_k, dtype=torch.int8, device=q.device)
        q = torch.cat([q, pad], dim=-1)

    nibbles = (q.to(torch.int32) & 0xF).reshape(*q.shape[:-1], num_groups, GROUP)
    permuted = nibbles[..., list(AWQ_ORDER)]  # -> [v0,v2,v4,v6,v1,v3,v5,v7] per group
    low = permuted[..., 0::2]  # one nibble per output byte (low half)
    high = permuted[..., 1::2]  # one nibble per output byte (high half)
    packed = (low | (high << 4)).to(torch.uint8)
    return packed.reshape(*packed.shape[:-2], num_groups * (GROUP // 2)), orig_k


def unpack_int4(packed: torch.Tensor, orig_k: int) -> torch.Tensor:
    """Inverse of pack_int4. packed: uint8, shape [..., ceil(orig_k/8)*4].
    Returns int8, shape [..., orig_k], sign-extended from 4-bit two's
    complement."""
    if packed.dtype != torch.uint8:
        raise ValueError(f"unpack_int4 expects uint8, got {packed.dtype}")
    bytes_per_group = GROUP // 2
    num_groups = packed.shape[-1] // bytes_per_group
    grouped = packed.reshape(*packed.shape[:-1], num_groups, bytes_per_group).to(torch.int32)

    low = grouped & 0xF
    high = (grouped >> 4) & 0xF
    permuted = torch.stack([low, high], dim=-1).reshape(*grouped.shape[:-1], GROUP)
    permuted = torch.where(permuted >= 8, permuted - 16, permuted)  # sign-extend

    v = permuted[..., list(_INV_AWQ_ORDER)]
    v = v.reshape(*v.shape[:-2], num_groups * GROUP)
    return v[..., :orig_k].to(torch.int8)
