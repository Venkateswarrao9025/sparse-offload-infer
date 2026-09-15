"""Symmetric weight quantization formats: per-tensor, per-channel, group-wise,
block-32, and OCP-style microscaling (E8M0 power-of-two shared scale).

All formats are symmetric (no zero-point): W ~= scale * q, with q a signed
integer in [-qmax, qmax] and qmax = 2**(bits-1) - 1 (7 for 4-bit, 127 for
8-bit) -- the full negative code (-8 / -128) is never produced, so no
zero-point offset is needed. Quantization always runs along the last
dimension (K); a weight tensor is expected as [..., N, K].

See csrc/include/layout.h for the on-disk/on-device memory layout (packing,
scale tensor shape) every CUDA kernel from M4 onward must agree with
byte-for-byte. This module operates on *unpacked* integer values (one int8
per element, even for 4-bit) -- see pack.py for the actual bit-packing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from . import calibrate

GRANULARITIES = ("per_tensor", "per_channel", "group", "block32", "mx_e8m0")
_BLOCK32_GROUP_SIZE = 32


def qmax_for_bits(bits: int) -> int:
    return 2 ** (bits - 1) - 1


@dataclass
class QuantConfig:
    bits: int
    granularity: str
    group_size: Optional[int] = None  # required for "group"; fixed at 32 for "block32"/"mx_e8m0"

    def __post_init__(self) -> None:
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"unknown granularity {self.granularity!r}, must be one of {GRANULARITIES}")
        if self.granularity == "group" and not self.group_size:
            raise ValueError("granularity='group' requires group_size")
        if self.granularity in ("block32", "mx_e8m0"):
            self.group_size = _BLOCK32_GROUP_SIZE


@dataclass
class QuantTensor:
    qweight: torch.Tensor  # int8, values in [-qmax, qmax], shape [..., N, padded_K]
    scale: torch.Tensor  # float32, shape [..., N, num_groups] (num_groups == 1 for per_tensor/per_channel)
    config: QuantConfig
    orig_k: int  # true K before any group-padding -- dequantize() truncates back to this


def _amax_per_group(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape the last dim into zero-padded groups of group_size and return
    (grouped view [..., num_groups, group_size], amax [..., num_groups])."""
    orig_k = w.shape[-1]
    num_groups = -(-orig_k // group_size)  # ceil div
    padded_k = num_groups * group_size
    if padded_k != orig_k:
        pad = torch.zeros(*w.shape[:-1], padded_k - orig_k, dtype=w.dtype, device=w.device)
        w = torch.cat([w, pad], dim=-1)
    grouped = w.view(*w.shape[:-1], num_groups, group_size)
    return grouped, grouped.abs().amax(dim=-1)


def round_to_pow2(scale: torch.Tensor) -> torch.Tensor:
    """Round a nonnegative scale to the nearest power of two (OCP E8M0 style).
    Zero maps to zero: an all-zero group has no meaningful scale, but
    dequant multiplies by it so any value keeps those elements at exactly 0."""
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    pow2 = torch.exp2(torch.round(torch.log2(safe)))
    return torch.where(scale > 0, pow2, torch.zeros_like(scale))


def quantize(w: torch.Tensor, config: QuantConfig, scale_fn=None) -> QuantTensor:
    """Symmetric quantize w (float, [..., N, K]) per config.

    scale_fn, if given, overrides how a group's raw values are turned into a
    scale: scale_fn(grouped, amax, qmax) -> scale, with amax ==
    grouped.abs().amax(dim=-1) (so amax.shape == grouped.shape[:-1]) --
    see calibrate.py's module docstring. Defaults to calibrate.min_max.
    """
    w = w.float()
    qmax = qmax_for_bits(config.bits)
    orig_k = w.shape[-1]
    scale_fn = scale_fn or calibrate.min_max

    if config.granularity == "per_tensor":
        flat = w.reshape(-1)
        amax = flat.abs().max()
        scale = scale_fn(flat, amax, qmax).reshape(1, 1)
        qweight = torch.clamp(torch.round(w / scale), -qmax, qmax)
        return QuantTensor(qweight.to(torch.int8), scale.to(torch.float32), config, orig_k)

    if config.granularity == "per_channel":
        amax = w.abs().amax(dim=-1)  # [..., N]
        scale = scale_fn(w, amax, qmax)  # [..., N]
        qweight = torch.clamp(torch.round(w / scale.unsqueeze(-1)), -qmax, qmax)
        return QuantTensor(qweight.to(torch.int8), scale.unsqueeze(-1).to(torch.float32), config, orig_k)

    # group / block32 / mx_e8m0: grouped along K
    grouped, amax = _amax_per_group(w, config.group_size)
    scale = scale_fn(grouped, amax, qmax)
    if config.granularity == "mx_e8m0":
        scale = round_to_pow2(scale)
    q_grouped = torch.clamp(torch.round(grouped / scale.unsqueeze(-1)), -qmax, qmax)
    qweight = q_grouped.reshape(*q_grouped.shape[:-2], -1)
    return QuantTensor(qweight.to(torch.int8), scale.to(torch.float32), config, orig_k)


def dequantize(qt: QuantTensor) -> torch.Tensor:
    qweight = qt.qweight.float()
    if qt.config.granularity in ("per_tensor", "per_channel"):
        return qweight * qt.scale

    group_size = qt.config.group_size
    padded_k = qweight.shape[-1]
    num_groups = padded_k // group_size
    grouped = qweight.reshape(*qweight.shape[:-1], num_groups, group_size)
    w = (grouped * qt.scale.unsqueeze(-1)).reshape(*qweight.shape[:-1], padded_k)
    return w[..., : qt.orig_k]


def fraction_exact_zero(qt: QuantTensor) -> float:
    """Fraction of (unpadded) quantized values forced to exactly zero -- the
    metric PROJECT_SPEC.md M3 asks for to explain the per-tensor INT4
    collapse: a scale set by one outlier can push every non-outlier weight
    into round(w/scale) == 0."""
    q = qt.qweight[..., : qt.orig_k]
    return (q == 0).float().mean().item()
