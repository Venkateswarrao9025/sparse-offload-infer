"""Thin typed wrappers over the soinfer CUDA extension (soinfer._C)."""
import torch

from . import _C


def add_one(x: torch.Tensor) -> torch.Tensor:
    """out[i] = x[i] + 1. float32 CUDA tensors only. M0 toolchain smoke test."""
    return _C.add_one(x)
