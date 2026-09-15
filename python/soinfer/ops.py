"""Thin typed wrappers over the soinfer CUDA extension (soinfer._C)."""
import torch

from . import _C


def add_one(x: torch.Tensor) -> torch.Tensor:
    """out[i] = x[i] + 1. float32 CUDA tensors only. M0 toolchain smoke test."""
    return _C.add_one(x)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """out = a + b, elementwise. float32 CUDA tensors only. M1."""
    return _C.vector_add(a, b)


def strided_copy(x: torch.Tensor, stride: int) -> torch.Tensor:
    """out[i] = x[i*stride] for i in [0, len(x)//stride). M1 coalescing sweep."""
    return _C.strided_copy(x, stride)


def reduce_naive_atomic(x: torch.Tensor) -> torch.Tensor:
    """Sum of x via v1: every thread atomicAdd's its element to the output. M1."""
    return _C.reduce_naive_atomic(x)


def reduce_shared_tree(x: torch.Tensor) -> torch.Tensor:
    """Sum of x via v2: per-block shared-memory tree reduction. M1."""
    return _C.reduce_shared_tree(x)


def reduce_warp_shuffle(x: torch.Tensor) -> torch.Tensor:
    """Sum of x via v3: per-block warp-shuffle reduction. M1."""
    return _C.reduce_warp_shuffle(x)


def reduce_vectorized(x: torch.Tensor) -> torch.Tensor:
    """Sum of x via v4: vectorized float4 loads + warp-shuffle. numel must be a multiple of 4. M1."""
    return _C.reduce_vectorized(x)


def transpose_naive(x: torch.Tensor) -> torch.Tensor:
    """Square matrix transpose v1: naive, one thread per element. M1."""
    return _C.transpose_naive(x)


def transpose_unpadded(x: torch.Tensor) -> torch.Tensor:
    """Square matrix transpose v2: shared-memory tiled, bank-conflicted. M1."""
    return _C.transpose_unpadded(x)


def transpose_padded(x: torch.Tensor) -> torch.Tensor:
    """Square matrix transpose v3: shared-memory tiled, padded to avoid bank conflicts. M1."""
    return _C.transpose_padded(x)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dim. x: [rows, hidden] half, weight: [hidden] half. hidden must be even. M2."""
    return _C.rmsnorm(x, weight, eps)


def softmax_twopass(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax v1: naive two-pass statistics (max pass, then sum pass). x: [rows, cols] half. M2."""
    return _C.softmax_twopass(x)


def softmax_online(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax v2: online single-pass statistics (FlashAttention rescaling recurrence). M2."""
    return _C.softmax_online(x)


def gemv_fp16_v1(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y = W @ x. v1: one thread per output row. W: [N,K] half, x: [K] half. M2."""
    return _C.gemv_fp16_v1(W, x)


def gemv_fp16_v2(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y = W @ x. v2: one warp per row, warp-shuffle reduce. M2."""
    return _C.gemv_fp16_v2(W, x)


def gemv_fp16_v3(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y = W @ x. v3: v2 + float4-vectorized loads. Requires K % 8 == 0. M2."""
    return _C.gemv_fp16_v3(W, x)


def gemv_fp16_v4_splitk(W: torch.Tensor, x: torch.Tensor, split: int) -> torch.Tensor:
    """y = W @ x. v4: split-K with atomics -- helps when N is too small to saturate the GPU. M2."""
    return _C.gemv_fp16_v4_splitk(W, x, split)
