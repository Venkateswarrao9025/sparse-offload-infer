"""M1 acceptance: CUDA fundamentals kernels match PyTorch references.

Skipped entirely on machines without a CUDA GPU; run for real on the
Colab/Kaggle T4 session via `make test`.

Tolerance notes:
- vector_add, strided_copy, transpose_*: exact operations (no reduction, no
  rounding-order sensitivity), so atol=0/rtol=0 against the PyTorch reference.
- reduce_*: these sum the same values in different orders (naive atomic vs.
  tree vs. warp-shuffle vs. vectorized), and float32 addition is not
  associative, so exact match is not expected. The reference is computed in
  float64 to avoid the reference itself being a low-precision comparison
  point; rtol=1e-3 comfortably covers the reordering error at n=2**20
  (fp32 epsilon ~1.2e-7, accumulated over ~sqrt(n) ~ 1000 additions).
"""
import pytest
import torch

soinfer = pytest.importorskip("soinfer")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="M1 kernels require a CUDA GPU")


def test_vector_add_matches_reference():
    a = torch.randn(1 << 20, device="cuda", dtype=torch.float32)
    b = torch.randn(1 << 20, device="cuda", dtype=torch.float32)
    expected = a + b
    actual = soinfer.ops.vector_add(a, b)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("stride", [1, 2, 4, 8, 16, 32])
def test_strided_copy_matches_reference(stride):
    n = 1 << 16
    x = torch.randn(n * stride, device="cuda", dtype=torch.float32)
    expected = x[: n * stride : stride]
    actual = soinfer.ops.strided_copy(x, stride)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "reduce_fn",
    [
        soinfer.ops.reduce_naive_atomic,
        soinfer.ops.reduce_shared_tree,
        soinfer.ops.reduce_warp_shuffle,
        soinfer.ops.reduce_vectorized,
    ],
)
def test_reduce_variants_match_reference(reduce_fn):
    n = 1_000_000  # divisible by 4 (for reduce_vectorized), not by 256 (exercises tail masking)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    expected = x.double().sum().float()
    actual = reduce_fn(x).squeeze()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize(
    "transpose_fn",
    [
        soinfer.ops.transpose_naive,
        soinfer.ops.transpose_unpadded,
        soinfer.ops.transpose_padded,
    ],
)
@pytest.mark.parametrize("n", [33, 257, 1024])  # 33/257: not tile-aligned; 1024: tile-aligned
def test_transpose_variants_match_reference(transpose_fn, n):
    x = torch.randn(n, n, device="cuda", dtype=torch.float32)
    expected = x.t().contiguous()
    actual = transpose_fn(x)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
