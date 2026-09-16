"""M4 acceptance (PROJECT_SPEC.md sec 6, M4): W8A16/W4A16 GEMV numerics match
the M3 Python fake-quant reference (soinfer.quant) exactly in the
dequantized values, and within 1e-2 in the GEMV output.

Skipped entirely on machines without a CUDA GPU; run for real on the
Colab/Kaggle T4 session via `make test`.
"""
import pytest
import torch

from soinfer.quant import formats, pack

soinfer = pytest.importorskip("soinfer")
if soinfer.ops is None:
    pytest.skip("soinfer._C (CUDA extension) not built", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="M4 kernels require a CUDA GPU")

# See tests/test_m2_kernels.py for why GEMV comparisons use a magnitude-scaled
# bound instead of a flat absolute tolerance.
MAX_ABS_ERR = 1e-2
GEMV_RTOL = 5e-3


def assert_gemv_matches(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    diff = (actual.float() - expected.float()).abs()
    bound = MAX_ABS_ERR + GEMV_RTOL * expected.float().abs()
    assert torch.all(diff < bound), (
        f"{label} exceeds scaled tolerance: max diff {diff.max().item()} " f"at bound {bound[diff.argmax()].item()}"
    )


def _quantize(bits: int, W: torch.Tensor, granularity: str, group_size) -> formats.QuantTensor:
    config = formats.QuantConfig(bits=bits, granularity=granularity, group_size=group_size)
    return formats.quantize(W.float(), config)


# ---------------------------------------------------------------------------
# W8A16
# ---------------------------------------------------------------------------

W8A16_GRANULARITIES = [("per_tensor", None), ("per_channel", None), ("group", 128)]


@pytest.mark.parametrize("granularity,group_size", W8A16_GRANULARITIES)
def test_w8a16_dequant_exact_via_basis_vectors(granularity, group_size):
    """gemv(Wq, scale, e_k) == dequant(Wq,scale)[:,k] exactly: a one-hot probe
    has only one nonzero term in the dot product, so fp32 accumulation order
    can't perturb it -- this isolates dequant correctness from summation
    error (the latter is covered, with tolerance, below)."""
    torch.manual_seed(0)
    N, K = 16, 256
    W = torch.randn(N, K) * 3.0
    qt = _quantize(8, W, granularity, group_size)
    Wq, _ = pack.pack_int8(qt.qweight)
    Wq = Wq.cuda()
    scale = qt.scale.cuda()
    gsize = group_size or K
    dequant_ref = formats.dequantize(qt).cuda().half()

    x = torch.eye(K, device="cuda", dtype=torch.float16)
    for k in range(0, K, 37):
        y = soinfer.ops.gemv_w8a16(Wq, scale, x[k], gsize)
        assert torch.equal(y, dequant_ref[:, k]), f"w8a16[{granularity}] mismatch at column {k}"


@pytest.mark.parametrize("granularity,group_size", W8A16_GRANULARITIES)
def test_w8a16_gemv_matches_reference(granularity, group_size):
    torch.manual_seed(1)
    N, K = 512, 4096
    W = torch.randn(N, K)
    qt = _quantize(8, W, granularity, group_size)
    Wq, _ = pack.pack_int8(qt.qweight)
    Wq = Wq.cuda()
    scale = qt.scale.cuda()
    gsize = group_size or K
    W_dequant = formats.dequantize(qt).cuda()

    x = torch.randn(K, device="cuda", dtype=torch.float16)
    expected = torch.mv(W_dequant.float(), x.float()).half()
    actual = soinfer.ops.gemv_w8a16(Wq, scale, x, gsize)
    assert_gemv_matches(actual, expected, f"gemv_w8a16[{granularity}]")


def test_w8a16_rejects_misaligned_group_size():
    W = torch.randn(4, 64)
    qt = _quantize(8, W, "group", 3)  # not a multiple of 4, num_groups > 1
    Wq, _ = pack.pack_int8(qt.qweight)
    Wq = Wq.cuda()
    scale = qt.scale.cuda()
    x = torch.randn(64, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        soinfer.ops.gemv_w8a16(Wq, scale, x, 3)


# ---------------------------------------------------------------------------
# W4A16 (naive scalar dequant, and the LOP3 bit-pattern-construction dequant)
# ---------------------------------------------------------------------------

W4A16_FNS = ["gemv_w4a16_group", "gemv_w4a16_group_lop3"]


@pytest.mark.parametrize("gemv_fn_name", W4A16_FNS)
@pytest.mark.parametrize("K", [256, 4096])
def test_w4a16_dequant_exact_via_basis_vectors(gemv_fn_name, K):
    torch.manual_seed(2)
    N, group_size = 16, 128
    W = torch.randn(N, K) * 3.0
    qt = _quantize(4, W, "group", group_size)
    Wq_packed, orig_k = pack.pack_int4(qt.qweight)
    assert orig_k == K
    Wq_packed = Wq_packed.cuda()
    scale = qt.scale.cuda()
    dequant_ref = formats.dequantize(qt).cuda().half()

    gemv_fn = getattr(soinfer.ops, gemv_fn_name)
    x = torch.eye(K, device="cuda", dtype=torch.float16)
    for k in range(0, K, 41):
        y = gemv_fn(Wq_packed, scale, x[k], K, group_size)
        assert torch.equal(y, dequant_ref[:, k]), f"{gemv_fn_name}[K={K}] mismatch at column {k}"


@pytest.mark.parametrize("gemv_fn_name", W4A16_FNS)
@pytest.mark.parametrize("K", [4096, 4099])  # 4099: not a multiple of group_size (128) or 8
def test_w4a16_gemv_matches_reference(gemv_fn_name, K):
    torch.manual_seed(3)
    N, group_size = 512, 128
    W = torch.randn(N, K)
    qt = _quantize(4, W, "group", group_size)
    # formats.quantize's "group" granularity zero-pads K up to a multiple of
    # group_size *before* pack_int4 ever sees it (see formats._amax_per_group)
    # -- so what pack_int4 packs, and what our kernel's `K` must describe, is
    # qt.qweight.shape[-1] (packed_k), not the true pre-quant K. The kernel's
    # own K is a separate, smaller padding (up to a multiple of 8) on top of
    # that; passing the true K here (4099) instead of packed_k (4224) is
    # exactly the shape mismatch gemv_w4a16_group's TORCH_CHECK exists to
    # catch, which is what caught this on first run against real hardware.
    Wq_packed, packed_k = pack.pack_int4(qt.qweight)
    Wq_packed = Wq_packed.cuda()
    scale = qt.scale.cuda()
    W_dequant = formats.dequantize(qt).cuda()  # truncated back to true K

    # x is sized to packed_k (what the kernel call requires); only the first
    # K elements matter for the dot product since dequantized weight columns
    # K..packed_k-1 are exactly zero (group-padding), so slicing x to K
    # before the reference matmul is mathematically equivalent regardless of
    # what's in the padding tail.
    x = torch.randn(packed_k, device="cuda", dtype=torch.float16)
    expected = torch.mv(W_dequant.float(), x[:K].float()).half()
    gemv_fn = getattr(soinfer.ops, gemv_fn_name)
    actual = gemv_fn(Wq_packed, scale, x, packed_k, group_size)
    assert_gemv_matches(actual, expected, f"{gemv_fn_name}[K={K}]")


def test_w4a16_naive_and_lop3_agree():
    """The two dequant paths are two implementations of the same numerics
    (see csrc/kernels/dequant.cuh) -- they should agree bit-for-bit, not just
    within tolerance."""
    torch.manual_seed(4)
    N, K, group_size = 256, 4096, 128
    W = torch.randn(N, K)
    qt = _quantize(4, W, "group", group_size)
    Wq_packed, _ = pack.pack_int4(qt.qweight)
    Wq_packed = Wq_packed.cuda()
    scale = qt.scale.cuda()
    x = torch.randn(K, device="cuda", dtype=torch.float16)

    a = soinfer.ops.gemv_w4a16_group(Wq_packed, scale, x, K, group_size)
    b = soinfer.ops.gemv_w4a16_group_lop3(Wq_packed, scale, x, K, group_size)
    assert torch.equal(a, b)


def test_w4a16_rejects_group_size_not_multiple_of_8():
    W = torch.randn(4, 64)
    qt = _quantize(4, W, "group", 32)
    Wq_packed, _ = pack.pack_int4(qt.qweight)
    Wq_packed = Wq_packed.cuda()
    scale = qt.scale.cuda()
    x = torch.randn(64, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        soinfer.ops.gemv_w4a16_group(Wq_packed, scale, x, 64, 7)
