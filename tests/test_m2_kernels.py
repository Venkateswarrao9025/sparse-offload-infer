"""M2 acceptance: RMSNorm, online softmax, and FP16 GEMV match PyTorch
references within max abs error < 1e-2 (PROJECT_SPEC.md sec 6, M2).

Skipped entirely on machines without a CUDA GPU; run for real on the
Colab/Kaggle T4 session via `make test`.
"""
import pytest
import torch

soinfer = pytest.importorskip("soinfer")
if soinfer.ops is None:
    pytest.skip("soinfer._C (CUDA extension) not built", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="M2 kernels require a CUDA GPU")

MAX_ABS_ERR = 1e-2

# GEMV output magnitude scales as sqrt(K) for unit-variance random inputs
# (std ~64 at K=4096, ~256 at K=65536). At that magnitude a single FP16 ULP
# is already 0.016-0.25 -- bigger than MAX_ABS_ERR -- so two independently
# -computed reductions (the kernel's summation order vs PyTorch's) can round
# to adjacent FP16 values with no actual error. GEMV comparisons use a
# magnitude-scaled bound (same shape as torch.allclose) instead; RMSNorm and
# softmax outputs stay near unit magnitude so the flat MAX_ABS_ERR is fine
# for them as-is.
GEMV_RTOL = 5e-3


def max_abs_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def assert_gemv_matches(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    diff = (actual.float() - expected.float()).abs()
    bound = MAX_ABS_ERR + GEMV_RTOL * expected.float().abs()
    assert torch.all(diff < bound), (
        f"{label} exceeds scaled tolerance: max diff {diff.max().item()} "
        f"at bound {bound[diff.argmax()].item()}"
    )


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    variance = x32.pow(2).mean(-1, keepdim=True)
    normed = x32 * torch.rsqrt(variance + eps)
    return (normed * weight.float()).half()


def test_rmsnorm_matches_reference():
    rows, hidden = 8, 2048
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float16)
    eps = 1e-6
    expected = rmsnorm_ref(x, weight, eps)
    actual = soinfer.ops.rmsnorm(x, weight, eps)
    err = max_abs_err(actual, expected)
    assert err < MAX_ABS_ERR, f"rmsnorm max abs error {err} >= {MAX_ABS_ERR}"


@pytest.mark.parametrize("softmax_fn", [soinfer.ops.softmax_twopass, soinfer.ops.softmax_online])
def test_softmax_matches_reference(softmax_fn):
    rows, cols = 8, 4096
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float16) * 5.0  # wider range to exercise stability
    expected = torch.softmax(x.float(), dim=-1).half()
    actual = softmax_fn(x)
    err = max_abs_err(actual, expected)
    assert err < MAX_ABS_ERR, f"{softmax_fn.__name__} max abs error {err} >= {MAX_ABS_ERR}"


def test_softmax_twopass_and_online_agree():
    rows, cols = 8, 4096
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float16) * 5.0
    a = soinfer.ops.softmax_twopass(x)
    b = soinfer.ops.softmax_online(x)
    err = max_abs_err(a, b)
    assert err < MAX_ABS_ERR, f"twopass vs online disagree: max abs error {err} >= {MAX_ABS_ERR}"


@pytest.mark.parametrize(
    "gemv_fn",
    [soinfer.ops.gemv_fp16_v1, soinfer.ops.gemv_fp16_v2, soinfer.ops.gemv_fp16_v3],
)
def test_gemv_variants_match_reference(gemv_fn):
    N, K = 512, 4096  # K % 8 == 0 so v3 is included
    W = torch.randn(N, K, device="cuda", dtype=torch.float16)
    x = torch.randn(K, device="cuda", dtype=torch.float16)
    expected = torch.mv(W.float(), x.float()).half()
    actual = gemv_fn(W, x)
    assert_gemv_matches(actual, expected, gemv_fn.__name__)


@pytest.mark.parametrize("gemv_fn", [soinfer.ops.gemv_fp16_v1, soinfer.ops.gemv_fp16_v2])
def test_gemv_v1_v2_handle_k_not_multiple_of_8(gemv_fn):
    N, K = 64, 4097  # not a multiple of 8 -- v3 excludes this shape by design
    W = torch.randn(N, K, device="cuda", dtype=torch.float16)
    x = torch.randn(K, device="cuda", dtype=torch.float16)
    expected = torch.mv(W.float(), x.float()).half()
    actual = gemv_fn(W, x)
    assert_gemv_matches(actual, expected, gemv_fn.__name__)


def test_gemv_v3_rejects_k_not_multiple_of_8():
    W = torch.randn(16, 4097, device="cuda", dtype=torch.float16)
    x = torch.randn(4097, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        soinfer.ops.gemv_fp16_v3(W, x)


@pytest.mark.parametrize("split", [1, 4, 8])
def test_gemv_v4_splitk_matches_reference(split):
    N, K = 64, 65536  # tall-skinny: small N, huge K -- the shape split-K targets
    W = torch.randn(N, K, device="cuda", dtype=torch.float16)
    x = torch.randn(K, device="cuda", dtype=torch.float16)
    expected = torch.mv(W.float(), x.float()).half()
    actual = soinfer.ops.gemv_fp16_v4_splitk(W, x, split)
    assert_gemv_matches(actual, expected, f"gemv_fp16_v4_splitk(split={split})")
