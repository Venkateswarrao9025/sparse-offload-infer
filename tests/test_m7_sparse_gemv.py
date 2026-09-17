"""M7 task 3 (PROJECT_SPEC.md sec 6, M7): sparse GEMVs restricted to a
selected index set. "up" needs no new kernel -- it's gemv_w4a16_group_lop3
(M4) applied to up_proj's already row-gathered k rows with N=k, since
up_proj is naturally row-indexed by intermediate channel. This file tests
the one genuinely new kernel, gemv_w4a16_sparse_accumulate ("down"
direction, transposed-storage accumulate), plus an end-to-end pipeline
test tying task 1 (topk_select) + task 3 (both GEMVs) together against a
masked-dense reference.

Skipped entirely on machines without a CUDA GPU; run for real on the
Colab/Kaggle T4 session via `make test`.
"""
import torch
import torch.nn.functional as F
import pytest

from soinfer.quant import formats, pack

soinfer = pytest.importorskip("soinfer")
if soinfer.ops is None:
    pytest.skip("soinfer._C (CUDA extension) not built", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="M7 kernels require a CUDA GPU")

# See tests/test_m4_kernels.py for why GEMV comparisons use a magnitude-scaled bound.
MAX_ABS_ERR = 1e-2
GEMV_RTOL = 5e-3


def assert_gemv_matches(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    diff = (actual.float() - expected.float()).abs()
    bound = MAX_ABS_ERR + GEMV_RTOL * expected.float().abs()
    assert torch.all(diff < bound), (
        f"{label} exceeds scaled tolerance: max diff {diff.max().item()} at bound {bound[diff.argmax()].item()}"
    )


def _quantize_pack(W: torch.Tensor, group_size: int):
    qt = formats.quantize(W.float(), formats.QuantConfig(bits=4, granularity="group", group_size=group_size))
    Wq_packed, _ = pack.pack_int4(qt.qweight)
    return Wq_packed.cuda(), qt.scale.cuda(), qt


def test_sparse_accumulate_dequant_exact_via_one_hot():
    """h_selected is a one-hot vector (1.0 at row i0, 0 elsewhere): only row
    i0's dequantized values should reach y, exactly (fp32 accumulation of
    exact-zero terms can't perturb the one nonzero term) -- isolates the
    new kernel's dequant/indexing from any summation error, mirroring
    test_m4_kernels.py's basis-vector pattern."""
    torch.manual_seed(0)
    k, H, group_size = 20, 256, 64
    W_T = torch.randn(k, H) * 3.0
    Wq, scale, qt = _quantize_pack(W_T, group_size)
    dequant_ref = formats.dequantize(qt).cuda().half()  # [k, H]

    for i0 in range(0, k, 3):
        h_selected = torch.zeros(k, device="cuda", dtype=torch.float16)
        h_selected[i0] = 1.0
        y = soinfer.ops.gemv_w4a16_sparse_accumulate(Wq, scale, h_selected, H, group_size)
        assert torch.equal(y, dequant_ref[i0]), f"one-hot at row {i0} mismatch"


@pytest.mark.parametrize("k,H,group_size", [(64, 512, 128), (37, 256, 64)])  # 37: odd k, not a power of 2
def test_sparse_accumulate_matches_reference(k, H, group_size):
    torch.manual_seed(1)
    W_T = torch.randn(k, H)
    Wq, scale, qt = _quantize_pack(W_T, group_size)
    W_dequant = formats.dequantize(qt).cuda()  # [k, H]

    h_selected = torch.randn(k, device="cuda", dtype=torch.float16)
    expected = (h_selected.float() @ W_dequant.float()).half()  # [H]
    actual = soinfer.ops.gemv_w4a16_sparse_accumulate(Wq, scale, h_selected, H, group_size)
    assert_gemv_matches(actual, expected, f"sparse_accumulate[k={k},H={H}]")


def test_sparse_mlp_pipeline_matches_masked_dense_reference():
    """Ties task 1 (topk_select) and task 3 (both sparse GEMVs) together:
    compute a full dense MLP forward (gate/up dense, both directions using
    the existing M4 GEMV), select the top-k channels by |gate|, and check
    that running the SPARSE pipeline on just those k channels matches a
    dense down_proj forward with the non-selected channels masked to zero
    -- the mathematical definition of what DIP approximates (PROJECT_SPEC.md
    M7's opening paragraph)."""
    torch.manual_seed(2)
    H, I, group_size, k = 256, 512, 64, 256  # k = 0.5*I

    x = torch.randn(H, device="cuda", dtype=torch.float16)
    gate_Wq, gate_scale, _ = _quantize_pack(torch.randn(I, H), group_size)
    up_Wq, up_scale, _ = _quantize_pack(torch.randn(I, H), group_size)
    downT_Wq, downT_scale, downT_qt = _quantize_pack(torch.randn(I, H), group_size)  # down_proj, TRANSPOSED [I, H]

    gate_out = soinfer.ops.gemv_w4a16_group_lop3(gate_Wq, gate_scale, x, H, group_size)  # [I]
    up_out = soinfer.ops.gemv_w4a16_group_lop3(up_Wq, up_scale, x, H, group_size)  # [I]
    h_full = F.silu(gate_out.float()) * up_out.float()  # [I]

    abs_g = gate_out.float().abs().contiguous()
    idx = soinfer.ops.topk_threshold_select(abs_g, k).long()  # [k], UNORDERED (atomicAdd race order)

    downT_dequant = formats.dequantize(downT_qt).cuda().float()  # [I, H]

    # Reference: masked-dense is mathematically h_masked @ downT_dequant summed
    # over all I channels (zero outside idx) -- but summed in NATURAL index
    # order over I=512 terms. Our kernel instead accumulates over exactly the
    # k selected terms in idx's (unordered) gather order. Both are valid fp32
    # summations of the same term SET, but floating-point addition isn't
    # associative, and at this test's output magnitudes (H2D-free GEMV over
    # random weights, values in the thousands with sign cancellation across
    # k~256 terms) different summation order alone produces few-ULP-scale
    # final differences after the fp16 cast -- not a kernel bug. Selecting
    # to idx's order first (rather than masking+full-length-matmul) removes
    # the larger, avoidable part of that order mismatch (k terms vs I
    # terms, many structurally zero) while keeping the reference computed
    # by torch's own matmul, independent of our kernel.
    expected = h_full.index_select(0, idx).float() @ downT_dequant.index_select(0, idx)  # [H]

    # Sparse pipeline: gather (plain index_select here -- task 2's actual
    # host-pinned-arena gather is tested separately in test_gather_rows.py;
    # this test isolates task 3's GEMV math given an already-gathered set).
    up_Wq_sel, up_scale_sel = up_Wq.index_select(0, idx), up_scale.index_select(0, idx)
    u_selected = soinfer.ops.gemv_w4a16_group_lop3(up_Wq_sel, up_scale_sel, x, H, group_size)  # [k]: "up" sparse GEMV
    g_selected = gate_out.index_select(0, idx)
    h_selected = (F.silu(g_selected.float()) * u_selected.float()).half()  # [k]

    downT_Wq_sel = downT_Wq.index_select(0, idx)
    downT_scale_sel = downT_scale.index_select(0, idx)
    y = soinfer.ops.gemv_w4a16_sparse_accumulate(downT_Wq_sel, downT_scale_sel, h_selected, H, group_size)  # [H]

    # Wider tolerance than assert_gemv_matches's default: this compares two
    # differently-ordered k~256-term fp32 summations (see comment above),
    # not a single kernel against a fixed-order reference of the same
    # arithmetic -- some extra few-ULP spread at large magnitudes is expected.
    diff = (y.float() - expected.float()).abs()
    bound = 0.5 + 2e-2 * expected.float().abs()
    assert torch.all(diff < bound), (
        f"sparse_mlp_pipeline exceeds scaled tolerance: max diff {diff.max().item()} "
        f"at bound {bound[diff.argmax()].item()}"
    )


def test_sparse_accumulate_rejects_shape_mismatches():
    k, H, group_size = 16, 128, 32
    Wq, scale, _ = _quantize_pack(torch.randn(k, H), group_size)
    h_selected = torch.randn(k, device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError):  # wrong H
        soinfer.ops.gemv_w4a16_sparse_accumulate(Wq, scale, h_selected, H + 8, group_size)
    with pytest.raises(RuntimeError):  # h_selected wrong length
        soinfer.ops.gemv_w4a16_sparse_accumulate(Wq, scale, h_selected[:-1], H, group_size)
    with pytest.raises(RuntimeError):  # scale wrong num rows
        soinfer.ops.gemv_w4a16_sparse_accumulate(Wq, scale[:-1], h_selected, H, group_size)
    with pytest.raises(RuntimeError):  # group_size not a multiple of 8
        soinfer.ops.gemv_w4a16_sparse_accumulate(Wq, scale, h_selected, H, 5)
