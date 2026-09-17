"""M7 acceptance (PROJECT_SPEC.md sec 6, M7): top-k selection, verified
against PyTorch's own top-k (the "full sort baseline" M7 task 1 asks to
compare against -- torch.topk is backed by an already-optimized GPU sort/
select, a fair reference rather than a hand-rolled naive sort).

Skipped entirely on machines without a CUDA GPU; run for real on the
Colab/Kaggle T4 session via `make test`.
"""
import pytest
import torch

soinfer = pytest.importorskip("soinfer")
if soinfer.ops is None:
    pytest.skip("soinfer._C (CUDA extension) not built", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="M7 kernels require a CUDA GPU")


def _reference_topk_indices(abs_g: torch.Tensor, k: int) -> set[int]:
    return set(torch.topk(abs_g, k).indices.tolist())


@pytest.mark.parametrize("n,k", [
    (11008, 5504),   # k = 0.5*I, the spec's own microsecond-budget benchmark point
    (17408, 8704),   # Qwen3-14B's actual intermediate_size, k=0.5*I
    (1024, 1),
    (1024, 1024),    # k == n (select everything)
    (4096, 1),
])
def test_topk_threshold_select_matches_reference_set(n, k):
    torch.manual_seed(0)
    abs_g = torch.rand(n, device="cuda", dtype=torch.float32) * 10.0  # distinct-with-probability-1 continuous values

    actual_idx = soinfer.ops.topk_threshold_select(abs_g, k)
    assert actual_idx.shape == (k,)
    assert actual_idx.dtype == torch.int32

    actual_set = set(actual_idx.tolist())
    assert len(actual_set) == k, "selected indices must be unique"

    expected_set = _reference_topk_indices(abs_g, k)
    assert actual_set == expected_set, (
        f"n={n} k={k}: selected set differs from torch.topk's reference set "
        f"(symmetric diff size {len(actual_set ^ expected_set)})"
    )

    # every selected value must be >= every unselected value (the actual
    # top-k property, independent of exactly which reference implementation
    # is used to check it)
    selected_vals = abs_g[actual_idx.long()]
    mask = torch.ones(n, dtype=torch.bool, device="cuda")
    mask[actual_idx.long()] = False
    if mask.any():
        assert selected_vals.min() >= abs_g[mask].max(), "selected set is not actually the top-k by value"


def test_topk_threshold_select_handles_many_exact_ties():
    """Ties are measure-zero for continuous data, but real gate activations
    can still collide in float32 -- confirm the kernel doesn't crash and
    still returns exactly k valid, in-range indices when ties are common."""
    torch.manual_seed(1)
    n, k = 8192, 2048
    abs_g = torch.randint(0, 10, (n,), device="cuda").float()  # heavy tie clustering: only 10 distinct values

    idx = soinfer.ops.topk_threshold_select(abs_g, k)
    assert idx.shape == (k,)
    assert len(set(idx.tolist())) == k
    assert idx.min().item() >= 0 and idx.max().item() < n


def test_topk_threshold_select_rejects_k_out_of_range():
    abs_g = torch.rand(1024, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        soinfer.ops.topk_threshold_select(abs_g, 0)
    with pytest.raises(RuntimeError):
        soinfer.ops.topk_threshold_select(abs_g, 1025)


def test_topk_threshold_select_rejects_non_1d():
    abs_g = torch.rand(4, 256, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        soinfer.ops.topk_threshold_select(abs_g, 10)
