"""M0 acceptance: the CUDA toolchain builds and produces correct numbers.

Skipped entirely on machines without a CUDA GPU (e.g. local dev on this repo);
run for real on the Colab/Kaggle T4 session via `make test`.
"""
import pytest
import torch

soinfer = pytest.importorskip("soinfer")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="M0 add_one kernel requires a CUDA GPU")


def test_add_one_matches_reference():
    x = torch.randn(1 << 20, device="cuda", dtype=torch.float32)
    expected = x + 1.0
    actual = soinfer.ops.add_one(x)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_add_one_rejects_non_cuda_input():
    x = torch.randn(16, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        soinfer.ops.add_one(x)
