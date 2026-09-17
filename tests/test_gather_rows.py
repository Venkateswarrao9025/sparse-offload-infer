"""M7 task 2 (PROJECT_SPEC.md sec 6, M7): row gather from a pinned host
arena into device memory, two variants (staged-then-single-copy vs
per-row cudaMemcpyAsync) that must agree byte-for-byte with each other
and with a plain CPU index_select reference.

Skipped entirely on machines without a CUDA GPU; run for real on the
Colab/Kaggle T4 session via `make test`.
"""
import pytest
import torch

soinfer = pytest.importorskip("soinfer")
if soinfer.ops is None:
    pytest.skip("soinfer._C (CUDA extension) not built", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="M7 kernels require a CUDA GPU")


def _make_matrix(num_rows: int, row_nbytes: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 256, (num_rows, row_nbytes), dtype=torch.uint8).pin_memory()


@pytest.mark.parametrize("num_rows,row_nbytes,k", [
    (17408, 2560, 8704),  # Qwen3-14B down_proj row shape (row_nbytes = ceil(5120/8)*4 INT4-packed), k=0.5*I
    (1024, 64, 1),
    (1024, 64, 1024),     # k == num_rows (gather everything)
    (256, 8, 37),          # small, odd k
])
def test_gather_rows_staged_matches_reference(num_rows, row_nbytes, k):
    matrix = _make_matrix(num_rows, row_nbytes)
    torch.manual_seed(1)
    indices = torch.randperm(num_rows)[:k].contiguous()

    staging = torch.empty(k, row_nbytes, dtype=torch.uint8).pin_memory()
    gpu_dst = torch.empty(k, row_nbytes, dtype=torch.uint8, device="cuda")

    soinfer.ops.gather_rows_staged(matrix, indices, staging, gpu_dst)
    torch.cuda.synchronize()

    expected = matrix.index_select(0, indices)
    assert torch.equal(gpu_dst.cpu(), expected)


@pytest.mark.parametrize("num_rows,row_nbytes,k", [
    (17408, 2560, 8704),
    (1024, 64, 1),
    (1024, 64, 1024),
    (256, 8, 37),
])
def test_gather_rows_naive_matches_reference(num_rows, row_nbytes, k):
    matrix = _make_matrix(num_rows, row_nbytes)
    torch.manual_seed(1)
    indices = torch.randperm(num_rows)[:k].contiguous()

    gpu_dst = torch.empty(k, row_nbytes, dtype=torch.uint8, device="cuda")
    soinfer.ops.gather_rows_naive(matrix, indices, gpu_dst)
    torch.cuda.synchronize()

    expected = matrix.index_select(0, indices)
    assert torch.equal(gpu_dst.cpu(), expected)


def test_gather_rows_staged_and_naive_agree():
    """Both variants must produce byte-identical output given the same input --
    the whole point is they differ only in transfer pattern, not in result."""
    matrix = _make_matrix(4096, 320)
    torch.manual_seed(2)
    indices = torch.randperm(4096)[:2048].contiguous()

    staging = torch.empty(2048, 320, dtype=torch.uint8).pin_memory()
    dst_staged = torch.empty(2048, 320, dtype=torch.uint8, device="cuda")
    dst_naive = torch.empty(2048, 320, dtype=torch.uint8, device="cuda")

    soinfer.ops.gather_rows_staged(matrix, indices, staging, dst_staged)
    soinfer.ops.gather_rows_naive(matrix, indices, dst_naive)
    torch.cuda.synchronize()

    assert torch.equal(dst_staged, dst_naive)


def test_gather_rows_rejects_unpinned_matrix():
    matrix = torch.randint(0, 256, (256, 8), dtype=torch.uint8)  # NOT pinned
    indices = torch.arange(10, dtype=torch.int64)
    gpu_dst = torch.empty(10, 8, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError):
        soinfer.ops.gather_rows_naive(matrix, indices, gpu_dst)


def test_gather_rows_rejects_undersized_staging():
    matrix = _make_matrix(256, 8)
    indices = torch.arange(10, dtype=torch.int64)
    staging = torch.empty(5, 8, dtype=torch.uint8).pin_memory()  # too small for k=10
    gpu_dst = torch.empty(10, 8, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError):
        soinfer.ops.gather_rows_staged(matrix, indices, staging, gpu_dst)
