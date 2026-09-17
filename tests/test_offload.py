"""M6 acceptance (PROJECT_SPEC.md sec 6, M6): pinned host arena with
row-addressable layout (task 1), and CUDA-stream double/triple buffering
that actually overlaps a transfer with compute (task 2). This file covers
correctness; bench/bench_m6_roofline.py covers the actual timing claim
("transfer-bound, and by how much").

Skipped entirely on machines without a CUDA GPU; run for real on the
Colab/Kaggle T4 session via `make test`. Pinned memory allocation itself
needs a CUDA context, so soinfer.offload's classes can't be exercised at
all without a GPU -- unlike soinfer.quant, which is pure Python/PyTorch.
"""
import pytest
import torch

soinfer = pytest.importorskip("soinfer")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory requires a CUDA GPU")

from soinfer.offload import stream_manager, weight_store  # noqa: E402


# ---------------------------------------------------------------------------
# PinnedWeightStore
# ---------------------------------------------------------------------------


def test_register_and_matrix_view_roundtrip():
    store = weight_store.PinnedWeightStore(total_bytes=1024)
    data = torch.randint(0, 256, (16, 32), dtype=torch.uint8)
    handle = store.register("layer0.gate_proj", data)

    assert handle.num_rows == 16 and handle.row_nbytes == 32
    assert store.handle("layer0.gate_proj") is handle
    assert torch.equal(store.matrix_view(handle).cpu(), data)
    assert store.arena.is_pinned()


def test_row_view_matches_matrix_row():
    store = weight_store.PinnedWeightStore(total_bytes=1024)
    data = torch.randint(0, 256, (16, 32), dtype=torch.uint8)
    handle = store.register("W", data)

    for row in [0, 5, 15]:
        assert torch.equal(store.row_view(handle, row).cpu(), data[row])


def test_row_view_rejects_out_of_range():
    store = weight_store.PinnedWeightStore(total_bytes=1024)
    handle = store.register("W", torch.zeros(4, 8, dtype=torch.uint8))
    with pytest.raises(IndexError):
        store.row_view(handle, 4)


def test_rows_view_gathers_selected_rows():
    store = weight_store.PinnedWeightStore(total_bytes=1024)
    data = torch.arange(16 * 32, dtype=torch.uint8).reshape(16, 32) % 251  # avoid uint8 overflow wraparound surprises
    handle = store.register("W", data)

    rows = torch.tensor([2, 0, 9])
    gathered = store.rows_view(handle, rows)
    assert torch.equal(gathered.cpu(), data[rows])


def test_register_two_matrices_are_independently_addressable():
    a = torch.randint(0, 256, (8, 16), dtype=torch.uint8)
    b = torch.randint(0, 256, (4, 64), dtype=torch.uint8)
    store = weight_store.PinnedWeightStore(total_bytes=a.numel() + b.numel())
    ha = store.register("a", a)
    hb = store.register("b", b)

    assert torch.equal(store.matrix_view(ha).cpu(), a)
    assert torch.equal(store.matrix_view(hb).cpu(), b)
    assert store.bytes_used == a.numel() + b.numel()


def test_register_rejects_overflow():
    store = weight_store.PinnedWeightStore(total_bytes=16)
    store.register("a", torch.zeros(1, 16, dtype=torch.uint8))
    with pytest.raises(ValueError):
        store.register("b", torch.zeros(1, 1, dtype=torch.uint8))


def test_register_rejects_duplicate_name():
    store = weight_store.PinnedWeightStore(total_bytes=64)
    store.register("a", torch.zeros(1, 16, dtype=torch.uint8))
    with pytest.raises(ValueError):
        store.register("a", torch.zeros(1, 16, dtype=torch.uint8))


# ---------------------------------------------------------------------------
# StreamManager
# ---------------------------------------------------------------------------


def test_prefetch_and_wait_transfers_correct_data():
    store = weight_store.PinnedWeightStore(total_bytes=4096)
    data = torch.randint(0, 256, (32, 128), dtype=torch.uint8)
    handle = store.register("W", data)

    sm = stream_manager.StreamManager(num_buffers=2)
    gpu_buf = torch.empty(handle.num_rows, handle.row_nbytes, dtype=torch.uint8, device="cuda")
    sm.prefetch(0, gpu_buf, store.matrix_view(handle))
    sm.wait(0)  # order the default stream after the copy
    torch.cuda.synchronize()

    assert torch.equal(gpu_buf.cpu(), data)


def test_double_buffering_pipeline_matches_reference():
    """Simulates the actual M6 usage pattern: N "layers" of pinned weights,
    prefetch layer i+1 while "computing" (here: just reading) layer i, using
    2 alternating GPU buffers. Checks every layer's GPU-side data matches
    its host source, in order -- this is what would catch a buffer-reuse
    race (reading buffer A while its next prefetch is still in flight)."""
    num_layers = 5
    rows, row_nbytes = 16, 64
    host_layers = [torch.randint(0, 256, (rows, row_nbytes), dtype=torch.uint8).pin_memory() for _ in range(num_layers)]

    sm = stream_manager.StreamManager(num_buffers=2)
    gpu_buffers = [torch.empty(rows, row_nbytes, dtype=torch.uint8, device="cuda") for _ in range(2)]

    sm.prefetch(0, gpu_buffers[0], host_layers[0])
    results = []
    for i in range(num_layers):
        cur = i % 2
        nxt = (i + 1) % 2
        sm.wait(cur)
        if i + 1 < num_layers:
            sm.prefetch(nxt, gpu_buffers[nxt], host_layers[i + 1])
        results.append(gpu_buffers[cur].clone())  # clone: snapshot before the buffer is reused 2 iterations later
    torch.cuda.synchronize()

    for i in range(num_layers):
        assert torch.equal(results[i].cpu(), host_layers[i]), f"layer {i} mismatch"


def test_prefetch_rejects_unpinned_host_data():
    sm = stream_manager.StreamManager(num_buffers=1)
    gpu_buf = torch.empty(4, 8, dtype=torch.uint8, device="cuda")
    unpinned = torch.zeros(4, 8, dtype=torch.uint8)  # not .pin_memory()
    with pytest.raises(ValueError):
        sm.prefetch(0, gpu_buf, unpinned)
