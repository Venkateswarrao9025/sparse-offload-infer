"""M6 task 3: the roofline. For each weight matrix in one Qwen3-1.7B-shaped
decoder layer, compare H2D transfer time (its INT4-packed bytes, at the real
measured PCIe bandwidth from bench_pcie.py, not the spec-sheet number)
against the GEMV kernel time that consumes it, to show the offload regime is
transfer-bound -- and by how much. "This plot justifies the entire rest of
the project" (PROJECT_SPEC.md M6 task 3).

Run on the Colab/Kaggle T4 session, not locally (no CUDA here).
Writes reports/m6_roofline.csv and reports/m6_roofline.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import time_cuda, write_csv

from soinfer.offload import stream_manager, weight_store
from soinfer.quant import formats, pack
import soinfer.ops as ops

WARMUP = 20
ITERS = 100
GROUP_SIZE = 128
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Real Qwen3-1.7B config (see docs/LEARNING_NOTES.md's M5 entry for where
# these came from -- Qwen/Qwen3-1.7B's config.json, not guessed).
HIDDEN = 2048
INTERMEDIATE = 6144
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM

# One decoder layer's weight matrices, [out_features, in_features] (nn.Linear
# convention, matching what gemv_w4a16_group_lop3 expects: [N, K]).
LAYER_MATRICES = {
    "qkv_proj": (Q_DIM + 2 * KV_DIM, HIDDEN),  # fused_qkv_projection's pre-concatenated weight
    "o_proj": (HIDDEN, Q_DIM),
    "gate_proj": (INTERMEDIATE, HIDDEN),
    "up_proj": (INTERMEDIATE, HIDDEN),
    "down_proj": (HIDDEN, INTERMEDIATE),
}


def _quantize_and_pack(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    w = torch.randn(n, k)
    config = formats.QuantConfig(bits=4, granularity="group", group_size=GROUP_SIZE)
    qt = formats.quantize(w, config)
    packed, packed_k = pack.pack_int4(qt.qweight)
    return packed, qt.scale, packed_k


def bench_matrix(name: str, n: int, k: int, sm: stream_manager.StreamManager, rows: list[dict]) -> dict:
    packed, scale, packed_k = _quantize_and_pack(n, k)
    row_nbytes = packed.shape[-1]
    nbytes = packed.numel()

    store = weight_store.PinnedWeightStore(total_bytes=nbytes)
    handle = store.register(name, packed)
    host_matrix = store.matrix_view(handle)
    gpu_buf = torch.empty(n, row_nbytes, dtype=torch.uint8, device="cuda")

    def _transfer():
        sm.prefetch(0, gpu_buf, host_matrix)
        sm.synchronize(0)

    transfer_result = time_cuda(_transfer, name=f"transfer_{name}", warmup=WARMUP, iters=ITERS)
    transfer_gb_s = (nbytes / 1e9) / (transfer_result.median_ms / 1e3)

    scale_gpu = scale.cuda()
    x = torch.randn(packed_k, device="cuda", dtype=torch.float16)

    def _compute():
        ops.gemv_w4a16_group_lop3(gpu_buf, scale_gpu, x, packed_k, GROUP_SIZE)

    kernel_result = time_cuda(_compute, name=f"kernel_{name}", warmup=WARMUP, iters=ITERS)
    kernel_gb_s = (nbytes / 1e9) / (kernel_result.median_ms / 1e3)

    row = {
        "matrix": name,
        "N": n,
        "K": k,
        "packed_bytes": nbytes,
        "transfer_ms": transfer_result.median_ms,
        "transfer_gb_s": transfer_gb_s,
        "kernel_ms": kernel_result.median_ms,
        "kernel_gb_s": kernel_gb_s,
        "transfer_over_kernel_ratio": transfer_result.median_ms / kernel_result.median_ms,
    }
    print(row)
    rows.append(row)
    return row


def make_plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    names = [r["matrix"] for r in rows]
    transfer_ms = [r["transfer_ms"] for r in rows]
    kernel_ms = [r["kernel_ms"] for r in rows]

    x = range(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar([i - width / 2 for i in x], transfer_ms, width, label="H2D transfer (pinned)")
    ax.bar([i + width / 2 for i in x], kernel_ms, width, label="gemv_w4a16_group_lop3")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("time (ms)")
    ax.set_title("M6 roofline: one Qwen3-1.7B-shaped decoder layer, INT4 weights\n(transfer vs GEMV kernel time per matrix)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench_m6_roofline.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    torch.manual_seed(0)
    sm = stream_manager.StreamManager(num_buffers=2)
    rows: list[dict] = []
    for name, (n, k) in LAYER_MATRICES.items():
        bench_matrix(name, n, k, sm, rows)

    write_csv(rows, REPORTS_DIR / "m6_roofline.csv")
    make_plot(rows, REPORTS_DIR / "m6_roofline.png")

    total_transfer_ms = sum(r["transfer_ms"] for r in rows)
    total_kernel_ms = sum(r["kernel_ms"] for r in rows)
    ratio = total_transfer_ms / total_kernel_ms
    status = "TRANSFER-BOUND" if ratio > 1.0 else "COMPUTE-BOUND"
    print(
        f"[{status}] one decoder layer: {total_transfer_ms:.3f}ms transfer vs {total_kernel_ms:.3f}ms compute "
        f"-- transfer is {ratio:.2f}x the compute time"
    )


if __name__ == "__main__":
    main()
