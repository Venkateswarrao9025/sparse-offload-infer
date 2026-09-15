"""M1 acceptance: achieved bandwidth for every fundamentals kernel vs the
320 GB/s T4 peak (PROJECT_SPEC.md sec 6, M1). Run on the Colab/Kaggle T4
session, not locally (no CUDA here).

Writes reports/m1_bandwidth.csv (every row plotted) and reports/m1_bandwidth.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import time_cuda, write_csv

import soinfer.ops as ops

T4_PEAK_GB_S = 320.0
WARMUP = 20
ITERS = 100
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def bench_vector_add(rows: list[dict]) -> None:
    for n in [1 << 18, 1 << 20, 1 << 22, 1 << 24]:
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        result = time_cuda(lambda: ops.vector_add(a, b), name=f"vector_add_n{n}", warmup=WARMUP, iters=ITERS)
        n_bytes = 3 * n * 4  # read a, read b, write out
        gb_s = (n_bytes / 1e9) / (result.median_ms / 1e3)
        row = {"category": "vector_add", "param": n, "median_ms": result.median_ms, "achieved_gb_s": gb_s}
        print(row)
        rows.append(row)


def bench_strided_copy(rows: list[dict]) -> None:
    n = 1 << 18
    for stride in [1, 2, 4, 8, 16, 32, 64, 128]:
        x = torch.randn(n * stride, device="cuda", dtype=torch.float32)
        result = time_cuda(lambda: ops.strided_copy(x, stride), name=f"strided_copy_s{stride}", warmup=WARMUP, iters=ITERS)
        n_bytes = 2 * n * 4  # nominal useful bytes: read n + write n (not the scattered DRAM traffic)
        gb_s = (n_bytes / 1e9) / (result.median_ms / 1e3)
        row = {"category": "strided_copy", "param": stride, "median_ms": result.median_ms, "achieved_gb_s": gb_s}
        print(row)
        rows.append(row)


def bench_reduce(rows: list[dict]) -> None:
    n = 1 << 24
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    variants = {
        "naive_atomic": ops.reduce_naive_atomic,
        "shared_tree": ops.reduce_shared_tree,
        "warp_shuffle": ops.reduce_warp_shuffle,
        "vectorized": ops.reduce_vectorized,
    }
    for name, fn in variants.items():
        result = time_cuda(lambda: fn(x), name=f"reduce_{name}", warmup=WARMUP, iters=ITERS)
        n_bytes = n * 4  # read-only; output write is a single float
        gb_s = (n_bytes / 1e9) / (result.median_ms / 1e3)
        row = {"category": "reduce", "param": name, "median_ms": result.median_ms, "achieved_gb_s": gb_s}
        print(row)
        rows.append(row)


def bench_transpose(rows: list[dict]) -> None:
    variants = {
        "naive": ops.transpose_naive,
        "unpadded": ops.transpose_unpadded,
        "padded": ops.transpose_padded,
    }
    for n in [512, 1024, 2048, 4096]:
        x = torch.randn(n, n, device="cuda", dtype=torch.float32)
        for name, fn in variants.items():
            result = time_cuda(lambda: fn(x), name=f"transpose_{name}_n{n}", warmup=WARMUP, iters=ITERS)
            n_bytes = 2 * n * n * 4  # read n*n + write n*n
            gb_s = (n_bytes / 1e9) / (result.median_ms / 1e3)
            row = {"category": "transpose", "param": f"{name}_n{n}", "median_ms": result.median_ms, "achieved_gb_s": gb_s}
            print(row)
            rows.append(row)


def make_plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    def panel(ax, category: str, xlabel: str, title: str, log_x: bool = False):
        cat_rows = [r for r in rows if r["category"] == category]
        labels = [str(r["param"]) for r in cat_rows]
        values = [r["achieved_gb_s"] for r in cat_rows]
        ax.bar(labels, values, color="tab:blue")
        ax.axhline(T4_PEAK_GB_S, color="tab:red", linestyle="--", label=f"{T4_PEAK_GB_S:.0f} GB/s peak (T4)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("achieved GB/s")
        ax.set_title(title)
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    panel(axes[0][0], "vector_add", "n (elements)", "Vector add vs n")
    panel(axes[0][1], "strided_copy", "stride", "Strided copy: coalescing collapse")
    panel(axes[1][0], "reduce", "variant", f"Sum reduction, n=2^24")
    panel(axes[1][1], "transpose", "variant_n", "Transpose: naive vs tiled (un)padded")

    fig.suptitle("M1: achieved bandwidth vs T4 peak (320 GB/s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench_m1.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    rows: list[dict] = []
    bench_vector_add(rows)
    bench_strided_copy(rows)
    bench_reduce(rows)
    bench_transpose(rows)

    write_csv(rows, REPORTS_DIR / "m1_bandwidth.csv")
    make_plot(rows, REPORTS_DIR / "m1_bandwidth.png")


if __name__ == "__main__":
    main()
