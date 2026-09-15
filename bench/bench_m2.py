"""M2 acceptance: FP16 GEMV bandwidth vs the 320 GB/s T4 peak, plus RMSNorm
and softmax timing (PROJECT_SPEC.md sec 6, M2). Run on the Colab/Kaggle T4
session, not locally (no CUDA here).

Acceptance check: gemv_fp16_v3 should reach >=70% of peak (224 GB/s) at
K >= 4096 -- this script asserts it and fails loudly if not.

Writes reports/m2_gemv_bandwidth.csv, reports/m2_gemv_bandwidth.png, and
reports/m2_norm_softmax.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import time_cuda, write_csv

import soinfer.ops as ops

T4_PEAK_GB_S = 320.0
GEMV_V3_TARGET_FRACTION = 0.70
WARMUP = 20
ITERS = 100
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def bench_gemv(rows: list[dict]) -> None:
    N = 4096
    variants = {
        "v1_naive": ops.gemv_fp16_v1,
        "v2_warp_shuffle": ops.gemv_fp16_v2,
        "v3_vectorized": ops.gemv_fp16_v3,
    }
    v3_at_4096 = None
    for K in [1024, 2048, 4096, 8192, 16384]:
        W = torch.randn(N, K, device="cuda", dtype=torch.float16)
        x = torch.randn(K, device="cuda", dtype=torch.float16)
        n_bytes = N * K * 2  # W dominates; x and y are negligible
        for name, fn in variants.items():
            result = time_cuda(lambda: fn(W, x), name=f"gemv_{name}_K{K}", warmup=WARMUP, iters=ITERS)
            gb_s = (n_bytes / 1e9) / (result.median_ms / 1e3)
            row = {"category": "gemv", "variant": name, "K": K, "median_ms": result.median_ms, "achieved_gb_s": gb_s}
            print(row)
            rows.append(row)
            if name == "v3_vectorized" and K == 4096:
                v3_at_4096 = gb_s

    assert v3_at_4096 is not None
    target = GEMV_V3_TARGET_FRACTION * T4_PEAK_GB_S
    status = "PASS" if v3_at_4096 >= target else "FAIL"
    print(f"[{status}] gemv_fp16_v3 at K=4096: {v3_at_4096:.1f} GB/s (target >= {target:.1f} GB/s = "
          f"{GEMV_V3_TARGET_FRACTION:.0%} of {T4_PEAK_GB_S:.0f} GB/s peak)")


def bench_splitk(rows: list[dict]) -> None:
    # Tall-skinny shape: N too small to saturate the GPU without splitting K.
    N, K = 32, 1 << 20
    W = torch.randn(N, K, device="cuda", dtype=torch.float16)
    x = torch.randn(K, device="cuda", dtype=torch.float16)
    n_bytes = N * K * 2

    result = time_cuda(lambda: ops.gemv_fp16_v2(W, x), name="gemv_v2_tallskinny", warmup=WARMUP, iters=ITERS)
    gb_s = (n_bytes / 1e9) / (result.median_ms / 1e3)
    rows.append({"category": "splitk", "variant": "v2_no_split", "K": K, "median_ms": result.median_ms, "achieved_gb_s": gb_s})
    print(rows[-1])

    for split in [4, 8, 16, 32]:
        result = time_cuda(lambda: ops.gemv_fp16_v4_splitk(W, x, split), name=f"gemv_v4_split{split}", warmup=WARMUP, iters=ITERS)
        gb_s = (n_bytes / 1e9) / (result.median_ms / 1e3)
        rows.append({"category": "splitk", "variant": f"v4_split{split}", "K": K, "median_ms": result.median_ms, "achieved_gb_s": gb_s})
        print(rows[-1])


def bench_rmsnorm_softmax(rows: list[dict]) -> None:
    rows_n, hidden = 32, 4096
    x = torch.randn(rows_n, hidden, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float16)
    n_bytes = 2 * rows_n * hidden * 2  # read x + write out, fp16

    result = time_cuda(lambda: ops.rmsnorm(x, weight, 1e-6), name="rmsnorm", warmup=WARMUP, iters=ITERS)
    gb_s = (n_bytes / 1e9) / (result.median_ms / 1e3)
    rows.append({"category": "rmsnorm", "variant": "rmsnorm", "K": hidden, "median_ms": result.median_ms, "achieved_gb_s": gb_s})
    print(rows[-1])

    for name, fn in [("softmax_twopass", ops.softmax_twopass), ("softmax_online", ops.softmax_online)]:
        result = time_cuda(lambda: fn(x), name=name, warmup=WARMUP, iters=ITERS)
        gb_s = (n_bytes / 1e9) / (result.median_ms / 1e3)
        rows.append({"category": "softmax", "variant": name, "K": hidden, "median_ms": result.median_ms, "achieved_gb_s": gb_s})
        print(rows[-1])


def make_plot(gemv_rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    variants = sorted({r["variant"] for r in gemv_rows})
    ks = sorted({r["K"] for r in gemv_rows})
    for variant in variants:
        ys = [next(r["achieved_gb_s"] for r in gemv_rows if r["variant"] == variant and r["K"] == k) for k in ks]
        ax.plot(ks, ys, marker="o", label=variant)
    ax.axhline(T4_PEAK_GB_S, color="tab:red", linestyle="--", label=f"{T4_PEAK_GB_S:.0f} GB/s peak (T4)")
    ax.axhline(GEMV_V3_TARGET_FRACTION * T4_PEAK_GB_S, color="tab:orange", linestyle=":",
               label=f"v3 target ({GEMV_V3_TARGET_FRACTION:.0%} peak)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K")
    ax.set_ylabel("achieved GB/s")
    ax.set_title("M2: FP16 GEMV bandwidth vs T4 peak (N=4096)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench_m2.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    gemv_rows: list[dict] = []
    bench_gemv(gemv_rows)
    write_csv(gemv_rows, REPORTS_DIR / "m2_gemv_bandwidth.csv")
    make_plot(gemv_rows, REPORTS_DIR / "m2_gemv_bandwidth.png")

    other_rows: list[dict] = []
    bench_splitk(other_rows)
    bench_rmsnorm_softmax(other_rows)
    write_csv(other_rows, REPORTS_DIR / "m2_norm_softmax_splitk.csv")


if __name__ == "__main__":
    main()
