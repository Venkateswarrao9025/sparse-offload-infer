"""M9 task 5: regenerates the three headline plots README.md/docs/RESULTS.md
reference, from CSVs already checked into reports/ -- no CUDA needed, pure
matplotlib/csv, runs anywhere (including this project's own no-GPU dev
machine). Run whenever an underlying CSV changes (e.g. a corrected Pareto
sweep) to keep the plots in sync.

Writes reports/m6_roofline.png, reports/m7_pareto.png, reports/m9_ablation_matrix.png.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _read_csv(name: str) -> list[dict]:
    with open(REPORTS_DIR / name, newline="") as f:
        return list(csv.DictReader(f))


def plot_roofline() -> None:
    rows = _read_csv("m6_roofline.csv")
    matrices = [r["matrix"] for r in rows]
    transfer = [float(r["transfer_gb_s"]) for r in rows]
    kernel = [float(r["kernel_gb_s"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(matrices))
    width = 0.35
    ax.bar([i - width / 2 for i in x], transfer, width, label="PCIe transfer (GB/s)", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], kernel, width, label="GEMV kernel (GB/s)", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(matrices, rotation=20, ha="right")
    ax.set_ylabel("GB/s")
    ax.set_title("M6: every weight matrix is transfer-bound, not compute-bound\n"
                  "(NVIDIA T4, pinned host memory, streaming INT4)")
    ax.legend()
    for i, r in enumerate(rows):
        ratio = float(r["transfer_over_kernel_ratio"])
        ax.annotate(f"{ratio:.1f}x", (i, max(transfer[i], kernel[i]) + 3), ha="center", fontsize=8)
    fig.tight_layout()
    out = REPORTS_DIR / "m6_roofline.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_pareto() -> None:
    rows = _read_csv("m7_pareto.csv")
    dense = next(r for r in rows if r["mode"] == "dense")
    dip_rows = [r for r in rows if r["mode"] == "dip" and float(r["k_over_I"]) < 1.0]
    dip_rows.sort(key=lambda r: float(r["k_over_I"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    k_over_I = [float(r["k_over_I"]) for r in dip_rows]
    ppl_ratio = [float(r["ppl_ratio_vs_dense"]) for r in dip_rows]
    ax1.plot(k_over_I, ppl_ratio, marker="o", color="#C44E52")
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="dense (no pruning)")
    ax1.set_yscale("log")
    ax1.set_xlabel("k / I (fraction of channels kept)")
    ax1.set_ylabel("perplexity ratio vs. dense (log scale)")
    ax1.set_title("M7: accuracy cliff -- corrected sweep\n(topk_select.cu's order-determinism fix, 2026-09-18)")
    ax1.invert_xaxis()
    ax1.legend()

    bytes_saved = [float(r["bytes_saved_vs_dense_pct"]) if "bytes_saved_vs_dense_pct" in r else
                   100.0 * (1.0 - float(r["bytes_per_token"]) / float(dense["bytes_per_token"])) for r in dip_rows]
    tok_s = [float(r["tokens_per_sec"]) for r in dip_rows]
    ax2b = ax2.twinx()
    ax2.plot(k_over_I, bytes_saved, marker="o", color="#4C72B0", label="bytes saved vs. dense")
    ax2b.plot(k_over_I, tok_s, marker="s", color="#55A868", label="tokens/sec")
    ax2b.axhline(float(dense["tokens_per_sec"]), color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("k / I (fraction of channels kept)")
    ax2.set_ylabel("bytes/token saved vs. dense (%)", color="#4C72B0")
    ax2b.set_ylabel("tokens/sec (dashed = dense)", color="#55A868")
    ax2.set_title("M7: bytes saved vs. throughput")
    ax2.invert_xaxis()
    fig.tight_layout()
    out = REPORTS_DIR / "m7_pareto.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_ablation_matrix() -> None:
    rows = _read_csv("m9_ablation_matrix.csv")
    cached = [r for r in rows if r["mode"] == "cache_aware_dip"]
    dense = next(r for r in rows if r["mode"] == "dense")
    k_values = sorted({float(r["k_over_I"]) for r in cached})

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in k_values:
        sub = sorted([r for r in cached if float(r["k_over_I"]) == k], key=lambda r: float(r["cache_frac"]))
        cache_frac = [float(r["cache_frac"]) for r in sub]
        tok_s = [float(r["tokens_per_sec"]) for r in sub]
        ax.plot(cache_frac, tok_s, marker="o", label=f"k/I={k:g}")
    ax.axhline(float(dense["tokens_per_sec"]), color="gray", linestyle="--", linewidth=1, label="dense")
    ax.set_xlabel("cache_frac (fraction of I resident in HotCache)")
    ax.set_ylabel("tokens/sec")
    ax.set_title("M9: cache-aware DIP throughput increases with cache size,\nat every k/I (Qwen3-1.7B)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = REPORTS_DIR / "m9_ablation_matrix.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_roofline()
    plot_pareto()
    plot_ablation_matrix()
