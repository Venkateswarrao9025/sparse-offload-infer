"""M7 acceptance (task 1): top-k selection kernel timing. "At k = 0.5*I you
want this in single-digit microseconds -- if selection costs more than the
transfer it saves, you've lost." Run on the Colab/Kaggle T4 session, not
locally (no CUDA here).

Writes reports/m7_topk_timing.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import time_cuda, write_csv

import soinfer.ops as ops

WARMUP = 20
ITERS = 200
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Qwen3-14B's actual intermediate_size (see docs/LEARNING_NOTES.md's M6
# entry for where this came from), plus the spec's own reference sizes.
INTERMEDIATE_SIZES = [11008, 17408, 27648]
K_FRACTIONS = [1.0, 0.75, 0.5, 0.375, 0.25, 0.125]  # matches M7 task 4's sweep


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench_m7_topk.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    torch.manual_seed(0)
    rows = []
    for I in INTERMEDIATE_SIZES:
        abs_g = torch.rand(I, device="cuda", dtype=torch.float32) * 10.0
        for frac in K_FRACTIONS:
            k = max(1, int(round(I * frac)))

            def _select(abs_g=abs_g, k=k):
                ops.topk_threshold_select(abs_g, k)

            result = time_cuda(_select, name=f"topk_I{I}_k{k}", warmup=WARMUP, iters=ITERS)
            row = {"I": I, "k": k, "k_over_I": frac, "median_us": result.median_ms * 1000,
                   "p90_us": result.p90_ms * 1000}
            print(row)
            rows.append(row)

    write_csv(rows, REPORTS_DIR / "m7_topk_timing.csv")

    half_row = next(r for r in rows if r["I"] == 17408 and abs(r["k_over_I"] - 0.5) < 1e-6)
    status = "PASS" if half_row["median_us"] < 10.0 else "FAIL"
    print(f"[{status}] topk_threshold_select at I=17408, k=0.5*I: {half_row['median_us']:.2f}us "
          f"(target: single-digit microseconds)")


if __name__ == "__main__":
    main()
