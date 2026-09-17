"""M7 acceptance (task 2): row gather from a pinned host arena into device
memory. Compares gather-into-staging-then-one-H2D-copy against one
cudaMemcpyAsync per selected row -- "compare against per-row
cudaMemcpyAsync (it will be far worse -- show the data)"
(PROJECT_SPEC.md M7 task 2). Run on the Colab/Kaggle T4 session, not
locally (no CUDA here).

Also records bytes transferred directly (not inferred from timing) per
M7's acceptance wording ("measurable reduction in bytes transferred per
token -- instrument it directly, count bytes, don't infer from timing").
For a fixed k, both variants move the exact same k*row_nbytes payload --
the comparison here is about transfer PATTERN (call count), which is why
task 3 (sparse fused GEMV, restricting compute to the selected set) is
where the actual bytes-per-token reduction comes from.

Writes reports/m7_gather_timing.csv.
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

# Qwen3-14B's shape (see docs/LEARNING_NOTES.md's M6/M7 entries): row count
# sweeps the same intermediate_size candidates as task 1's topk bench;
# row_nbytes is up_proj/down_proj's actual INT4-packed row width at
# hidden_size=5120 (ceil(5120/8)*4 bytes -- see soinfer.quant.pack.pack_int4).
NUM_ROWS_CANDIDATES = [11008, 17408, 27648]
HIDDEN = 5120
ROW_NBYTES = ((HIDDEN + 7) // 8) * 4  # 2560
K_FRACTIONS = [1.0, 0.75, 0.5, 0.375, 0.25, 0.125]  # matches task 1 and task 4's sweep


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench_m7_gather.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    torch.manual_seed(0)
    rows = []
    for num_rows in NUM_ROWS_CANDIDATES:
        matrix = torch.randint(0, 256, (num_rows, ROW_NBYTES), dtype=torch.uint8).pin_memory()
        for frac in K_FRACTIONS:
            k = max(1, int(round(num_rows * frac)))
            indices = torch.randperm(num_rows)[:k].contiguous()
            nbytes = k * ROW_NBYTES

            staging = torch.empty(k, ROW_NBYTES, dtype=torch.uint8).pin_memory()
            gpu_dst = torch.empty(k, ROW_NBYTES, dtype=torch.uint8, device="cuda")

            def _staged(matrix=matrix, indices=indices, staging=staging, gpu_dst=gpu_dst):
                ops.gather_rows_staged(matrix, indices, staging, gpu_dst)

            def _naive(matrix=matrix, indices=indices, gpu_dst=gpu_dst):
                ops.gather_rows_naive(matrix, indices, gpu_dst)

            staged_result = time_cuda(_staged, name=f"gather_staged_R{num_rows}_k{k}", warmup=WARMUP, iters=ITERS)
            naive_result = time_cuda(_naive, name=f"gather_naive_R{num_rows}_k{k}", warmup=WARMUP, iters=ITERS)

            row = {
                "num_rows": num_rows, "row_nbytes": ROW_NBYTES, "k": k, "k_over_num_rows": frac,
                "bytes_transferred": nbytes,
                "staged_median_us": staged_result.median_ms * 1000,
                "staged_p90_us": staged_result.p90_ms * 1000,
                "naive_median_us": naive_result.median_ms * 1000,
                "naive_p90_us": naive_result.p90_ms * 1000,
                "naive_over_staged_ratio": naive_result.median_ms / staged_result.median_ms,
            }
            print(row)
            rows.append(row)

    write_csv(rows, REPORTS_DIR / "m7_gather_timing.csv")

    half_row = next(r for r in rows if r["num_rows"] == 17408 and abs(r["k_over_num_rows"] - 0.5) < 1e-6)
    status = "PASS" if half_row["naive_over_staged_ratio"] > 1.0 else "FAIL"
    print(
        f"[{status}] at num_rows=17408, k=0.5*num_rows: staged={half_row['staged_median_us']:.2f}us, "
        f"naive={half_row['naive_median_us']:.2f}us ({half_row['naive_over_staged_ratio']:.2f}x slower)"
    )


if __name__ == "__main__":
    main()
