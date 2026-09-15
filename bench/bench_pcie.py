"""M0 task 4: measure real H2D PCIe bandwidth, pinned vs pageable, vs transfer size.

Write the number down -- every offload measurement from M6 onward is judged
against this, not the spec-sheet PCIe figure. Run on the actual target GPU
(Colab/Kaggle T4), not locally.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import time_cuda, write_csv

SIZES_MB = [1, 4, 16, 64, 256, 512, 1024]
WARMUP = 20
ITERS = 100


def bench_h2d(size_mb: int, pinned: bool) -> dict:
    n_bytes = size_mb * 1024 * 1024
    n_elems = n_bytes // 4  # float32
    host = torch.empty(n_elems, dtype=torch.float32, pin_memory=pinned)
    device = torch.empty(n_elems, dtype=torch.float32, device="cuda")

    def _copy():
        device.copy_(host, non_blocking=pinned)

    result = time_cuda(_copy, name=f"h2d_{'pinned' if pinned else 'pageable'}_{size_mb}MB", warmup=WARMUP, iters=ITERS)
    gb_s = (n_bytes / 1e9) / (result.median_ms / 1e3)
    return {
        "size_mb": size_mb,
        "pinned": pinned,
        "median_ms": result.median_ms,
        "p90_ms": result.p90_ms,
        "stddev_ms": result.stddev_ms,
        "achieved_gb_s": gb_s,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench_pcie.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    rows = []
    for size_mb in SIZES_MB:
        for pinned in (False, True):
            row = bench_h2d(size_mb, pinned)
            print(row)
            rows.append(row)

    out_path = Path(__file__).resolve().parent.parent / "reports" / "m0_pcie_bandwidth.csv"
    write_csv(rows, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
