"""Benchmark harness: honest CUDA-event timing with warmup, percentiles, and CSV output.

Rule (see PROJECT_SPEC.md sec 6, M0): >=20 warmup iters, >=100 measured iters,
report median / p90 / stddev, torch.cuda.synchronize() around every sample so
queued work never leaks across measurements. Never put a number in prose that
isn't backed by a CSV in reports/.
"""
from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass
class TimingResult:
    name: str
    median_ms: float
    p90_ms: float
    stddev_ms: float
    min_ms: float
    max_ms: float
    n_iters: int

    def __str__(self) -> str:
        return (
            f"{self.name}: median={self.median_ms:.4f}ms p90={self.p90_ms:.4f}ms "
            f"stddev={self.stddev_ms:.4f}ms (n={self.n_iters})"
        )


def time_cuda(fn: Callable[[], None], *, name: str = "unnamed", warmup: int = 20, iters: int = 100) -> TimingResult:
    """Time `fn` on the current CUDA device using CUDA events.

    `fn` takes no arguments and launches exactly the work to be measured (wrap
    args with a closure/lambda). Synchronizes before and after every sample so
    one iteration's queued work can't bleed into the next iteration's timer.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("time_cuda requires a CUDA device")
    if warmup < 20 or iters < 100:
        raise ValueError("PROJECT_SPEC.md M0 requires warmup>=20 and iters>=100 for a result to count")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_ms: list[float] = []

    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end))

    samples_ms.sort()
    p90_idx = min(len(samples_ms) - 1, int(0.9 * len(samples_ms)))
    return TimingResult(
        name=name,
        median_ms=statistics.median(samples_ms),
        p90_ms=samples_ms[p90_idx],
        stddev_ms=statistics.pstdev(samples_ms),
        min_ms=samples_ms[0],
        max_ms=samples_ms[-1],
        n_iters=iters,
    )


def write_csv(rows: list[dict], path: str | Path) -> None:
    """Write `rows` (checked into reports/, per PROJECT_SPEC.md sec 5) as a CSV report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
