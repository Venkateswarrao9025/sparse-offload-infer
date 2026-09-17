"""M6 acceptance: computes overlap efficiency (achieved vs ideal) from an
`nsys stats --report cuda_gpu_trace --format csv` export -- the second half
of the check profile_m6_overlap.py's module docstring documents.

Usage: python bench/analyze_nsys_overlap.py <cuda_gpu_trace.csv>
Writes reports/m6_overlap_efficiency.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _total_duration(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _intersect_duration(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    total = 0
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if start < end:
            total += end - start
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(csv_path: Path) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    events = []
    for r in rows:
        try:
            start = int(r["Start (ns)"])
            dur = int(r["Duration (ns)"])
        except (KeyError, ValueError):
            continue
        events.append((start, start + dur, r.get("Name", "")))

    is_copy = lambda name: "memcpy" in name.lower()  # noqa: E731
    copy_events = [(s, e) for s, e, name in events if is_copy(name)]
    compute_events = [(s, e) for s, e, name in events if not is_copy(name)]

    copy_merged = _merge_intervals(copy_events)
    compute_merged = _merge_intervals(compute_events)
    copy_busy_ns = _total_duration(copy_merged)
    compute_busy_ns = _total_duration(compute_merged)
    overlap_ns = _intersect_duration(copy_merged, compute_merged)
    ideal_ns = min(copy_busy_ns, compute_busy_ns)

    wall_start = min(s for s, _, _ in events)
    wall_end = max(e for _, e, _ in events)
    wall_ns = wall_end - wall_start
    idle_ns = wall_ns - copy_busy_ns - compute_busy_ns + overlap_ns

    return {
        "copy_busy_ms": copy_busy_ns / 1e6,
        "compute_busy_ms": compute_busy_ns / 1e6,
        "overlap_achieved_ms": overlap_ns / 1e6,
        "overlap_ideal_ms": ideal_ns / 1e6,
        "overlap_efficiency_pct": (overlap_ns / ideal_ns * 100) if ideal_ns else 0.0,
        "wall_clock_ms": wall_ns / 1e6,
        "gpu_idle_ms": idle_ns / 1e6,
        "gpu_idle_pct": idle_ns / wall_ns * 100 if wall_ns else 0.0,
        "num_copy_events": len(copy_events),
        "num_compute_events": len(compute_events),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python bench/analyze_nsys_overlap.py <cuda_gpu_trace.csv>")
    csv_path = Path(sys.argv[1])
    result = analyze(csv_path)
    for k, v in result.items():
        print(f"{k}: {v}")

    out_path = REPORTS_DIR / "m6_overlap_efficiency.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
