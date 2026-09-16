"""M4 acceptance: quantized GEMV throughput vs FP16 GEMV (PROJECT_SPEC.md sec
6, M4). Run on the Colab/Kaggle T4 session, not locally (no CUDA here).

Acceptance check: gemv_w4a16_group_lop3 should reach >=3x the throughput of
gemv_fp16_v3 at K=N=4096 (1/4 the bytes moved; 3x is the realistic yield
after overheads the spec asks for). Also reports the naive-vs-LOP3-dequant
delta for W4A16 (M4 task 3).

Writes reports/m4_gemv_throughput.csv and reports/m4_gemv_throughput.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import time_cuda, write_csv

import soinfer.ops as ops
from soinfer.quant import formats, pack

WARMUP = 20
ITERS = 100
GROUP_SIZE = 128
W4A16_SPEEDUP_TARGET = 3.0
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _quantized_int8(N: int, K: int):
    W = torch.randn(N, K)
    config = formats.QuantConfig(bits=8, granularity="per_channel")
    qt = formats.quantize(W, config)
    Wq, _ = pack.pack_int8(qt.qweight)
    return Wq.cuda(), qt.scale.cuda()


def _quantized_int4_group(N: int, K: int, group_size: int):
    W = torch.randn(N, K)
    config = formats.QuantConfig(bits=4, granularity="group", group_size=group_size)
    qt = formats.quantize(W, config)
    Wq_packed, _ = pack.pack_int4(qt.qweight)
    return Wq_packed.cuda(), qt.scale.cuda()


def bench(rows: list[dict]) -> tuple[float, float]:
    N = 4096
    fp16_gb_s_at_4096 = None
    w4a16_lop3_gb_s_at_4096 = None

    for K in [1024, 2048, 4096, 8192]:
        x = torch.randn(K, device="cuda", dtype=torch.float16)
        fp16_bytes = N * K * 2  # half weights

        W = torch.randn(N, K, device="cuda", dtype=torch.float16)
        result = time_cuda(lambda: ops.gemv_fp16_v3(W, x), name=f"fp16_v3_K{K}", warmup=WARMUP, iters=ITERS)
        fp16_gb_s = (fp16_bytes / 1e9) / (result.median_ms / 1e3)
        rows.append({"category": "gemv", "variant": "fp16_v3", "K": K, "median_ms": result.median_ms,
                     "achieved_gb_s": fp16_gb_s})
        print(rows[-1])
        if K == N:
            fp16_gb_s_at_4096 = fp16_gb_s

        Wq8, scale8 = _quantized_int8(N, K)
        int8_bytes = N * K * 1
        result = time_cuda(lambda: ops.gemv_w8a16(Wq8, scale8, x, K), name=f"w8a16_K{K}", warmup=WARMUP, iters=ITERS)
        gb_s = (int8_bytes / 1e9) / (result.median_ms / 1e3)
        rows.append({"category": "gemv", "variant": "w8a16", "K": K, "median_ms": result.median_ms,
                     "achieved_gb_s": gb_s})
        print(rows[-1])

        Wq4, scale4 = _quantized_int4_group(N, K, GROUP_SIZE)
        int4_bytes = N * ((K + 7) // 8) * 4  # packed nibble bytes actually transferred

        result = time_cuda(lambda: ops.gemv_w4a16_group(Wq4, scale4, x, K, GROUP_SIZE), name=f"w4a16_naive_K{K}",
                            warmup=WARMUP, iters=ITERS)
        gb_s = (int4_bytes / 1e9) / (result.median_ms / 1e3)
        rows.append({"category": "gemv", "variant": "w4a16_naive", "K": K, "median_ms": result.median_ms,
                     "achieved_gb_s": gb_s})
        print(rows[-1])

        result = time_cuda(lambda: ops.gemv_w4a16_group_lop3(Wq4, scale4, x, K, GROUP_SIZE),
                            name=f"w4a16_lop3_K{K}", warmup=WARMUP, iters=ITERS)
        gb_s = (int4_bytes / 1e9) / (result.median_ms / 1e3)
        rows.append({"category": "gemv", "variant": "w4a16_lop3", "K": K, "median_ms": result.median_ms,
                     "achieved_gb_s": gb_s})
        print(rows[-1])
        if K == N:
            w4a16_lop3_gb_s_at_4096 = gb_s

    return fp16_gb_s_at_4096, w4a16_lop3_gb_s_at_4096


def make_plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    variants = sorted({r["variant"] for r in rows})
    ks = sorted({r["K"] for r in rows})
    for variant in variants:
        ys = [next(r["achieved_gb_s"] for r in rows if r["variant"] == variant and r["K"] == k) for k in ks]
        ax.plot(ks, ys, marker="o", label=variant)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K")
    ax.set_ylabel("achieved GB/s (bytes of quantized weight actually moved)")
    ax.set_title("M4: quantized GEMV bandwidth vs FP16 GEMV (N=4096)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench_m4.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    rows: list[dict] = []
    fp16_gb_s, w4a16_lop3_gb_s = bench(rows)
    write_csv(rows, REPORTS_DIR / "m4_gemv_throughput.csv")
    make_plot(rows, REPORTS_DIR / "m4_gemv_throughput.png")

    assert fp16_gb_s is not None and w4a16_lop3_gb_s is not None
    # Throughput here is GB/s of each kernel's own (differently-sized) input,
    # so "3x throughput" per PROJECT_SPEC.md M4 means 3x the row-time speedup,
    # i.e. compare wall-clock time, not GB/s directly (GB/s is bytes/time and
    # w4a16 moves 1/4 the bytes, so equal-GB/s would already mean 4x-faster).
    fp16_row = next(r for r in rows if r["variant"] == "fp16_v3" and r["K"] == 4096)
    w4a16_row = next(r for r in rows if r["variant"] == "w4a16_lop3" and r["K"] == 4096)
    speedup = fp16_row["median_ms"] / w4a16_row["median_ms"]
    status = "PASS" if speedup >= W4A16_SPEEDUP_TARGET else "FAIL"
    print(f"[{status}] gemv_w4a16_group_lop3 vs gemv_fp16_v3 at K=N=4096: {speedup:.2f}x "
          f"(target >= {W4A16_SPEEDUP_TARGET:.1f}x)")


if __name__ == "__main__":
    main()
