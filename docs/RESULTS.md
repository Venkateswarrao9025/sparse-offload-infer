# Results

Every number below is backed by a checked-in CSV in `reports/`. No number goes
in here otherwise (PROJECT_SPEC.md sec 6/9).

## M0 -- Harness and baseline

Hardware: Google Colab, NVIDIA Tesla T4 (15360 MiB), driver 580.82.07, CUDA
13.0 (torch built against CUDA 12.8), SM clock locked to 1590 MHz via
`nvidia-smi -lgc`. Source: `reports/m0_pcie_bandwidth.csv`, `reports/m0_baseline.csv`.

**PCIe H2D bandwidth** (pinned vs pageable host memory, median of 100 runs
after 20 warmup, `bench/bench_pcie.py`):

| size | pageable | pinned |
|---|---|---|
| 1 MB | 3.2 GB/s | 11.2 GB/s |
| 16 MB | 6.5 GB/s | 12.2 GB/s |
| 256 MB | 6.7 GB/s | 12.3 GB/s |
| 1024 MB | 6.8 GB/s | **12.3 GB/s** |

Pinned memory plateaus around **12.3 GB/s** at large transfer sizes -- this is
the number M6+ offload streaming is judged against, not the PCIe Gen3 x16
spec-sheet figure (~15.75 GB/s theoretical). Pageable memory tops out at
~6.9 GB/s, well under half of pinned -- confirms pinning host buffers is not
optional for the offload path.

**Dev-model decode baseline** (Qwen3-1.7B, FP16, greedy, 128 new tokens,
`bench/bench_baseline.py`):

| variant | tokens/sec | peak VRAM |
|---|---|---|
| HF FP16 | 23.2 | 3.47 GB |
| HF FP16 + `torch.compile` | 22.6 | 3.47 GB |
| bitsandbytes INT8 | not run -- `bitsandbytes` not installed on this Colab image |

`torch.compile` did not help here (single-sample greedy decode, short
generation -- compilation overhead likely isn't amortized; not investigated
further, out of scope for M0). This FP16 number (~23 tok/s) is the baseline
soinfer's own kernels are compared against from M4 onward.

Pending: M1 onward.
