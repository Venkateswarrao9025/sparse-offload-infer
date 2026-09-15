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

## M1 -- CUDA fundamentals

Same T4 session as M0 (clock locked to 1590 MHz). Source: `reports/m1_bandwidth.csv`,
`reports/m1_bandwidth.png`.

**Vector add** (`out = a + b`, achieved GB/s = 3*n*4 bytes / time):

| n (elements) | achieved GB/s |
|---|---|
| 262,144 | 167.0 |
| 1,048,576 | 213.1 |
| 4,194,304 | 244.1 |
| 16,777,216 | **254.9** |

Approaches but doesn't reach the 320 GB/s peak even at 16M elements -- launch
overhead and imperfect occupancy still cost a few percent at this size; the
larger-still sizes needed to fully amortize that were out of scope for a M1
warm-up kernel.

**Strided copy** (coalescing collapse; nominal GB/s = 2*n*4 bytes / time, n=2^18 fixed):

| stride | achieved GB/s |
|---|---|
| 1 | 114.5 |
| 2 | 113.8 |
| 4 | 71.8 |
| 8 | 46.5 |
| 16 | 26.3 |
| 32 | 24.4 |
| 64 | 22.6 |
| 128 | **19.4** |

Monotonic collapse as stride grows -- by stride 128 achieved bandwidth is
~6x lower than stride 1, for the exact same number of "useful" bytes moved.
(Absolute numbers here are lower than vector_add's because this sweep uses a
much smaller working set, 1-128 MB vs up to 64 MB x2; the collapse *shape*,
not the absolute GB/s, is the point of this kernel.)

**Sum reduction** (n=2^24, achieved GB/s = n*4 bytes read / time):

| variant | achieved GB/s |
|---|---|
| v1 naive atomic | 2.0 |
| v2 shared-memory tree | 92.0 |
| v3 warp-shuffle | 147.4 |
| v4 vectorized float4 + warp-shuffle | **273.1** |

v1-to-v2 is a **46x** jump (eliminating global-atomic contention by reducing
within a block first); v2-to-v3 is another 1.6x (avoiding shared-memory
traffic and `__syncthreads()` entirely inside a warp); v3-to-v4 is another
1.9x (4x fewer thread-instructions issued per byte, via `float4` loads).
v4 reaches within 15% of the 320 GB/s peak.

**Transpose** (n x n, achieved GB/s = 2*n^2*4 bytes / time):

| n | naive | tiled, unpadded | tiled, padded |
|---|---|---|---|
| 512 | 53.9 | 57.1 | 52.6 |
| 1024 | 91.0 | 154.6 | 177.0 |
| 2048 | 96.8 | 143.9 | **200.0** |
| 4096 | 79.1 | 177.0 | **201.8** |

At n=512 the three are within noise of each other (too few tiles to amortize
launch overhead). From n=1024 up, the pattern is consistent: naive (fully
uncoalesced writes) is slowest, tiling through shared memory roughly doubles
throughput by making both global reads and writes coalesced, and padding the
shared-memory tile by one column adds another consistent ~15-25% by removing
the 32-way bank conflict on the transposed read out of shared memory.

See `docs/LEARNING_NOTES.md` for the fuller explanation of each jump.

Pending: M2 onward.
