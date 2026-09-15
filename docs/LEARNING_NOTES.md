# Learning notes

Running log for this project -- for me, not recruiters. Check off topics as
they're internalized well enough to explain at a whiteboard (PROJECT_SPEC.md
sec 9). Add a dated paragraph per milestone under "Milestone log" explaining
what clicked and what didn't.

## Topic checklist

### CUDA core
- [x] thread/block/warp/grid model
- [x] memory hierarchy
- [x] coalescing
- [x] shared memory and bank conflicts
- [ ] occupancy
- [ ] warp divergence
- [x] warp shuffles
- [x] atomics
- [ ] streams and events
- [x] pinned memory
- [ ] async copy
- [x] `__restrict__` and pointer aliasing
- [x] vectorized access
- [ ] register pressure and spilling
- [ ] launch overhead and CUDA graphs

### Numerics
- [ ] FP16/BF16/FP32/TF32
- [ ] accumulation order and error
- [ ] symmetric vs asymmetric quantization
- [ ] granularity (tensor/channel/group/block)
- [ ] zero-point
- [ ] outlier channels
- [ ] E8M0 and microscaling
- [ ] fake quant vs real quant
- [ ] calibration (min-max, percentile, MSE, AWQ)
- [ ] GPTQ

### Inference systems
- [ ] prefill vs decode
- [ ] why decode is memory-bound
- [ ] arithmetic intensity and roofline
- [ ] KV cache sizing
- [ ] paged attention
- [ ] GQA/MQA
- [ ] continuous batching (conceptually)
- [ ] speculative decoding (conceptually)
- [ ] offloading and PCIe limits
- [ ] contextual/dynamic sparsity
- [ ] weight streaming and prefetch

### Kernels written
- [x] reductions
- [ ] RMSNorm
- [ ] online softmax
- [ ] GEMV (FP16, W8A16, W4A16-group)
- [ ] INT8 tensor-core GEMM
- [ ] fused SwiGLU
- [ ] fused QKV
- [ ] KV-cache append
- [ ] decode attention
- [ ] top-k / radix select
- [ ] stream compaction
- [ ] row gather
- [ ] cache-or-stream fused GEMV

### Engineering
- [ ] CUDA extension builds and arch flags
- [ ] pybind11
- [ ] pytest for numerics
- [ ] benchmark methodology
- [ ] Nsight Systems and Compute
- [ ] reproducibility and seeding
- [ ] ablation design
- [ ] technical writing

## Milestone log

### M0 -- Harness and baseline (done)

2026-09-14 -- 2026-09-15. Repo scaffolded locally (no NVIDIA GPU on the dev
machine -- see docs/DESIGN.md), pushed to GitHub, driven from a Colab T4
session via the `colab-mcp` MCP bridge (Claude Code editing/running notebook
cells directly rather than a human clicking through them).

What actually happened, worth remembering:
- `uvx git+https://...` for an MCP server is slow on its very first
  invocation (cloning + resolving ~100 deps) and blew through Claude Code's
  30s MCP connection timeout on the first try. Once cached locally, it
  connects in ~8s. If a similar MCP server ever times out on first connect,
  retry rather than assume it's broken.
- The Colab browser bridge attaches to a specific tab at connect time, not
  dynamically to "whatever tab is focused now" -- if cell reads look wrong
  (e.g. an unexpectedly blank notebook), close other Colab tabs and
  re-open the connection.
- transformers 5.x removed the `load_in_8bit=True` shorthand kwarg;
  needs `quantization_config=BitsAndBytesConfig(load_in_8bit=True)` now.
  Also: don't let one baseline variant's failure (missing bitsandbytes)
  crash the script before `write_csv` runs -- wrap each variant so partial
  results survive.
- `add_one` built and passed on the first real compile against `sm_75` on
  an actual T4 -- the `setup.py`/`common.cuh` arch-gating scaffolding from
  the GPU-less local session translated correctly to real hardware.
- Colab's T4 instance allowed `nvidia-smi -lgc` clock locking this session
  (not guaranteed -- the notebook falls back gracefully if it's ever
  refused).

Results: see docs/RESULTS.md and `reports/m0_pcie_bandwidth.csv` /
`reports/m0_baseline.csv`. Next: M1 (CUDA fundamentals -- vector add,
coalescing sweep, reduction variants, transpose with/without padding).

### M1 -- CUDA fundamentals (done)

2026-09-15. Four kernel families, each isolating one memory-hierarchy
concept. Full numbers in docs/RESULTS.md and `reports/m1_bandwidth.{csv,png}`;
here's *why* each jump happens.

**Vector add** climbs from 167 -> 255 GB/s as n grows from 256K to 16M
elements, never quite reaching the 320 GB/s peak. This is the launch-overhead
/ occupancy-ramp story: a kernel this trivial is memory-bound from the first
instruction, but at small n there simply aren't enough in-flight warps to
hide DRAM latency, and the fixed cost of a kernel launch (a few microseconds)
is a larger fraction of a shorter-running kernel. Bigger n amortizes both.

**Strided copy** is the coalescing demo made visible: at stride 1, a warp's
32 threads request 32 contiguous floats -- one 128-byte transaction serves
the whole warp. At stride 8, those same 32 threads scatter across 8x the
address range, and the hardware issues far more transactions to serve the
same 32 loads (worst case, one full cache-line fetch per thread instead of
per warp). Same nominal bytes "delivered," 6x less effective bandwidth by
stride 128. This is exactly why layout.h (row-major, row-addressable weight
rows) matters so much later in M6+ -- indirection through a gather index
must still land on contiguous rows, or offload streaming pays this same tax.

**Sum reduction**, the biggest set of jumps:
- v1 (naive atomic) -> v2 (shared-memory tree): **46x**. v1 has every one of
  16M threads issue a global atomicAdd to the *same* address -- those
  serialize almost completely, so this kernel isn't memory-bound at all, it's
  atomic-contention-bound. v2 reduces within a block first (tree over shared
  memory, hence only one atomicAdd per block, and 2^24/256 = 65536x fewer
  global atomics.
- v2 -> v3 (warp-shuffle): 1.6x. The shared-memory tree still costs a
  `__syncthreads()` and a shared-memory read/write per level (8 levels for
  256 threads). Warp-shuffle reduces the first 5 of those levels (32 threads)
  via register-to-register `__shfl_down_sync` with no shared memory and no
  barrier, only falling back to a tiny shared-memory step to combine the 8
  warp-partials.
- v3 -> v4 (vectorized float4): 1.9x. Same reduction structure, but each
  thread now issues one 128-bit load carrying 4 floats instead of four
  32-bit loads -- 4x fewer load instructions per byte moved, which matters
  because at this point the kernel is genuinely bandwidth-bound and issue
  rate / instruction overhead is the remaining bottleneck. v4 lands within
  15% of the 320 GB/s peak -- about as close as a plain reduction gets.

**Transpose**: naive writes are the failure mode -- `out[col*n+row]` means
consecutive threads (consecutive `col`) write to addresses `n` floats apart,
so writes are fully uncoalesced even though reads are fine. Tiling through
shared memory fixes this by buffering a tile with coalesced reads *and*
coalesced writes, transposing inside shared memory instead of in global
memory -- roughly a 2x win from n=1024 up. Padding the tile
(`[32][33]` instead of `[32][32]`) fixes a subtler problem: reading the tile
column-wise during the write-out step means all 32 threads in a warp hit
the same shared-memory bank (stride-32 access into a 32-bank memory) --
a 32-way conflict serialized into 32 separate transactions. The extra
padding column shifts each row's start by one bank, so the same column-wise
access pattern now lands on 32 distinct banks. At n=512 all three are within
noise (too few tiles per launch to amortize overhead); the effect only shows
up once the kernel runs long enough for bank conflicts to actually dominate.
