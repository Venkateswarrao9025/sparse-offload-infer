# Learning notes

Running log for this project -- for me, not recruiters. Check off topics as
they're internalized well enough to explain at a whiteboard (PROJECT_SPEC.md
sec 9). Add a dated paragraph per milestone under "Milestone log" explaining
what clicked and what didn't.

## Topic checklist

### CUDA core
- [ ] thread/block/warp/grid model
- [ ] memory hierarchy
- [ ] coalescing
- [ ] shared memory and bank conflicts
- [ ] occupancy
- [ ] warp divergence
- [ ] warp shuffles
- [ ] atomics
- [ ] streams and events
- [ ] pinned memory
- [ ] async copy
- [ ] `__restrict__` and pointer aliasing
- [ ] vectorized access
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
- [ ] reductions
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

### M0 -- Harness and baseline (in progress)

Started 2026-09-14. Repo scaffolded locally (no NVIDIA GPU on the dev
machine -- see docs/DESIGN.md). Next: push to GitHub, run the Colab
bootstrap notebook, confirm `add_one` builds and passes, then run
`bench_pcie.py` and `bench_baseline.py` on a T4 session to produce
`reports/m0_pcie_bandwidth.csv` and `reports/m0_baseline.csv`.
