# Sparse-Offload Inference Engine
### A from-scratch INT4/INT8 CUDA inference runtime with Dynamic Input Pruning

**Status:** specification / build plan
**Intended executor:** Claude Cowork (agentic, multi-session)
**Owner:** you — you make the architecture calls, the agent does the typing and the grinding

---

## 0. What this document is

This is a complete build plan for a portfolio-grade CUDA + LLM-inference project. It is written to be handed to an agent session by session. Each milestone has:

- a **goal** (what exists at the end),
- a **task list** (what to actually write),
- **acceptance criteria** (how you know it's done, numerically),
- **topics you will have learned** (so you can talk about it in an interview).

Do not skip milestones. Each one depends on the numerics and the harness built in the previous one. The single most common way this project fails is writing a fast kernel that produces wrong numbers and not finding out for three weeks.

---

## 1. The project in one paragraph

Build an inference runtime for a decoder-only LLM that is **larger than the GPU's VRAM**. Weights live quantized to INT4 in pinned host memory; they are streamed over PCIe on demand. Because PCIe is ~25× slower than VRAM, the bottleneck is transfer, not compute — so the runtime exploits the fact that in a SwiGLU MLP, most intermediate channels contribute almost nothing for any given token. It computes the gate activation first, selects the top-k channels by magnitude, and streams **only the corresponding rows** of the up- and down-projection matrices. A resident on-GPU cache holds the channels that are selected most often, and a fused GEMV kernel reads each row from either cache or stream in a single pass. Everything in the hot path — dequantization, GEMV, SwiGLU, top-k selection, row gather, KV-cache append — is a hand-written CUDA kernel, validated against a PyTorch reference to a fixed tolerance.

That's the flagship. Two supporting studies (microscaling format benchmarking, and a dense small-model kernel suite) hang off the same codebase and reuse the same quantization library.

---

## 2. Scope: what's in and what's out

**In scope**
- Decode-time (autoregressive, batch size 1–8) inference. This is where memory-bound kernels and offload matter.
- Weight-only quantization: INT8 and INT4, symmetric, group-wise.
- Custom CUDA kernels compiled as a PyTorch C++ extension.
- Host↔device streaming, CUDA streams, overlap, pinned memory.
- Rigorous benchmarking with Nsight and accuracy evaluation.

**Out of scope (say so in the README — scoping is a signal of judgment, not a weakness)**
- Training or fine-tuning.
- Multi-GPU / tensor parallelism.
- Activation quantization below INT8 (W4A16 and W8A16 only; activations stay FP16).
- Beating vLLM/TensorRT-LLM on throughput in the resident (fits-in-VRAM) regime. You will not, and claiming it will get you caught. Your claim is specifically about the **offload-bound regime**.

---

## 3. Hardware and software targets

| Item | Primary target | Notes |
|---|---|---|
| GPU | NVIDIA T4 (Turing, `sm_75`, 16 GB, ~320 GB/s) | Cheap on Colab/Kaggle/Lambda. Matches the reference work. |
| Fallback | Any `sm_80`+ (A10G, L4, A100, RTX 30/40) | See arch notes below — do **not** silently assume features. |
| PCIe | Gen3 x16, ~12–13 GB/s effective | Measure yours; don't trust the spec sheet. |
| CUDA | 12.x toolkit | `nvcc`, Nsight Systems, Nsight Compute |
| Python | 3.10+, PyTorch 2.x, `transformers`, `accelerate` |

### Architecture gotchas the agent must respect

These are real and they will break builds if ignored:

- **Turing (`sm_75`) has no `bf16`.** Compute dtype is `fp16` (`__half`, `half2`). If you target Ampere+ you may use `bf16`, but then the code is no longer T4-compatible. Pick one and gate it with `#if __CUDA_ARCH__ >= 800`.
- **`cp.async` is `sm_80`+.** On Turing, shared-memory double buffering is done with ordinary loads into registers then `__syncthreads()`. Do not emit `cp.async` for `sm_75`.
- **Turing does have INT4 tensor cores** (`mma.sync.aligned.m8n8k32.s32.s4.s4.s32`) and INT8 (`m8n8k16`). These INT4 MMA instructions were removed in later architectures, so any code path using them must be arch-gated. Treat them as an optional stretch goal, not the main path.
- **Decode is GEMV, not GEMM.** At batch 1, tensor cores buy you almost nothing — you are bandwidth-bound. The tensor-core path matters for **prefill** only. Don't waste weeks optimizing MMA for decode.

---

## 4. Model choice

Use two models, for two different purposes:

1. **Development model — a 1–3B decoder** (SmolLM3-3B, Qwen3-1.7B, or Llama-3.2-1B). Fits in VRAM at FP16. Fast iteration, easy to validate layer-by-layer against HuggingFace. All kernel correctness work happens here.
2. **Headline model — a 14B–32B decoder** (Qwen3-14B or Qwen3-32B). Does **not** fit at INT4 on 15 GB, so it forces the offload regime that the whole DIP story depends on. Only bring this in at Milestone 6.

Both must use **SwiGLU MLPs** (gate/up/down structure) — DIP depends on it. Verify this before committing to a model.

---

## 5. Repository structure

```
sparse-offload-infer/
├── README.md                     # the artifact recruiters read — written LAST, from real numbers
├── pyproject.toml
├── setup.py                      # torch.utils.cpp_extension build, arch flags
├── Makefile                      # make build / test / bench / profile
│
├── csrc/
│   ├── include/
│   │   ├── common.cuh            # CUDA_CHECK, vectorized load/store, ceil_div, arch guards
│   │   ├── reduce.cuh            # warp shuffle reduce, block reduce
│   │   ├── dequant.cuh           # int4/int8 -> half fast paths (LOP3 trick)
│   │   ├── layout.h              # packing + group-scale memory layouts (SHARED SOURCE OF TRUTH)
│   │   └── dispatch.h            # dtype/arch dispatch macros
│   ├── kernels/
│   │   ├── elementwise.cu        # M1 warm-up
│   │   ├── reduce_demo.cu        # M1 warm-up
│   │   ├── rmsnorm.cu            # M2
│   │   ├── softmax_online.cu     # M2
│   │   ├── gemv_fp16.cu          # M2
│   │   ├── gemv_w8a16.cu         # M4
│   │   ├── gemv_w4a16_group.cu   # M4
│   │   ├── gemm_int8_tc.cu       # M4 (prefill, optional)
│   │   ├── swiglu_fused.cu       # M5
│   │   ├── qkv_fused.cu          # M5
│   │   ├── kv_cache.cu           # M5
│   │   ├── topk_select.cu        # M7
│   │   ├── gather_rows.cu        # M7
│   │   └── gemv_dip_fused.cu     # M8 — the centerpiece
│   └── bindings.cpp              # pybind11 / TORCH_LIBRARY registration
│
├── python/soinfer/
│   ├── ops.py                    # thin typed wrappers over the extension
│   ├── quant/
│   │   ├── formats.py            # per_tensor | per_channel | group_k | block32 | mx_e8m0
│   │   ├── pack.py               # int4 packing/unpacking, scale layout
│   │   ├── calibrate.py          # min-max, percentile, MSE-optimal search, AWQ-style scaling
│   │   └── quantize_model.py     # HF checkpoint -> quantized artifact on disk
│   ├── modules/
│   │   ├── linear_w4a16.py
│   │   ├── mlp_dip.py
│   │   └── attention.py
│   ├── offload/
│   │   ├── weight_store.py       # pinned host arena, row-addressable
│   │   ├── stream_manager.py     # CUDA streams, double buffering, events
│   │   └── hot_cache.py          # resident hot-channel cache + calibration
│   ├── runtime/
│   │   ├── loader.py
│   │   ├── kv_manager.py
│   │   └── generate.py           # the decode loop
│   └── bench/
│       ├── harness.py            # timing w/ warmup, CUDA events, percentiles
│       └── roofline.py
│
├── tests/
│   ├── test_reductions.py
│   ├── test_quant_roundtrip.py
│   ├── test_gemv_numerics.py
│   ├── test_layer_parity.py      # layer-by-layer vs HuggingFace
│   ├── test_kv_cache.py
│   ├── test_topk.py
│   └── test_end_to_end.py        # greedy decode matches reference token-for-token
│
├── bench/
│   ├── bench_gemv.py
│   ├── bench_pcie.py
│   ├── bench_decode.py
│   └── bench_ablation.py         # the table that goes in the README
│
├── studies/
│   ├── mx_formats/               # Study B (see §8)
│   └── dense_small_model/        # Study C (see §8)
│
├── configs/                      # yaml: model, quant scheme, k, cache size, seeds
├── reports/                      # generated plots + csv, checked in
└── docs/
    ├── DESIGN.md
    ├── RESULTS.md
    └── LEARNING_NOTES.md         # your running log — this is for you, not recruiters
```

---

## 6. Milestones

> **Rule for every milestone:** a kernel is not done until there is a test asserting it matches a PyTorch reference within tolerance, and a benchmark recording its achieved bandwidth or FLOPs against the hardware peak.

---

### M0 — Harness and baseline
**Goal:** you can build a CUDA extension, run it from Python, time it honestly, and you know exactly what you're trying to beat.

Tasks:
1. `setup.py` with `torch.utils.cpp_extension.CUDAExtension`, arch flags (`-gencode arch=compute_75,code=sm_75`), `-lineinfo` for profiling, `-O3`.
2. A trivial `add_one` kernel end-to-end to prove the toolchain works.
3. `bench/harness.py`: CUDA-event timing, ≥20 warmup iters, ≥100 measured iters, report median / p90 / stddev, `torch.cuda.synchronize()` in the right places, lock clocks if possible (`nvidia-smi -lgc`).
4. `bench_pcie.py`: measure real H2D bandwidth, pinned vs pageable, as a function of transfer size. **Write the number down.** Everything in M6–M9 is judged against it.
5. Baseline the dev model: HF FP16, HF + `torch.compile`, and (if it builds) `bitsandbytes` INT8 — record tokens/sec decode, prefill latency, peak VRAM.

Acceptance: `make bench` produces `reports/m0_baseline.csv` with error bars. Re-running gives medians within 3%.

Topics: nvcc toolchain, PyTorch extensions, CUDA events vs wall clock, pinned memory, benchmark hygiene, clock throttling.

---

### M1 — CUDA fundamentals
**Goal:** you actually understand the memory hierarchy rather than copying kernels.

Tasks: write and profile, in order — vector add (measure achieved bandwidth vs peak); a strided copy sweep to *see* coalescing collapse; sum reduction in four versions (naive atomic → shared memory tree → warp-shuffle → vectorized `float4`); a transpose with and without shared-memory padding to demonstrate bank conflicts.

Acceptance: a plot in `reports/` showing achieved bandwidth for each variant vs the 320 GB/s peak, plus a paragraph in `LEARNING_NOTES.md` explaining each jump.

Topics: grid/block/warp model, coalescing, occupancy, shared memory, bank conflicts, warp shuffle intrinsics (`__shfl_down_sync`), vectorized loads, `__restrict__`, memory-bound vs compute-bound.

---

### M2 — First real kernels: RMSNorm, softmax, FP16 GEMV
**Goal:** three kernels from the transformer hot path, correct and fast.

Tasks:
1. **RMSNorm** — one block per row, block reduce for sum of squares, `half2` vectorization. Watch FP16 accumulation error: accumulate in FP32.
2. **Online softmax** (the numerically stable single-pass formulation — same recurrence FlashAttention uses). Implement the two-pass version first, then the online one, and show they agree.
3. **FP16 GEMV** (`y = Wx`, W is `[N, K]`): v1 one thread per output row; v2 one warp per row with shuffle reduction; v3 vectorized `half2`/`float4` loads; v4 split-K with atomics for tall-skinny shapes.

Acceptance: max abs error vs `torch` reference < 1e-2 in FP16; GEMV v3 achieves ≥ 70% of peak memory bandwidth for K ≥ 4096.

Topics: block-level reduction patterns, numerical stability, FP32 accumulation, split-K, arithmetic intensity.

---

### M3 — Quantization library (pure Python/PyTorch, no CUDA yet)
**Goal:** a correct, well-tested quantization layer. Getting this wrong poisons every kernel downstream, and it's much easier to debug in Python.

Tasks:
1. Implement formats in `quant/formats.py`:
   - per-tensor symmetric INT8 / INT4
   - per-output-channel symmetric
   - **group-wise, group size 128** (the production default)
   - block-32
   - **microscaling (MX)**: shared **E8M0 power-of-two** scale per block of 32, OCP-style
2. Packing (`pack.py`): two INT4 values per byte, with an explicitly documented layout in `layout.h`. Decide interleaving now (the `[0,2,4,6,1,3,5,7]` nibble order that makes the LOP3 dequant trick work) and write it down — the CUDA kernels must agree byte-for-byte.
3. Calibration (`calibrate.py`): min-max, percentile clipping, MSE-optimal scale search, and an AWQ-style activation-aware channel scaling.
4. Fake-quant evaluation: perplexity on WikiText-2 for every (format, bit-width) pair on the dev model.

Acceptance:
- round-trip test: `unpack(pack(q)) == q` exactly, for random tensors of every shape including ragged tails.
- A table in `reports/m3_quant_accuracy.csv`. You should be able to reproduce the known result that **per-tensor INT4 collapses** — check what fraction of weights get forced to exact zero, and report it. That fraction *is* the explanation.

Topics: symmetric vs asymmetric quantization, scale/zero-point, granularity vs accuracy, outlier channels, E8M0 shared exponents, why power-of-two scales cost you accuracy, GPTQ/AWQ at a conceptual level.

---

### M4 — Quantized GEMV kernels
**Goal:** W8A16 and W4A16 GEMV, dequantizing in registers, faster than FP16 GEMV by roughly the compression ratio.

Tasks:
1. `gemv_w8a16`: load INT8 weights as `uint32` (4 at a time), dequant to `half2`, FMA into FP32 accumulator, warp-shuffle reduce.
2. `gemv_w4a16_group`: same but 8 weights per `uint32`, with a group scale reloaded every 128 elements along K. Keep the scale in a register across the group — reloading it per element is the classic performance bug here.
3. Fast dequant: implement the **LOP3 / bit-manipulation INT4→FP16 conversion** (construct FP16 bit patterns directly rather than going through int→float conversion instructions). Benchmark against the naive version and report the delta.
4. *(Optional, stretch)* INT8 tensor-core GEMM for prefill using `mma.m8n8k16` on `sm_75` or WMMA.

Acceptance:
- numerics match the M3 Python fake-quant reference exactly in the dequantized values, and within 1e-2 in the GEMV output.
- W4A16 GEMV reaches ≥ 3× the FP16 GEMV throughput at K=N=4096 (you're moving 1/4 the bytes; 3× is a realistic yield after overheads).

Topics: bit packing in registers, PTX inline asm, `LOP3.LUT`, instruction-level parallelism, register pressure, dequant-fused GEMV, why W4A16 beats FP16 at batch 1.

---

### M5 — Fused transformer kernels
**Goal:** the MLP and attention paths run on your kernels, and a full layer matches HuggingFace.

Tasks:
1. **Fused SwiGLU MLP**: `down(silu(gate(x)) * up(x))` — fuse the SiLU, the elementwise multiply, and ideally the gate/up GEMVs into one kernel so the intermediate never round-trips to global memory.
2. **Fused QKV projection**: one kernel, three outputs, one pass over `x`.
3. **KV cache append**: write new K/V into a preallocated cache with correct layout for the attention kernel. Support a contiguous layout first; add a **paged** layout (fixed-size blocks + block table) as a stretch goal.
4. **Decode attention**: single-query attention over the cache, one block per (head, sequence-chunk), online softmax from M2, FP32 accumulation.
5. **RMSNorm + quantize fusion**: emit quantized activations directly from the norm.

Acceptance: `test_layer_parity.py` — a full transformer block, your implementation vs HF, max relative error < 2e-2 on real activations. Then `test_end_to_end.py`: greedy decode of 128 tokens produces the **identical token sequence** as the HF reference. If tokens diverge, find out where before proceeding. This test is the backbone of the project.

Topics: kernel fusion, memory-traffic accounting, KV cache layout, paged attention, GQA/MQA head mapping, RoPE.

---

### M6 — Offload: streaming weights over PCIe
**Goal:** run a model whose weights exceed VRAM, with transfer overlapped against compute.

Tasks:
1. `weight_store.py`: allocate a **pinned** host arena, lay out each quantized weight matrix **row-major with rows as the addressable unit** (this is what makes M7's row gather possible — decide it now).
2. `stream_manager.py`: N CUDA streams, double (then triple) buffering, `cudaMemcpyAsync` + events, prefetch layer *i+1* while computing layer *i*.
3. Build the roofline: for each layer, bytes-to-transfer ÷ PCIe bandwidth vs kernel time. Show that you are transfer-bound and by how much. **This plot justifies the entire rest of the project** — put it in the README.
4. Now bring in the 14B/32B headline model.

Acceptance: the big model generates coherent text on a 15 GB GPU. Nsight Systems timeline shows copy and compute kernels genuinely overlapping, not serialized. Report the overlap efficiency (achieved vs ideal).

Topics: pinned memory, async copies, CUDA streams and events, producer–consumer pipelining, PCIe as the real bottleneck, roofline analysis, Nsight Systems.

---

### M7 — Dynamic Input Pruning (selective streaming)
**Goal:** cut per-token PCIe traffic by only streaming the MLP rows that matter for this token.

The insight: for token *t*, compute `g = gate_proj(x)` and `a = silu(g)`. Most entries of `a` are near zero, and they gate the corresponding intermediate channel. If channel *j* contributes ~nothing, you don't need row *j* of `up_proj` or column *j* of `down_proj` at all.

Tasks:
1. **Top-k selection kernel** (`topk_select.cu`): given `|g|` of size `I` (intermediate dim, ~11k–27k), select top-k indices. Implement three and compare: (a) full sort baseline, (b) threshold + count + compact, (c) radix select. At `k = 0.5·I` you want this in single-digit microseconds — if selection costs more than the transfer it saves, you've lost.
2. **Row gather**: gather selected rows from the pinned host arena into a contiguous staging buffer, then one coalesced H2D copy. Compare against per-row `cudaMemcpyAsync` (it will be far worse — show the data).
3. **Sparse fused GEMV**: `up` and `down` GEMVs restricted to the selected index set.
4. **Sweep k**: for `k/I ∈ {1.0, 0.75, 0.5, 0.375, 0.25, 0.125}`, measure both perplexity and tokens/sec. Produce the accuracy-vs-speedup **Pareto curve**. Pick the knee, and say why.

Acceptance: measurable reduction in bytes transferred per token (instrument it directly — count bytes, don't infer from timing), with perplexity degradation you have quantified rather than hand-waved.

Topics: dynamic/contextual sparsity, activation statistics, GPU top-k and radix select, stream compaction, gather/scatter, coalescing under indirection, accuracy–latency Pareto analysis.

---

### M8 — Cache-aware DIP
**Goal:** exploit the fact that channel selection is not uniformly random across tokens — some channels are chosen far more often than others. Keep those resident on the GPU.

Tasks:
1. **Calibration pass**: run a few hundred calibration tokens, record per-channel selection frequency per layer. Dump histograms to `reports/` — the skew is the story, so plot it.
2. **Hot cache** (`hot_cache.py`): pin the top-C channels of each layer in VRAM, sized to whatever VRAM is left over. Evaluate static-frequency vs LRU vs LFU-with-decay. Ties must break **deterministically** (by index) so runs are reproducible.
3. **Fused cache-or-stream GEMV** (`gemv_dip_fused.cu`): the centerpiece. For each selected channel, read its row from either the resident cache or the staging buffer **in a single pass**, with no host-side branching and no second kernel launch. Suggested design: a per-channel `int32` descriptor encoding `(source_flag, offset)` built by the selection kernel; the GEMV reads the descriptor and indexes the right pointer. Consider a predicated pointer select rather than a branch to avoid warp divergence.
4. Measure **hit rate** and the further reduction in transferred bytes. Sweep cache size.

Acceptance: an ablation table — dense offload → +DIP → +cache-aware DIP — with bytes/token, tokens/sec, and perplexity for each row. This table is the single most important artifact in the repo.

Topics: cache design and replacement policies, working-set analysis, warp divergence, predication, descriptor-driven kernels, determinism and reproducibility.

---

### M9 — Profiling, hardening, and the writeup
**Goal:** turn a working system into something that reads as engineering.

Tasks:
1. Nsight Compute on every hot kernel: achieved occupancy, memory throughput %, warp stall reasons, register count. For each, one sentence on what limits it. Where you optimized, show before/after counters.
2. Full ablation matrix across model size, quant format, k, and cache size. Three seeds, error bars.
3. Robustness: assert on unsupported shapes, handle ragged tails (K not divisible by group size), guard arch-specific paths, fail loudly rather than silently producing garbage.
4. CI: GitHub Actions running the CPU-side tests (quant round-trip, packing, layout) — the CUDA tests need a GPU runner, so mark them and document how to run locally.
5. `docs/RESULTS.md` and a README with the roofline plot, the Pareto curve, and the ablation table.

Acceptance: a stranger can clone, run `make build && make test && make bench`, and reproduce your headline number.

Topics: Nsight Compute, occupancy and stall analysis, experimental design, reproducibility.

---

## 7. How to state your results (this matters more than you think)

Write claims in this shape:

> On a single T4 (15 GB, PCIe 3.0 x16 at 11.8 GB/s measured), decoding Qwen3-14B quantized to group-128 symmetric INT4 with weights offloaded to host memory, cache-aware DIP at k/I = 0.5 reaches **X tok/s** vs **Y tok/s** for dense offload with the same kernels — a **Z× speedup** — at a WikiText-2 perplexity cost of **+P** (A.AA → B.BB). Median of 3 runs, 128-token generations, batch 1.

Always state: hardware, model, quant scheme, batch size, sequence length, what the baseline is, and the accuracy cost. A speedup number without a stated baseline and an accuracy cost is worthless, and a reviewer who knows the field will spot it immediately. Compare against **your own dense-offload implementation using the same kernels** — that's the honest comparison and it isolates the contribution of DIP.

---

## 8. Supporting studies

### Study B — Microscaling format benchmarking
Reuse `quant/formats.py`. Benchmark per-tensor, block-32, and OCP microscaling (E8M0) at 8/4/2 bits across several models. Two things to nail:
- **Quantify the per-tensor INT4 collapse**: report the percentage of weights forced to exactly zero, and connect it to the weight distribution. The mechanism is the finding, not the accuracy drop.
- **Derive the cost of E8M0 power-of-two scales analytically**, then check it empirically. Rounding a scale to the nearest power of two costs you a factor uniformly distributed over roughly [1, √2] in expectation and 2× worst case; the measured error ratio should land near that. A derivation that matches measurement is what makes this study interesting.

If you have vision-language models available, run it on VLMs for VQA-style tasks; otherwise run it on text LLMs with perplexity plus a couple of task benchmarks. Don't claim VLM results you didn't run.

### Study C — Dense small-model kernel suite
For the 3B dev model, ship a clean INT8 W8A16 path: GEMV with warp reduction, fused MLP, fused KV-cache update. Benchmark against `torch.compile` FP16. Report speedup **and** memory reduction. This is the "here's a self-contained thing that clearly works" project — some readers will only look at this one, so make it polished and easy to run.

---

## 9. Topic checklist

Tick these off in `LEARNING_NOTES.md` as you go. If you can't explain one at a whiteboard, you haven't finished that milestone.

**CUDA core** — thread/block/warp/grid · memory hierarchy · coalescing · shared memory and bank conflicts · occupancy · warp divergence · warp shuffles · atomics · streams and events · pinned memory · async copy · `__restrict__` and pointer aliasing · vectorized access · register pressure and spilling · launch overhead and CUDA graphs

**Numerics** — FP16/BF16/FP32/TF32 · accumulation order and error · symmetric vs asymmetric quantization · granularity (tensor/channel/group/block) · zero-point · outlier channels · E8M0 and microscaling · fake quant vs real quant · calibration (min-max, percentile, MSE, AWQ) · GPTQ

**Inference systems** — prefill vs decode · why decode is memory-bound · arithmetic intensity and roofline · KV cache sizing · paged attention · GQA/MQA · continuous batching (conceptually) · speculative decoding (conceptually) · offloading and PCIe limits · contextual/dynamic sparsity · weight streaming and prefetch

**Kernels you will have written** — reductions · RMSNorm · online softmax · GEMV (FP16, W8A16, W4A16-group) · INT8 tensor-core GEMM · fused SwiGLU · fused QKV · KV-cache append · decode attention · top-k / radix select · stream compaction · row gather · cache-or-stream fused GEMV

**Engineering** — CUDA extension builds and arch flags · pybind11 · pytest for numerics · benchmark methodology · Nsight Systems and Compute · reproducibility and seeding · ablation design · technical writing

---

## 10. Working protocol for Claude Cowork

Paste this section at the start of each agent session.

> You are implementing the project specified in `PROJECT_SPEC.md`. Work on **one milestone at a time**. Do not jump ahead.
>
> For each task:
> 1. Write the reference implementation in PyTorch first. Save it as the test oracle.
> 2. Write the CUDA kernel.
> 3. Write the test comparing them **before** benchmarking anything. State the tolerance and justify it.
> 4. Only once the test passes, benchmark and optimize. After each optimization, re-run the test.
> 5. Record every measurement in `reports/` as CSV with the config that produced it. Never put a number in prose that isn't in a CSV.
>
> Constraints:
> - Target `sm_75` by default. No `bf16`, no `cp.async` unless the code is arch-gated with `#if __CUDA_ARCH__ >= 800`.
> - `layout.h` is the single source of truth for weight packing. Python and CUDA must agree; if you change one, change both in the same commit.
> - Never report a speedup without also reporting the baseline configuration and the accuracy impact.
> - If a kernel is slower than expected, profile before rewriting. Report the limiter (bandwidth / occupancy / divergence / launch overhead) and only then change code.
> - If something in this spec turns out to be wrong or infeasible on the actual hardware, **stop and say so** rather than working around it silently. Note it in `docs/DESIGN.md` as a deviation with reasoning.
> - Commit at every green test with a message naming the milestone.
>
> At the end of each session, write a short status block: what's done, what the numbers are, what's next, what's blocked.

### Suggested session breakdown

| Session | Covers | Rough effort |
|---|---|---|
| 1 | M0 | 1 day |
| 2 | M1 | 1–2 days |
| 3 | M2 | 2–3 days |
| 4 | M3 | 3–4 days |
| 5–6 | M4 | 4–6 days |
| 7–8 | M5 | 5–7 days |
| 9–10 | M6 | 4–6 days |
| 11–13 | M7 | 6–9 days |
| 14–16 | M8 | 6–9 days |
| 17–18 | M9 | 3–5 days |
| side | Studies B & C | 4–6 days |

Call it 8–12 weeks part-time. If you need something presentable sooner, **Study C plus M0–M5 is a complete, defensible project on its own** — ship that first, then continue into the offload work.

---

## 11. Risks and how to handle them

| Risk | Mitigation |
|---|---|
| Kernel is fast but wrong | Parity tests from M2 onward; end-to-end token-identity test in M5 is non-negotiable |
| Top-k selection costs more than it saves | Benchmark selection in isolation in M7 before integrating; if it can't hit single-digit µs, fall back to a fixed threshold instead of exact top-k |
| Accuracy collapses at aggressive k | Sweep k and report the Pareto curve. A well-characterized limit is a result, not a failure |
| No T4 access | Any GPU works; adjust arch flags and re-derive the PCIe roofline. The *ratio* story survives; the absolute numbers change |
| Model doesn't fit even to offload | Drop to a 7B/14B headline model. The regime matters, not the parameter count |
| Scope creep into training/serving | Re-read §2. Say no |

---

## 12. Definition of done

- [ ] `make build && make test` green on a clean clone
- [ ] End-to-end greedy decode matches the HF reference token-for-token on the dev model
- [ ] Big model runs on a GPU too small to hold it
- [ ] Ablation table: dense offload → DIP → cache-aware DIP, with bytes/token, tok/s, and perplexity
- [ ] Roofline plot and accuracy-vs-speedup Pareto curve in the README
- [ ] Nsight counters reported for every hot kernel
- [ ] `docs/RESULTS.md` written, with every claim traceable to a CSV in `reports/`
- [ ] Every topic in §9 ticked off in `LEARNING_NOTES.md`
