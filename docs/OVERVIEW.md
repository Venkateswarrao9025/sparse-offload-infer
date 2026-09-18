# What this project is

A from-scratch CUDA inference engine for running large language models that
are **bigger than the GPU's VRAM** — built kernel by kernel, from CUDA
fundamentals up through a working decode loop, on a single NVIDIA T4 (15 GB).
Every number anywhere in this repo is backed by a checked-in CSV in
`reports/`; nothing is asserted without a measurement behind it.

If you only read one section, read "The idea in one paragraph" and "What
actually got built," then look at the three plots in [README.md](../README.md).

---

## The idea in one paragraph

A modern LLM's weights don't fit in a cheap GPU's memory. The standard fix is
to keep weights in ordinary (host) RAM and stream them over PCIe into the GPU
as each layer needs them — but PCIe is roughly 25x slower than the GPU's own
memory bandwidth, so **the bottleneck becomes the transfer, not the
compute**. This project's answer has three layers, each built on the one
before it:

1. **Shrink the bytes**: quantize weights to INT4 (and INT8), so there are
   4x fewer bytes to move per weight than FP16.
2. **Overlap the transfer with compute**: stream weight `N+1` over PCIe on a
   second CUDA stream while the GPU is still computing with weight `N`, so
   the transfer is (mostly) hidden behind useful work instead of adding to
   the critical path.
3. **Don't transfer what you don't need**: in a transformer's SwiGLU MLP,
   most of the "intermediate" channels contribute almost nothing to any
   given token's output. Compute which channels matter *first* (a cheap
   top-k over the gate activation), then stream only those channels'
   weight rows — not the whole matrix. A GPU-resident cache holds the
   channels that turn out to matter across *many* tokens, so the ones
   that matter most never need to be re-streamed at all.

Layer 3 is the actual research contribution — everything else is standard
practice, executed carefully. It's called **Dynamic Input Pruning (DIP)**
in this repo, and the cache-resident version is **cache-aware DIP**.

## Why this is hard to get right (and why that's the point)

None of these three ideas is individually exotic. What makes this project
worth doing is that **every one of them can be silently wrong** in a way
that produces a plausible-looking number instead of an error:

- A quantization scheme can look fine on paper and collapse in practice
  (this project measured exactly that: naive per-tensor INT4 quantization
  drives 99.6% of weights to exactly zero and destroys the model — see
  [RESULTS.md](RESULTS.md)'s M3 section for the mechanism, not just the
  symptom).
- Overlapping two CUDA streams can *look* overlapped in a profiler while a
  synchronization bug quietly corrupts results only under specific timing
  conditions — this project found a real one (a missing "compute must
  finish reading a buffer before the next copy overwrites it" dependency)
  that was invisible on a 14B model and produced non-deterministic output
  on a 1.7B model, purely from timing luck.
- A kernel that selects the "same set" of items on every run can still
  return them in a *different order* — invisible if nothing downstream
  cares about order, silently wrong if something does (this project's
  own top-k kernel did exactly this, and it fed a floating-point
  summation that's order-sensitive, so the same selected channels
  produced a different final answer depending on GPU thread scheduling).

None of these are contrived examples — they're bugs this project actually
had, found by testing at real scale on real hardware instead of trusting a
unit test on a toy input. **The recurring lesson, documented throughout
[LEARNING_NOTES.md](LEARNING_NOTES.md), is that "verified" always needs a
second clause: verified *at what scale, on what hardware, checked for
reproducibility how*.** A claim that hasn't been checked that way isn't
wrong on purpose — it just hasn't been checked yet, and the gap between
"looks right" and "is right" is where this project spent a lot of its real
effort.

## What actually got built

Working backward from what a decode step needs, in the order it was built:

**A quantization library** (pure Python/PyTorch, no CUDA) that supports five
formats — per-tensor, per-channel, group-wise, block-32, and OCP
microscaling (E8M0) — with calibration methods (min-max, percentile,
MSE-optimal, AWQ-style reparameterization) and bit-exact pack/unpack for
INT4/INT8, including ragged shapes (a dimension not evenly divisible by the
group size).

**Hand-written CUDA kernels**, each validated against a PyTorch reference
before it was ever benchmarked: RMSNorm, online softmax, three generations
of FP16 GEMV (naive → warp-shuffle-reduced → vectorized), quantized GEMV for
W8A16 and W4A16 (with a bit-pattern-construction dequant path that beats the
scalar version by 1.3-2.3x), fused SwiGLU, KV-cache append, decode
attention, a single-kernel-launch top-k selector, a row-gather kernel (stage
selected rows into pinned memory, then one batched host→device copy instead
of many small ones), and the "cache-or-stream" fused GEMV that reads each
selected channel's weights from either a GPU-resident cache or a
freshly-streamed staging buffer, transparently, in one kernel.

**A streaming weight store**: quantized weights live in a pinned host-memory
arena; a `WeightPipeline` double-buffers the PCIe copies so weight `N+1`'s
transfer overlaps weight `N`'s compute, continuously across an entire
generation.

**Dynamic Input Pruning wired into a real decode loop**: for each token,
compute the gate projection densely (need every channel's magnitude to know
which ones matter), select the top-k, then only stream (or cache-hit) those
channels' up/down-projection rows — reducing bytes transferred per token
without touching accuracy any more than necessary.

**A calibration-driven hot cache**: run a representative passage through the
model once, record which channels get selected most often (the skew turns
out to be real and substantial — the top 25% of channels account for
~45% of selections, not 25%), and keep those resident on the GPU so the
*common case* costs zero transfer.

**A full evaluation harness**: an accuracy-vs-bytes-saved Pareto curve
(sweep how many channels get kept, measure perplexity and throughput at
each point against this project's own dense baseline), an ablation table
(dense → +DIP → +cache-aware DIP, isolating each layer's actual
contribution), Nsight Compute profiling of every kernel in the real decode
path (occupancy, memory throughput, register pressure, what's actually
limiting each one), a robustness pass (every kernel binding validates its
inputs and fails loudly instead of silently corrupting memory on a
malformed shape), and CI that runs the CPU-only half of the test suite on
every push.

## The headline result

> On a single NVIDIA T4, decoding Qwen3-1.7B quantized to group-128
> symmetric INT4 with weights offloaded to host memory, cache-aware DIP at
> k/I = 0.5 (cache = 20% of intermediate channels) reaches 7.29 tok/s vs
> 7.70 tok/s for plain DIP and 12.08 tok/s for dense offload with the same
> kernels — at a perplexity cost of 14.01x dense — while transferring 33.6%
> fewer bytes/token than dense.

The more interesting fact underneath that headline: **cache-aware DIP's
perplexity is bit-identical to plain DIP's**, confirmed across independent
runs and again across an 18-point sweep of cache sizes and pruning
fractions. That's a hardware-confirmed correctness property, not just a
design intention — caching changes *where* weight bytes come from and
never changes the arithmetic. See [README.md](../README.md) for the full
headline writeup, honest caveats included (this is measured on the 1.7B
model with one eval passage, not the larger 14B model the spec's own
example claim format uses — a deliberate, documented scope decision, not an
oversight).

## How the pieces fit together

```
   host RAM (pinned)                         GPU (T4)
  ┌─────────────────────┐                  ┌──────────────────────────┐
  │ quantized weights    │  PCIe, ~12 GB/s │                          │
  │ (INT4/INT8, packed)  │ ───────────────▶ │  WeightPipeline          │
  │                       │  double-buffered│  (2 CUDA streams,        │
  │  - always streamed:   │  async copies   │   overlap copy + compute)│
  │    q/k/v/o/gate proj  │                 │                          │
  │  - selectively        │                 │  decode step:            │
  │    streamed (DIP):    │                 │  rmsnorm → q/k/v/o GEMV  │
  │    up/down proj rows  │                 │  → rope → kv-cache       │
  │    NOT in the cache   │                 │  → attention → rmsnorm   │
  └─────────────────────┘                  │  → gate GEMV (dense)      │
                                             │  → top-k select           │
   GPU-resident HotCache                    │  → build descriptors      │
  ┌─────────────────────┐                  │    (cache-slot or         │
  │ up/down proj rows for │◀────────────────┤     staging-miss, per     │
  │ the ~10-20% of        │   cache hits    │     selected channel)     │
  │ channels selected      │   cost zero     │  → fused cache-or-stream  │
  │ most often (from a      │   transfer      │    GEMV (up, then down)  │
  │ calibration pass)       │                │  → residual add           │
  └─────────────────────┘                  └──────────────────────────┘
```

## The milestone story, in plain language

Each milestone builds directly on the last — skipping ahead was explicitly
disallowed by the project's own build plan (`PROJECT_SPEC.md`), because a
fast kernel with wrong numbers, caught three milestones later, is much more
expensive to fix than catching it immediately.

- **M0-M1**: harness and CUDA fundamentals — measure the real hardware
  (PCIe bandwidth, memory bandwidth) rather than trusting spec sheets, and
  build the basic reduction/transpose/vector-add kernels that every later
  kernel's design choices (coalescing, shared memory, warp shuffles) come
  from directly.
- **M2**: the first real kernels — RMSNorm, softmax, FP16 GEMV — each
  validated against PyTorch before being benchmarked at all.
- **M3**: the quantization library, entirely in Python/PyTorch — deliberately
  built and understood *before* writing a single quantized CUDA kernel, so
  the numerics were already trusted by the time performance work started.
- **M4**: quantized GEMV kernels (INT8, INT4) — where the "INT4 is smaller
  but not proportionally faster" finding first showed up (dequantizing on
  the fly costs real ALU time), which directly foreshadowed M6's much
  bigger finding.
- **M5**: the rest of a transformer decoder layer as fused kernels — RoPE,
  KV-cache append, decode attention, QK-norm.
- **M6**: weight streaming over PCIe — the roofline measurement that
  reframes the whole project (every weight transfer takes 5.5x-10.2x longer
  than the GEMV that consumes it, so compute speed was never the real
  bottleneck) and the real double-buffering race condition found and fixed.
- **M7**: Dynamic Input Pruning — select and stream only the channels that
  matter per token, with a real accuracy-vs-bytes-saved curve, not a
  hand-waved one.
- **M8**: cache-aware DIP — the calibration pass, the hot-channel cache, and
  the fused cache-or-stream kernel that's this project's actual centerpiece.
- **M9**: profiling, hardening, and the writeup — Nsight Compute on every
  hot kernel, a fuller ablation sweep, input validation, CI, and this
  documentation.

The full numbers for every milestone are in [RESULTS.md](RESULTS.md); the
full story of what went wrong and had to be fixed along the way — including
five real bugs found only by checking reproducibility at real scale, not by
a unit test — is in [LEARNING_NOTES.md](LEARNING_NOTES.md).

## What's explicitly out of scope

Said plainly, per this project's own stated principle that scoping honestly
is a signal of judgment, not a weakness:

- Training or fine-tuning — this is an inference-only engine.
- Multi-GPU or tensor parallelism.
- Activation quantization below INT8 — weights are quantized to INT4/INT8;
  activations stay FP16 throughout.
- Beating production inference engines (vLLM, TensorRT-LLM) on raw
  throughput in the regime where a model already fits in VRAM. This
  project's claim is specifically about the *offload-bound* regime, where a
  model does not fit.
- Two "supporting studies" named in the original project spec (a deeper
  microscaling-format benchmark, and a separate polished small-model INT8
  kernel suite) were scoped as optional side work, separate from the main
  milestone track, and were not built.

## Where to look next

- **[README.md](../README.md)** — the headline result, the three key plots,
  and how to reproduce them from a clean clone.
- **[RESULTS.md](RESULTS.md)** — every number this project has ever
  reported, organized by milestone, each one sourced from a checked-in CSV.
- **[LEARNING_NOTES.md](LEARNING_NOTES.md)** — the running engineering log:
  what was tried, what broke, what was learned, dated and in order,
  including every real bug this project found on real hardware and how it
  was actually diagnosed.
- **[DESIGN.md](DESIGN.md)** — specific decisions and deviations from the
  original build plan, with reasoning.
- **[../PROJECT_SPEC.md](../PROJECT_SPEC.md)** — the original build plan
  this whole project was executed against, milestone by milestone.
