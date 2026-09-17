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
- [x] FP16/BF16/FP32/TF32
- [x] accumulation order and error
- [x] symmetric vs asymmetric quantization
- [x] granularity (tensor/channel/group/block)
- [ ] zero-point
- [x] outlier channels
- [x] E8M0 and microscaling
- [x] fake quant vs real quant
- [x] calibration (min-max, percentile, MSE, AWQ)
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
- [x] RMSNorm
- [x] online softmax
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

### M2 -- RMSNorm, online softmax, FP16 GEMV (done)

2026-09-15. Code was written and committed in a GPU-less local session; this
entry covers what happened running it for real on the Colab T4 -- two bugs
that only exist on hardware, both worth remembering because they're the
kind that a GPU-less write-then-hope workflow can't catch in advance.

**Bug 1 -- NaN in `softmax_online` from combining two identity states.**
`block_reduce_softmax` pads unused warp lanes (this launch uses 256 threads
= 8 warps, so the second-stage warp-shuffle tree always has real values in
lanes 0-7 and identity padding in lanes 8-31) with `SoftmaxState{-INFINITY,
0.0f}`. The FlashAttention-style combine recurrence computes `a.l *
exp(a.m - m) + b.l * exp(b.m - m)` -- correct when at most one side is the
identity, but when the shuffle tree combines two identity states together
(which it does, structurally, whenever `num_warps < 32`), `a.m - m` becomes
`-inf - -inf = NaN`. `fmaxf` alone would have resolved `m` fine (it ignores
NaN operands), but `l` still gets poisoned because the *other* operand's
`l` is independently NaN by the same mechanism, and `real_l + NaN = NaN`
regardless of what `m` resolves to. Fix: use a finite very-negative
sentinel (`-1e30f`) as the identity's `m` instead of `-INFINITY`, so two
identities combine to `0 - 0 = 0` (finite) rather than `NaN`. The general
lesson -- and the reason this is worth writing down rather than just
patching -- is that `-INFINITY` is a dangerous reduction identity for any
recurrence that *subtracts* two instances of it, even though it's the
mathematically "obvious" choice for a running max.

**Bug 2 -- flaky GEMV parity tests, but the kernels were never wrong.**
`gemv_fp16_v1` and `gemv_fp16_v4_splitk(split=1)` failed intermittently
against a flat `max_abs_err < 1e-2` bound, with observed errors that were
suspiciously always exact powers of two (0.015625, then 0.03125 on a
rerun with fresh random data -- no fixed seed). That pattern is the
signature of FP16 ULP spacing, not a computation bug: GEMV output
magnitude scales as `sqrt(K)` for unit-variance random inputs (std ~64 at
K=4096, ~256 at K=65536, the two shapes these tests use), and at that
magnitude a single FP16 ULP is already 0.016-0.25 -- bigger than the flat
tolerance. Two *independently* fp32-accumulated dot products (the kernel's
summation order vs PyTorch's) can legitimately round to adjacent FP16
values with zero real error between them. It hit v1 and split=1
specifically (not v2/v3/other splits) because those two happen to sum the
most terms sequentially into a single accumulator before any tree
reduction, giving their rounding trajectory the most chances to land on
the "wrong side" of a representable-value boundary relative to PyTorch's
own reduction order. Fix: switch the GEMV parity checks to a
magnitude-scaled bound (`atol + rtol * |expected|`, the same shape as
`torch.allclose`) instead of a flat absolute one. RMSNorm and softmax
outputs stay near unit magnitude, so the flat bound was never actually
wrong for those -- this is specific to GEMV's `sqrt(K)`-scaled output.

Both fixes pushed and re-verified on the T4 (`make test`: 35/35 passed,
rerun three times to confirm the ULP flakiness was actually gone and not
just not-triggered). Results: `reports/m2_gemv_bandwidth.{csv,png}`,
`reports/m2_norm_softmax_splitk.csv`. `gemv_fp16_v3` hits 260.1 GB/s at
K=4096 against a 224 GB/s (70% of the 320 GB/s peak) bar -- comfortably
clears the M2 acceptance criterion. v1->v2 is roughly a 5x jump (46 ->
225 GB/s) from the same warp-shuffle-reduction win M1's sum-reduction
already demonstrated; v2->v3's vectorized `float4` loads add another
~15%, smaller than M1's reduction case because GEMV is already spending
more of its time on the shuffle-reduce and FMA work relative to load
instruction count. Split-K past 8-way stops helping (8-way: 232 GB/s,
32-way: 210 GB/s) -- more splits means more `atomicAdd` contention on the
same small set of output accumulators, so at some point added parallelism
loses to atomic serialization, mirroring the M1 naive-atomic-reduction
lesson from the opposite direction.

### M3 -- quantization library (done)

2026-09-15. Pure Python/PyTorch, no CUDA -- the whole point of doing this
work before any dequant kernel exists (PROJECT_SPEC.md M3's own framing:
"much easier to debug in Python"). Two infrastructure issues surfaced
before the quantization work itself, both worth remembering:

**The M2-fix-for-M3 problem, and the regression it caused.** `soinfer/quant/`
needed to be importable on this GPU-less dev machine, but `soinfer/__init__.py`
unconditionally did `from . import ops`, which imports the compiled
`soinfer._C` extension -- so `import soinfer` hard-failed here, and would
have taken `soinfer.quant` down with it. Fixed by making the `ops` import
lazy (`try/except ImportError: ops = None`). That fix broke something else,
though: `tests/test_m1_fundamentals.py` and `test_m2_kernels.py` reference
`soinfer.ops.reduce_naive_atomic` etc. directly inside `@pytest.mark.parametrize(...)`
decorator arguments, which Python evaluates at *module import time*, before
any `skipif` marker gets a chance to run. Previously `pytest.importorskip("soinfer")`
caught this (import failure -> skip, before the parametrize lines ever
executed); once `import soinfer` started succeeding with `ops = None`, those
lines hit `AttributeError: 'NoneType' object has no attribute '...'`
instead of skipping cleanly. Fix: an explicit
`if soinfer.ops is None: pytest.skip(..., allow_module_level=True)` right
after the importorskip line, before any parametrize decorator runs. Lesson worth
keeping: `pytest.importorskip` only protects against *import* failure, not
against a module that imports fine but leaves something you depend on
`None` -- decorator arguments are the sharp edge because they run at
collection time, not test time, so a module-level skip must come before
them explicitly.

**A local pytest run needs `soinfer` on `sys.path` without an editable
install.** `pip install -e .` needs to compile the CUDA extension via
`torch.utils.cpp_extension`, which needs `nvcc` -- not available here. Added
a root `conftest.py` that inserts `python/` onto `sys.path` directly, which
is exactly what an editable install's `.pth` file would do anyway; on Colab
(where the real editable install already exists) this is a harmless no-op.
This is what actually makes M3's "no CUDA needed" promise real rather than
aspirational -- without it, `soinfer.quant` was reachable in principle but
not in practice on this machine.

**The quantization library itself.** `formats.py` implements five symmetric
(no zero-point) granularities -- per-tensor, per-channel, group-wise
(default 128), block-32, and OCP-style microscaling -- all sharing one
`quantize`/`dequantize` pair; `mx_e8m0` is implemented as block32's exact
scale rounded to the nearest power of two via `round_to_pow2`, so the two
formats differ in *only* that one step, by design (this is what lets Study
B isolate the cost of power-of-two scales rather than conflating it with a
grouping-size difference). `calibrate.py` provides four scale strategies
(`min_max`, `percentile`, `mse_optimal`, and AWQ-style activation-aware
per-channel pre-scaling) as a `scale_fn(grouped, amax, qmax) -> scale`
callback pluggable into `quantize()`. `pack.py` implements the INT4
bit-packing decided in `csrc/include/layout.h`: two values per byte in AWQ
order (`[0,2,4,6,1,3,5,7]`), chosen so a future CUDA dequant kernel can pull
the four even-indexed and four odd-indexed values out of one 32-bit load
via two 16-bit masks with no further shuffling -- decided and written down
now (M3) specifically so M4's LOP3 dequant kernel doesn't have to
re-derive and re-test a packing scheme from scratch. All of it round-trips
exactly (`tests/test_quant_roundtrip.py`, 31 tests, including ragged K not
a multiple of 8 for packing and not a multiple of 32/128 for grouping).

**Reproducing the per-tensor INT4 collapse.** `bench/bench_m3_quant.py`
sweeps all five formats at 4 and 8 bits on a synthetic weight tensor
(Gaussian base, ~0.5% of *columns* -- i.e. input/K-channels, shared across
every row -- scaled 25x to stand in for the real "outlier feature" columns
reported in the LLM.int8() / AWQ literature). Results in
`reports/m3_quant_accuracy.csv`. At 4-bit: per-tensor forces 99.6% of
weights to exact zero, per-channel 99.4%, group-128 43.8%, block-32/mx_e8m0
~21.5%. The per-channel number is the interesting one and not a bug: this
project's `per_channel` granularity scales per *output row*, and the
injected outliers live in specific *input columns* shared by every row --
so a per-row scale is set by the same outlier columns no matter which row
you look at, and per-channel quantization gives zero protection against a
column-shared outlier. That is exactly the failure mode AWQ's activation-
aware per-channel *weight* scaling (implemented in `calibrate.awq_scale`)
exists to fix from the other direction: since the outlier can't be escaped
by choosing a different grouping axis, AWQ instead shrinks the quantization
error on those specific channels by scaling them relative to how much they
actually matter (their activation magnitude), rather than by grouping.

**The real WikiText-2 perplexity sweep (M3 task 4, now done).** Ran
`bench/bench_m3_perplexity.py` on Colab against Qwen3-1.7B (verified
SwiGLU MLP), all 5 formats at 4 and 8 bits, WikiText-2 raw test split.
Results in `reports/m3_perplexity.csv`. Two real bugs surfaced getting
this to actually run (neither is a quantization-logic bug -- both are the
kind of environment/API-drift issue that only shows up running against
real infrastructure, which this GPU-less dev machine can't do for anything
touching a real model):
- `load_dataset("wikitext", "wikitext-2-raw-v1", ...)` fails under current
  `datasets`/`huggingface_hub` versions -- the bare `"wikitext"` repo id
  needs a namespace now (`HfUriError: Repository id must be
  'namespace/name'`). Fixed: `"Salesforce/wikitext"`.
- `total_nll` was a CPU tensor (`torch.zeros()` defaults to CPU) accumulating
  `loss * n` where `loss` lives on `cuda:0` -- device-mismatch
  `RuntimeError`. Fixed by `.item()`-ing the loss into a plain Python float
  before accumulating; a scalar running sum has no device to get wrong.

**The result, and why it's worth having a real model to confirm the
synthetic one.** FP16 baseline perplexity: 18.52. At 8-bit every format
stays close (18.28-21.76 -- `mx_e8m0` pays the expected small premium for
power-of-two scales). At 4-bit: `per_tensor` perplexity is **11,054,325**
-- not "worse," a completely broken model -- while every other 4-bit
format lands in a normal 22-31 range (`per_channel` 31.3, `group128` 24.3,
`block32` 22.8, `mx_e8m0` 22.6). This is the exact per-tensor INT4 collapse
`bench_m3_quant.py`'s synthetic experiment predicted, now confirmed on
real weights -- and it's a sharper result than the synthetic one in one
respect: on the real model, `per_channel` (31.3) recovers most of the way
back toward the well-behaved formats, unlike the synthetic experiment
where per_channel was exactly as broken as per_tensor (99.4% vs 99.6%
exact-zero). That's not a contradiction, it's the synthetic experiment's
outlier shape showing its limits: it deliberately injected *column*-shared
outliers (the same input channels large in every row) specifically to
demonstrate that per-channel-by-row scaling gives zero protection against
that shape -- a real point, but real Qwen3-1.7B weight outliers evidently
aren't dominated by that particular shape, so per-row scaling recovers
real signal here that it couldn't in the adversarial synthetic case.
Lesson: a synthetic stress test can correctly demonstrate a *mechanism*
(per-tensor collapse; per-channel's blind spot to column-shared outliers)
without its exact severity numbers transferring to a real model whose
outlier structure differs -- which is exactly why M3 task 4 asked for the
real number rather than treating the synthetic table as sufficient.

### M4 -- quantized GEMV kernels (in progress)

2026-09-16. Wrote `gemv_w8a16` and `gemv_w4a16_group` (+ a second
LOP3-style dequant variant, `gemv_w4a16_group_lop3`) plus
`tests/test_m4_kernels.py` and `bench/bench_m4.py`, all authored locally
(this dev machine has no GPU -- see docs/DESIGN.md). **Not yet run on
hardware** -- `colab-mcp` failed to connect this session (`CONNECT_TIMEOUT`
on first `uvx` invocation, a known slow-first-connect issue per
docs/DESIGN.md, not retried successfully within the session). Nothing in
this entry is a verified result; it's a design record to pick up from once
Colab is reachable. Checklist boxes for GEMV/register-pressure etc. stay
unchecked until a real run confirms the kernels are actually correct, not
just argued to be.

**First attempt at register-caching the group scale broke coalescing --
caught by the benchmark, not the correctness tests.** The initial
`gemv_w8a16`/`gemv_w4a16_group` gave each of a warp's 32 lanes a private
*contiguous* chunk of K (instead of the usual `gemv_fp16_v2/v3` stride-32
split), reasoning that grouped quantization shares one scale across
`group_size` consecutive elements, and M4 task 2 explicitly calls out
re-reading that scale from global memory once per *element* as "the
classic performance bug" -- a contiguous chunk lets a lane's scale sit in
one register across a whole group instead of reloading it. This passed
every correctness test (17/17, including the exact-dequant basis-vector
tests and naive-vs-LOP3 agreement) on the real T4, but `bench_m4.py`
showed why correctness tests alone aren't enough: **10-20x SLOWER** than
`gemv_fp16_v3` despite moving 1/4 to 1/8 the bytes (e.g. 12-24 GB/s vs
FP16's ~240 GB/s). The bug: giving each lane a *private, far-apart* chunk
of K means that within one warp instruction, the 32 lanes read addresses
hundreds of bytes apart instead of 32 *consecutive* words -- exactly the
coalescing collapse M1's `strided_copy.cu` was built to demonstrate, just
reintroduced by accident while solving a different problem. Fixed by
reverting to the standard lane-strided pattern (`for k4 = lane; k4 < K4;
k4 += 32`, so all 32 lanes' reads in one iteration are 32 consecutive
words = one coalesced transaction) and instead just hoisting the scale
load to once-per-WORD (amortized over 4-8 elements) rather than once
per scalar element. That's nowhere near as aggressive an amortization as
"once per whole group," but it turns out coalescing dominates by an order
of magnitude at this problem size -- the marginal extra scale reads (a
tiny, L1/L2-resident array, and often a broadcast read since group_size is
usually >= the elements-per-warp-iteration) cost essentially nothing next
to a 10-20x memory-coalescing penalty. Lesson worth keeping: a
"structural" optimization argued purely from re-read counts, without
checking what it does to the access pattern of the much bigger tensor
(the weights) sitting right next to it, can lose badly -- and the fix
isn't visible from a correctness test, only from GB/s. (Benchmark results
after the fix are below, in the M4 close-out entry once the run
completes.)

**Deriving the LOP3 dequant by adapting a published trick, not inventing
one.** The well-known AWQ/FasterTransformer `dequantize_s4_to_fp16x2` bit
trick (mask a 4-bit field into an FP16 mantissa next to a power-of-two
exponent, so the *bit pattern itself* already encodes `1024 + nibble`, then
one `sub`/`fma` per half2 recovers the value with no int->float conversion
instruction) assumes nibbles are **unsigned** `[0,15]` with the sign
recovered via a flat `-8` (a zero-point-8 convention). This project's
`layout.h` packing is plain two's-complement signed 4-bit instead
(`sign_extend(n) = n<8 ? n : n-16`). Worked out by hand (see
`csrc/kernels/dequant.cuh`'s comment) that the two conventions agree after
XOR-ing every nibble's sign bit first: `sign_extend(n) == (n ^ 8) - 8` for
all 16 values of `n` (checked both branches concretely, e.g. `n=8`: ours
gives `8-16=-8`, `(8^8)-8 = 0-8 = -8` ✓.) So the plan is: XOR the whole
packed 32-bit word with `0x88888888` up front (one cheap, obviously-correct
op, not part of the risky bit-magic), then run the *unmodified* published
sequence, which is reassuring because that sequence is widely used in
production (AWQ, FasterTransformer, vLLM) rather than something hand-rolled
here. Also worked out, by tracing `AWQ_ORDER = [0,2,4,6,1,3,5,7]` against
the trick's own nibble grouping, that the trick's four output half2 lanes
land on `(v0,v1)`, `(v2,v3)`, `(v4,v5)`, `(v6,v7)` -- four *consecutive*
pairs along K, needing no shuffling before pairing with a plain contiguous
`float4` load of `x`. That confluence (AWQ's packing order + AWQ's dequant
trick happening to compose with zero extra shuffling) is exactly what
layout.h predicted back in M3 ("this is what makes the LOP3 dequant trick
work") -- satisfying to see the payoff materialize, though it still needs a
real GPU run to confirm the derivation didn't miss something. Implemented
via CUDA C half2 intrinsics (`__hsub2`/`__hfma2`) rather than hand-written
`asm volatile("lop3.b32 ...")`, deliberately: functionally the same
bit-pattern-construction technique the spec asks for, but the intrinsic
form is something I can actually reason about (and expect nvcc to lower to
real `LOP3.LUT` instructions on Turing where profitable) without betting
correctness on hand-typed inline PTX I have no way to compile-check here.

**Close-out: ran on a real T4 the same session.** `colab-mcp` reconnected
(the earlier `CONNECT_TIMEOUT` was exactly the known slow-first-`uvx`
issue -- pre-warming the `uv` cache locally with one direct invocation
fixed it for the next session). Correctness: all 17 `test_m4_kernels.py`
cases pass, including the exact-dequant basis-vector tests for both W8A16
and W4A16, and `test_w4a16_naive_and_lop3_agree` (bit-exact, confirming the
hand-derived LOP3 bit trick from the entry above is correct). Two real
bugs surfaced by the T4 build/test run, not visible from local Python
checks:
- `gemv_w4a16_group.cuh`/`gemv_w8a16.cuh` used `uint8_t`/`int8_t` without
  `#include <cstdint>` -- `cuda_fp16.h` alone doesn't pull it in, so nvcc
  failed with "identifier undefined." Trivial once seen, invisible without
  an actual nvcc invocation.
- The ragged-K test (`K=4099`) passed the wrong K to the kernel: `formats.
  quantize`'s `"group"` granularity zero-pads K up to a multiple of
  `group_size` *before* `pack.pack_int4` ever sees it, so the packed
  buffer is sized for the group-padded K (4224), not the original 4099.
  The kernel's own `TORCH_CHECK` caught the mismatch correctly -- a test
  bug, not a kernel bug, but a genuine "two independently-padded systems
  composing for the first time" gotcha worth remembering when any future
  code chains `formats.quantize` directly into `pack.pack_int4`.

**Then a real performance bug, caught only by benchmarking.** First
`bench-m4` run: 17/17 correctness tests green, but `gemv_w4a16_group_lop3`
measured **0.99x** vs `gemv_fp16_v3` -- barely tied, despite moving 1/4 the
bytes, and 10-20x slower than expected in raw GB/s. Root cause: the
contiguous-per-lane K-chunking described above (meant to cache the group
scale in a register) makes a warp's 32 simultaneous reads land hundreds of
bytes apart instead of 32 consecutive words, destroying coalescing --
exactly the failure mode M1's `strided_copy.cu` exists to teach, walked
right back into it while solving a different problem. Fixed by reverting
to the lane-strided access pattern from `gemv_fp16_v2/v3` (all 32 lanes'
reads in one iteration are 32 consecutive words) and hoisting the scale
load to once-per-word instead of once-per-group; correctness held (17/17
again) and throughput jumped to **1.7-1.9x** (run-to-run noise at this
problem size; `reports/m4_gemv_throughput.csv` has the full sweep).

Tried two further optimizations to close the gap to the spec's 3x bar,
both measured, neither kept:
- **More warps per block** (4 -> 8): no measurable change (1.73x vs
  1.74x) -- occupancy isn't the bottleneck here.
- **Wider per-lane reads** (`uint4` = 4 packed words = 32 elements/lane
  vs `uint32` = 8 elements/lane, to match `gemv_fp16_v3`'s 16-byte
  transactions): *regressed* to 1.62x. The wider weight read stays
  coalesced (lane `l`'s `uint4` index is still `l + 32*iter`), but getting
  32 elements per lane per iteration instead of 8 means the matching `x`
  reads (needed once per sub-word) are no longer coalesced across lanes
  (stride-4 `float4` reads), and that cost more than the wider weight
  transaction saved. Reverted cleanly (`git checkout --`) back to the
  1.7-1.9x version.

**Where that leaves M4:** numerics are solid (both acceptance-criteria
tests pass: dequant matches the M3 reference exactly, GEMV output matches
within 1e-2). Throughput does **not** meet the spec's "W4A16 >= 3x FP16 at
K=N=4096" bar -- it lands at 1.7-1.9x. The spec's own text calls 3x "a
realistic yield after overheads" off an ideal 4x; here overhead (per-
element unpack/decode work, and a memory-transaction size for INT4 that's
inherently 4x narrower than FP16's per the format itself) is eating more
than that framing anticipated. Closing this gap for real (rather than by
guessing-and-benchmarking one change at a time, which is what the two
failed attempts above were) needs Nsight Compute -- achieved occupancy,
memory throughput %, warp stall reasons -- which is explicitly M9's job
in PROJECT_SPEC.md, not M4's. Recorded here as an open gap rather than
quietly declared "done": M4's kernels are correct and meaningfully faster
than FP16 (a legitimate, real result), but the 3x acceptance bar is
unmet and that's the honest number to carry forward.

### M5 -- fused transformer kernels (in progress)

2026-09-16. Wrote tasks 1-4 (fused SwiGLU MLP, fused QKV projection, KV
cache append, decode attention) locally, all with PyTorch-reference tests
in `tests/test_m5_kernels.py`. **Not yet run on hardware** -- lost the
Colab tab mid-session (the browser bridge reconnected to a blank
notebook instead of the one with the M4 build; see
[[project-colab-workflow]]) and picked M5 up locally in the meantime
rather than block on it. Task 5 ("RMSNorm + quantize fusion") is skipped
as a scope conflict, not an oversight -- see docs/DESIGN.md's M5 entry:
this project's only real quant formats (W4A16, W8A16) keep activations in
FP16, so there's no activation-quantized format for an RMSNorm fusion to
emit into.

**Fetched the real Qwen3-1.7B config instead of guessing shapes.**
`curl https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json` (no
`transformers` install needed, no GPU needed, just the JSON file) gives
hidden_size=2048, intermediate_size=6144, num_attention_heads=16,
num_key_value_heads=8, head_dim=128 -- GQA with a 2:1 query:KV head
ratio. `tests/test_m5_kernels.py`'s decode-attention/KV-cache tests use
these exact numbers rather than arbitrary ones, so a shape bug that only
shows up at the real model's dimensions (e.g. an edge case in how 128
threads-per-block interacts with `block_reduce_sum`) has a chance of
surfacing now instead of only during actual M5/M6 model integration.

**Fused SwiGLU MLP (task 1): what "fusion" buys here, concretely.** The
naive path is 4 kernel launches -- `gate_proj` GEMV, `up_proj` GEMV, an
elementwise `silu(gate)*up`, `down_proj` GEMV -- and materializes gate and
up as two separate `[I]` buffers that immediately get read back for the
elementwise step. `swiglu_gate_up` (csrc/kernels/swiglu_fused.cu) collapses
the first three into one kernel: each warp handles one intermediate-channel
row `i`, reads ONE float4 chunk of `x` and reuses it for BOTH the gate and
the up dot product (rather than two separate kernels each re-reading all of
`x`), then writes `h[i] = silu(gate_i) * up_i` directly -- gate/up
pre-activations never touch global memory as their own buffers, only `h`
does. `down_proj` still needs the *complete* `h` before any output element
exists, so it's a second kernel (reusing `gemv_fp16_v3` as-is, no new code)
-- fusing across that boundary would need a grid-wide sync mid-kernel,
which isn't worth it for what's left to gain. Net: 4 launches -> 2, and one
fewer full round-trip of an `[I]`-sized buffer.

**Fused QKV projection (task 2): the win is launches, not shared compute --
so no new kernel at all.** Unlike gate/up, Q/K/V don't feed into a shared
elementwise op afterward, so there's no arithmetic to fuse the way SwiGLU's
dot products were. The only real lever at batch=1 (where GEMV kernels are
short and launch-overhead can dominate) is cutting 3 launches to 1, which
falls straight out of concatenating `Wq/Wk/Wv` into one `[q_dim+2*kv_dim,
H]` matrix ONCE at model-load time and calling the existing `gemv_fp16_v3`
on it -- `concat_qkv_weights` + `fused_qkv_projection` in ops.py, zero new
CUDA. Worth noticing when a "fusion" task doesn't need a kernel at all.

**KV cache append (task 3): contiguous layout, per layout.h's row-major
convention.** `[num_kv_heads, max_seq_len, head_dim]` half, head_dim
innermost so one head's whole history is one contiguous span (what decode
attention's per-timestep dot products want). Append is one block per KV
head copying `head_dim` contiguous halfs -- about as simple as a kernel
gets, correctness here is really about the *layout* choice, not the copy
itself. Paged (block-table) layout is noted as a stretch goal, not done.

**Decode attention (task 4): a correctness-first, sync-heavy first cut.**
One block per query head, `blockDim.x == head_dim` (one thread per feature
dim). For each cached timestep, every thread computes its dim's product,
`block_reduce_sum` (reused from M2's reduce.cuh) combines all `head_dim`
partials into one score, broadcast back to every thread via a `__shared__`
scalar (block_reduce_sum's result is only valid on thread 0 -- same
broadcast pattern `softmax_online.cu` already uses), then every thread
applies the *same* FlashAttention-style online-softmax rescale to its own
slice of a running output accumulator: `new_m=max(m,s); corr=exp(m-new_m);
p=exp(s-new_m); l=l*corr+p; acc[d]=acc[d]*corr+p*v[t,d]`. GQA head mapping
(`kvh = qh / (num_q_heads/num_kv_heads)`) matches HF's `repeat_kv` grouping
order exactly (checked against transformers' actual repeat/reshape, not
assumed). This is O(cur_len) block-wide `__syncthreads()` calls per head --
correct and simple, almost certainly slow at real sequence lengths (each
sync is a real cost, and cur_len can be in the thousands by late decode).
Explicitly scoped as "get the numerics right first" (mirrors this
project's own M2 progression: `softmax_twopass` before `softmax_online`,
`gemv_fp16_v1` before `v2/v3/v4`) -- a tiled/blocked-over-timesteps version
that cuts the sync count is the natural next kernel once this is verified
against HF, not a target for right now.

**Deliberately NOT in the attention kernel: RoPE and Qwen3's QK-norm.**
`decode_attention` takes `q` as already rotated (and, for Qwen3
specifically, already per-head RMSNorm'd -- Qwen3 applies `q_norm`/`k_norm`
to each head's Q/K before RoPE, which Llama does not do). Both are
logically separate preprocessing steps on Q/K before the attention math
proper, and getting RoPE's rotation convention and Qwen3's QK-norm exactly
byte-right without a live HF reference to check against felt like exactly
the kind of thing likely to be subtly wrong in a way only real numbers
would catch. Scoped out for now rather than guessed at; needed before
`test_layer_parity.py`/`test_end_to_end.py` can actually run, and is the
first thing to build once Colab is back.

**Close-out: verified on a real T4, same session, first try.** Found a new
Colab tab, set it to a T4 runtime, fresh clone, `make build` (all of
swiglu_fused.cu/kv_cache.cu/decode_attention.cu compiled clean), `pytest
tests/ -v` -- **95/95 passed**, including all of tasks 1-4: fused SwiGLU
matches the SiLU/matmul reference, fused QKV matches per-matrix GEMVs,
`kv_cache_append` writes exactly the target position and nothing else, and
`decode_attention` matches the reference at cur_len=1/17/300 plus the
combined append-then-attend test. No bugs surfaced this time -- unlike
M4's LOP3 kernel (which needed a from-scratch bit-trick derivation with
real risk of a subtle error) and unlike the M4 coalescing disaster, these
four kernels were architecturally simpler (no bit-packing, no novel access
pattern) and the design reasoning (block_reduce_sum broadcast, GQA
grouping order checked against HF's actual `repeat_kv`) held up against
real numbers on the first pass. Good data point: the M4 near-misses were
about genuinely hard problems (dequant bit tricks, memory coalescing under
a shared per-group scale), not a general sign that "nothing works without
three iterations" -- straightforward kernels built carefully can just work.

**RoPE and Qwen3 QK-norm, read from the actual transformers source rather
than from memory.** Same session: `inspect.getsource()` on
`qwen3_mod.rotate_half`, `apply_rotary_pos_emb`, `Qwen3RotaryEmbedding`,
`Qwen3Attention`, and `Qwen3RMSNorm` directly on Colab (transformers is
already installed there) instead of recalling the RoPE convention from
training data -- this project's own experience with the M4 LOP3 kernel
was exactly a case where a remembered/derived convention needed real
verification, so there was no reason to trust memory here when the actual
source was one `inspect.getsource()` away. Confirmed: QK-norm is plain
`RMSNorm(head_dim)` per head (so `qk_norm` in ops.py is a zero-new-code
rename of the M2 `rmsnorm` kernel), and RoPE is the standard "rotate-half"
/ NEOX convention (`cos`/`sin` built by duplicating a length-head_dim/2
`freqs` vector, `out = x*cos + rotate_half(x)*sin`). `rope.cu` implements
the reduced two-line-per-pair form of that formula directly (derivation in
the file's header comment). Both were tested against the real HF functions
(`test_apply_rope_matches_huggingface_qwen3`, `test_qk_norm_matches_huggingface_qwen3`),
not an independently-derived reference -- and both passed on the real T4
without any bugs.

**Close-out: a full decoder layer matches real HuggingFace Qwen3, verified
live.** With every M5 piece now built and RoPE/QK-norm done, assembled one
full `Qwen3DecoderLayer` forward pass (input RMSNorm -> fused QKV ->
QK-norm -> RoPE -> KV cache append -> decode attention -> `o_proj` ->
residual -> post-attention RMSNorm -> fused SwiGLU MLP -> residual)
entirely from `soinfer.ops` kernels, using weights pulled directly out of
a real (randomly-initialized, Qwen3-1.7B-shaped) `Qwen3DecoderLayer` --
same weights feed both my pipeline and the HF reference, so any mismatch
is purely a bug in the kernels/assembly, not a weight-loading issue.
First result: max abs diff ~0.002 (about 1 FP16 ULP), and a max *relative*
diff of 3.8% that looked alarming until checked -- it was one output
element near zero (`hf_out=0.00061`) where a tiny absolute error produces
a large ratio. Excluding elements with `|hf_out| < 0.05` (rounding noise,
not signal), max relative error is **0.26%**, well inside PROJECT_SPEC.md
M5's own acceptance bound (`< 2e-2`). Wrote this up as
`tests/test_layer_parity.py` -- exactly the milestone's own backbone test
("a full transformer block, your implementation vs HF, max relative error
< 2e-2 on real activations") -- parametrized over position 0/5/100 (0
matters specifically because RoPE is the identity there, so it's the one
case that *wouldn't* catch a RoPE bug; 5 and 100 do exercise real
rotation). All 3 pass on the real T4. This is M5's real headline result:
every non-skipped task (1-4, plus the RoPE/QK-norm work needed to actually
use them) composes into a working, HF-matching transformer layer.

**Still open for M5/M6:** `test_end_to_end.py` (greedy decode matching HF
token-for-token across a full multi-layer model + generation loop) is the
next real target -- this single-layer parity result is necessary but not
sufficient for it (error could still compound across layers/steps in a way
a single-layer test can't see). Also open: INT8 tensor-core GEMM (M4's
optional stretch task), and the paged KV-cache layout (M5 task 3's stretch
goal, currently contiguous-only).
