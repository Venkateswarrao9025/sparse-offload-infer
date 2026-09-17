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

**M5 done: end-to-end greedy decode matches HF token-for-token.** Built a
full (random-weight, Qwen3-1.7B-shaped, 4-layer) `Qwen3ForCausalLM`, took
HF's `model.generate(do_sample=False)` token sequence as ground truth, and
ran an independent greedy-decode loop built entirely from `soinfer.ops`
kernels -- the same per-layer assembly as `test_layer_parity.py`, looped
across all decoder layers and every generated position, with each layer
keeping its own growing KV cache across steps (embedding lookup and
argmax are the only two ops left as plain tensor indexing -- everything
from the first decoder layer onward, including the final norm and the
`lm_head` GEMV, is a `soinfer.ops` kernel call). Tried 6 different
(seed, prompt, layer-count) combinations interactively before writing the
test file: **exact token-for-token match, every time, no divergence.**
Wrote up 3 of those as `tests/test_end_to_end.py`, PROJECT_SPEC.md M5's
own words: "greedy decode... produces the identical token sequence as the
HF reference... this test is the backbone of the project." All pass on
the real T4; full suite is 104/104.

This closes M5's two named acceptance tests (`test_layer_parity.py` and
`test_end_to_end.py`) for the first time in this project, on the same
day the milestone's kernels were written -- worth noting because M4 took
several benchmark-and-fix iterations to reach its (still not fully met)
bar, while M5's correctness work went from "nothing written" to
"token-for-token match against real HF" in one session with zero
correctness bugs along the way, once RoPE/QK-norm were read from the
actual source instead of guessed. The difference: M4's remaining gap is a
*performance* problem (needs profiling tools this project doesn't have
easy access to yet); M5's tasks were *correctness* problems, and reading
the real reference implementation directly, rather than working from a
remembered convention, seems to have been what made the difference.

**Remaining for M5/M6:** the paged KV-cache layout (M5 task 3's stretch
goal, currently contiguous-only) and INT8 tensor-core GEMM (M4's optional
stretch task) are both un-started but explicitly optional per spec. The
real next milestone is M6 (offload: streaming weights over PCIe), which
is where this project's actual thesis (DIP, PCIe-bound offload) begins --
M5 was prerequisite plumbing, not the point.

### Back to M4: Nsight Compute finally answers what's actually limiting W4A16 GEMV

2026-09-16, later same session. `ncu` (Nsight Compute's CLI) is installed
and *works* on this Colab T4 -- no permission wall (a real risk with cloud
GPUs; worth having checked before promising anything). Profiled
`gemv_w4a16_group_lop3_kernel` at the canonical benchmark shape (K=N=4096)
with `ncu --set full`. The verdict, straight from the tool: **"Compute is
more heavily utilized than Memory"** -- Compute (SM) Throughput 68.9% vs.
Memory Throughput 51.9% of their respective peaks. This kernel is
ALU-bound, not memory-bound, which is the opposite of what "GEMV is
memory-bound" (true for FP16 GEMV, PROJECT_SPEC.md sec 3's own framing)
would suggest -- the INT4 dequant/decode work per element is expensive
enough to flip that. Supporting detail: ALU is the single highest-utilized
pipeline (55.3% of active cycles); achieved occupancy is a healthy 87.5%
(not an occupancy problem); registers/thread is 38 (no spilling); global
loads are already ~97% coalesced-efficient (30.9/32 bytes per sector
utilized) -- so the earlier coalescing fix from this session's first M4
pass was correct and isn't leaving anything on the table. The bottleneck
is real per-element instruction count, not access pattern or occupancy.

**Tried the obvious ALU-reduction lever -- it broke correctness, reverted
immediately.** The LOP3 dequant path's inner loop does 2 `__half22float2`
conversions + 2 float multiplies + 1 float add per half2 pair (8 ops for
2 elements). Native `__hmul2` can do the multiply as ONE half2 instruction,
converting only the *product* to float2 (1 conversion instead of 2) before
summing -- cuts the op count meaningfully. Patched it, rebuilt, ran the
existing test suite before ever benchmarking: `test_w4a16_naive_and_lop3_agree`
and both `test_w4a16_gemv_matches_reference[...]` cases (full K=4096/4099
dot products) failed the 1e-2/5e-3 tolerance, while the exact-dequant
basis-vector tests (single nonzero term) still passed. Diagnosis: rounding
each product to fp16 *before* summing, instead of multiplying in fp32,
compounds across ~4096 terms into materially larger total error than
before -- exactly the failure mode this project's fp32-accumulation
convention (common.cuh, every GEMV kernel so far) exists to prevent.
Reverted via `git checkout --` before ever measuring whether it was even
faster, because correctness comes first regardless of the payoff --
PROJECT_SPEC.md's own framing: "the single most common way this project
fails is writing a fast kernel that produces wrong numbers."

**Where this leaves M4's 3x gap:** now backed by real profiling data
instead of a guess, but not closed. The path forward would need to reduce
ALU work *without* sacrificing fp32-accumulated products -- e.g.
restructuring which operations happen in half2 vs. float (the dequant
bit-manipulation itself is integer ops, already cheap; the expensive part
is the float conversions and accumulation, which need to stay fp32-safe),
or a different algorithmic approach entirely (e.g. more elements decoded
per instruction via wider LOP3 batching, though the earlier `uint4`
per-lane-read experiment already showed that widening naively trades away
coalescing). This needs more iteration than fits in one sitting; recorded
here as a validated, data-backed open problem rather than a guess to try
next time.

### M6 -- offload: streaming weights over PCIe (in progress)

2026-09-16, later same session. Tasks 1-3 done and verified on the T4:
`weight_store.PinnedWeightStore` (one pinned host arena, row-addressable,
format-agnostic -- stores whatever `soinfer.quant.pack` already produced),
`stream_manager.StreamManager` (N CUDA streams + events, `prefetch`/`wait`
split so a transfer's async issue is separate from the GPU-side ordering
constraint that lets it overlap compute), and the roofline benchmark
(task 3). 10/10 new tests pass (arena addressing, overflow/duplicate-name
rejection, row gather, and a simulated multi-layer double-buffering
pipeline checked for buffer-reuse races); full suite 114/114.

**The roofline result, and why it matters more than M4's remaining gap.**
`bench/bench_m6_roofline.py`: for each of one Qwen3-1.7B-shaped decoder
layer's 5 weight matrices (qkv_proj, o_proj, gate_proj, up_proj,
down_proj), quantized to INT4-group128 and packed exactly as M4 expects,
measured real pinned-H2D transfer time (via the new StreamManager, not a
back-of-envelope PCIe-spec number) against the `gemv_w4a16_group_lop3`
kernel time that consumes it. Result (`reports/m6_roofline.csv`): **one
layer's total transfer time is 8.69x its total compute time** (2.31ms vs
0.27ms). Per-matrix ratios range 5.5x (o_proj) to 10.2x (gate/up_proj) --
smaller matrices have relatively more fixed transfer overhead, larger
ones scale more predictably with bytes. This is the plot PROJECT_SPEC.md
says "justifies the entire rest of the project," and it does: at ~9x
transfer-bound, the actual lever for making decode faster in the offload
regime is *reducing bytes transferred* (M7's whole premise -- only stream
the MLP rows that matter for this token), not further kernel
micro-optimization.

That reframes M4's still-open 3x-vs-1.7x-throughput gap: even if that
kernel were made *fully* memory-bound-efficient (some further multiple
faster), the system would still be transfer-bound by a wide margin --
9x would become somewhat more, not disappear. M4's gap is real and worth
closing eventually (it's the acceptance bar M4 itself set), but this
roofline is the first hard evidence in this project that kernel speed
isn't actually the bottleneck for the thing the project is ultimately
about. Good example of why M6 task 3 comes *before* the headline model in
the spec's own ordering -- the number changes what's worth optimizing
next.

**Not yet done:** M6 task 4 (bring in the real 14B/32B headline model) is
a much heavier resource commitment (multi-GB download, real VRAM/host-RAM
budget on the Colab instance) than anything else this session did --
holding off on it pending a decision with the user rather than just
downloading a large model unprompted. Nsight Systems timeline evidence of
*actual* overlap (not just the roofline's time comparison) is also not
yet captured -- the roofline shows transfer *should* dominate, but doesn't
by itself prove the double-buffering pipeline achieves good overlap
efficiency in practice; that needs `nsys profile` on a real multi-layer
decode loop.

**M6 task 4 done: a real 14B model generates coherent text from streamed
INT4 weights.** Checked with the user first (this needed a real resource
commitment -- confirmed Colab free tier, no direct monetary cost, but a
meaningful chunk of the free-tier GPU/RAM allowance) and picked Qwen3-14B
over 32B specifically because the RAM math mattered: Colab free tier has
~10GB available host RAM, and a 32B checkpoint's INT4-quantized form
(~16GB) plain doesn't fit, while 14B's does (~6.6GB).

The real engineering problem wasn't the GPU at all -- it was that
Qwen3-14B's checkpoint is **28GB in bf16**, comfortably bigger than the
~10GB of host RAM available. `transformers.AutoModelForCausalLM.
from_pretrained` would materialize the whole thing in RAM and OOM before
ever reaching quantization. The fix (`soinfer/offload/load_hf_checkpoint.py`):
stream straight from the safetensors shards using `safe_open`'s lazy
per-tensor access -- `get_slice(name).get_shape()` reads a tensor's shape
from the file's JSON header without touching its data (a cheap first pass
to size the pinned arena exactly, 6.606 GB, matching a hand-calculated
estimate exactly), then `get_tensor(name)` materializes ONE weight at a
time, which gets quantized, packed, and registered into the
`PinnedWeightStore` immediately, with the bf16 copy freed before the next
tensor. Peak *extra* RAM beyond the growing arena is one tensor (at most
a few hundred MB for this model) -- the full checkpoint is never resident
at once. This is the actual mechanism that makes "run a model bigger than
convenient RAM" work, not just "bigger than VRAM" (VRAM turned out to be
a complete non-issue here -- see below).

**Two real Colab kernel crashes along the way, and what they taught.**
Mid-session, the Colab Python kernel died and silently reconnected to a
fresh one twice (diagnosed via `ps aux` showing a `<defunct>` zombie
process and a new kernel PID/start time) -- almost certainly the Linux
OOM killer, from residual pinned-memory pages that `del` + `gc.collect()`
didn't actually release back to the OS (PyTorch's caching host allocator
holds onto freed pinned pages for reuse rather than returning them
immediately, so `free -h`'s "available" column stayed misleadingly low
after supposedly freeing tens of GB of Python objects). Lesson: for a
RAM-critical operation like this, don't trust incremental cleanup in a
long-lived interactive session -- verify the *actual committed script*
in a genuinely fresh kernel before trusting the result. Doing exactly
that caught nothing new (the refactored, modularized version reproduced
the interactive result byte-for-byte), but it was the right thing to
check rather than assume, given two unexplained crashes in the same
session.

**The result** (`reports/m6_headline_generation.{csv,txt}`): load
(stream-quantize all 40 layers) took 506s (~8.4 min) on a T4. Two
prompts, both coherent, both matching what a 14B model should plausibly
say:
- "The capital of France is" -> "...Paris. What is the capital of the
  United States?"
- "Once upon a time, there was a" -> "...young girl named Lily who lived
  in a small village surrounded by"

Peak VRAM: **3.78 GB of 15 GB** -- nowhere close to the limit, confirming
what the RAM analysis predicted: VRAM was never the constraint for this
architecture (embeddings + LM head + norms resident, ~2.3GB; small
per-layer staging buffers; a modest KV cache), host RAM during the
*loading* phase was. Throughput: ~1.1-1.3 tokens/sec -- slow, and
honestly reported as such: this run is deliberately correctness-first
(single-buffered streaming, no cross-layer prefetch overlap; unfused
Q/K/V, three separate GEMVs instead of M5's fused one). The M6 roofline
already measured the overlap opportunity (8.69x); wiring `StreamManager`
to actually prefetch layer i+1 while layer i computes, in this real loop,
is the natural next pass, not a fix for a correctness problem.

### M6 close-out: real double buffering, and what Nsight Systems actually showed

**Wired the overlap in for real.** `WeightPipeline` (new, in
`runtime/generate.py`) continuously prefetches one weight ahead across the
*entire* generation -- not reset per layer or per token, since the
weight-load sequence is completely static regardless of which tokens end
up generated, so there's nothing to gain from ever letting it drain. Two
generic byte buffers sized to the single largest weight, cycling through
the 7-per-layer role sequence via `itertools.cycle`.

**First result: no measurable speedup (14.51ms vs 14.60ms/layer) --
and initially I nearly mis-attributed this to a stale-module caching bug**
(this Colab kernel had been alive since before the pipelining commit, and
a plain `import` doesn't reload an already-cached module even after
`git pull` changes the file on disk -- `importlib.reload`/clearing
`sys.modules` was needed to actually test the new code). After forcing a
clean reload and re-confirming the timing was still flat, that ruled out
staleness as the explanation, which meant the "no speedup" result was
real and needed an actual explanation, not a retry.

**Nsight Systems settled it, and the answer wasn't what a quick guess
would predict.** `nsys` isn't on PATH on Colab but is installed (ships
with Nsight Compute, under `/opt/nvidia/nsight-compute/<version>/host/
target-linux-x64/nsys`) -- profiled `bench/profile_m6_overlap.py` (a small
synthetic Qwen3-14B-shaped model, isolating the pipelining *mechanism*
from any specific checkpoint) with `--capture-range=cudaProfilerApi`
bracketing the loop, then parsed the `cuda_gpu_trace` CSV export
(`bench/analyze_nsys_overlap.py`) into merged busy-intervals per stream
and computed their intersection. Two numbers, and they tell two different
stories:
- **Overlap efficiency (achieved/ideal): 94.0%** (`reports/m6_overlap_efficiency.json`,
  reproduced via the committed `make profile-m6-overlap` pipeline, not
  just the interactive exploration that found it -- consistent with an
  independent interactive run's 92.9%, run-to-run variance). When the
  copy stream and compute stream both have work queued, they overlap
  almost perfectly (9.88ms of actual concurrent copy+compute out of
  10.51ms ideal). The double-buffering mechanism itself is not broken --
  it's about as good as physically possible.
- **GPU idle 51.4% of wall-clock time** (86.2ms of 167.8ms) -- over half
  the total time, *neither* the copy stream nor the compute stream has
  anything running at all.

This resolves the "no speedup" result completely: the ~10ms saved by
overlap is real, but it's dwarfed by ~85ms of dead time where the GPU
sits idle waiting on the CPU side -- Python dispatch overhead between the
many small per-layer ops (`.view()`/`.contiguous()` calls, dict lookups
in `WeightPipeline`/`StreamingModel`, individual kernel-launch overhead
for ~30 tiny kernels per layer, `torch.cuda.current_stream().
wait_event()` calls). Both the single-buffered and pipelined versions pay
this SAME CPU-bound cost, which is why they measured identically --
the thing I changed (overlap efficiency) genuinely improved, but it
wasn't the bottleneck to begin with. Satisfies M6's actual acceptance
wording ("Nsight Systems timeline shows copy and compute kernels
genuinely overlapping... report the overlap efficiency (achieved vs
ideal)") with a real number, not a demonstration that dodges the harder
finding underneath it.

**A smaller, related false lead along the way, also worth recording:**
before reaching for `nsys`, tried a "fire all copies on 2 streams, sync
once at the end" micro-benchmark with no compute involved at all, to
sanity-check that concurrent transfers help. They didn't -- 21.2ms vs
15.1ms sequential, i.e. *worse*. In hindsight this makes sense and isn't
evidence against the pipeline design: two H2D copies issued concurrently
still contend for the same physical PCIe link, so there's no bandwidth to
gain from parallelizing transfer-with-transfer, only overhead to lose.
The real overlap this project wants is transfer-with-*compute* (different
hardware: DMA engine vs SM), which is exactly what the `nsys` numbers
above confirm is working.

**What actually would move the needle:** given the finding, the right
next optimization is reducing CPU dispatch overhead -- CUDA graphs
(capture the whole per-layer op sequence once, replay with near-zero
per-launch CPU cost) is the standard tool for exactly this shape of
problem (many small ops, batch-1 decode). Not implemented here -- a real
follow-up, not a same-session fix. M6 is otherwise complete: roofline
(8.69x transfer-bound), the real headline model (coherent generation,
verified), and now the overlap efficiency number the acceptance criteria
ask for.

### 2026-09-17 -- M7 task 1: top-k threshold-select kernel

**Correctness first, and it's solid.** `topk_threshold_select` finds the
top-k by *value* magnitude via bisection on a threshold tau rather than
sorting: maintain `[lo, hi]` bounding the correct tau, and on each
iteration every thread in a single block counts how many elements are
`>= mid` via a block-wide reduction (`block_reduce_sum`, reusing the same
shared-memory reduction primitive as `block_reduce_max`), then narrows
the range. 24 iterations is enough halvings for fp32 precision on
realistic activation magnitudes. A final compaction pass walks the array
once more, atomically claiming output slots for everything `>= tau`. All
8 tests in `tests/test_m7_kernels.py` pass on the T4, including an exact
set-equality check against `torch.topk` at Qwen3-14B's real
`intermediate_size` (17408) and a heavy-tie stress test (10 distinct
values repeated across 8192 elements) -- ties don't break it because the
compaction pass doesn't care about relative order among values `>= tau`,
only membership.

**The performance target is a genuine miss, and the reason is
structural, not a bug.** The spec's bar is single-digit microseconds at
I=17408, k=0.5*I. First measurement (`kThreads=256`): 639us -- about
100x over target. Root cause: the kernel launches a single thread block
(`<<<1, kThreads>>>`), so it only ever occupies one SM out of the T4's
40. Every one of the 24 bisection iterations does a full O(n) pass over
the array PLUS a `__syncthreads()`-gated block reduction, all serialized
on that one SM, while the other 39 sit idle. Bumping `kThreads` 256 ->
1024 (Turing's per-block max) gave a real 3.5x speedup -- 639us -> 180.9us
median at the I=17408/k=0.5*I point (`reports/m7_topk_timing.csv`,
`make bench-m7`) -- because more resident warps on that one SM hide more
memory latency during the linear scans. But it's still ~18x over the
single-digit-us target: more threads per block raises occupancy on ONE
SM, it doesn't recruit the other 39. That's a ceiling this design
structurally can't cross.

Recorded honestly rather than declared "good enough": task 1's spec
explicitly frames the multi-block/radix-select variants as the
follow-up once correctness is nailed down, and this single-block version
is deliberately the correctness-first baseline, not the final answer.
Two credible next steps, neither attempted yet: (1) a multi-block
version using cooperative groups' grid-wide sync (so all 40 SMs
participate in every bisection iteration's count, then one final
cross-block reduction), or (2) a proper radix-select (bucket by
exponent/mantissa bits, no per-iteration full-array rescan needed) --
the more standard GPU top-k approach and likely the bigger win, since it
avoids the 24x redundant full-array scan this bisection design pays for.

### 2026-09-17 -- M7 task 2: row gather (staged vs per-row cudaMemcpyAsync)

**The spec's prediction held up cleanly, first try.** Two variants,
both implemented as raw C++/CUDA functions (not Python loops -- see
below for why that matters) over a `PinnedWeightStore`-style matrix:
`gather_rows_staged` does a host-side `memcpy` of each selected row into
a contiguous pinned staging buffer, then ONE `cudaMemcpyAsync` H2D of the
whole block; `gather_rows_naive` issues one `cudaMemcpyAsync` per
selected row, straight from its scattered offset in the pinned arena.
11 correctness tests pass (byte-exact match against a CPU
`index_select` reference for both variants, at sizes up to Qwen3-14B's
real up/down-proj row shape, plus a direct check that the two variants
produce identical output given identical input -- they should differ
only in transfer pattern, never in result).

`bench/bench_m7_gather.py` sweeps the same I/k grid as task 1's topk
bench, using Qwen3-14B's real row shape (row_nbytes=2560, the INT4-packed
width of one row at hidden_size=5120). At the spec's own acceptance
point (num_rows=17408, k=0.5*num_rows): **staged=5.65ms vs naive=24.27ms,
a 4.29x slowdown for going row-by-row.** Across the full sweep the ratio
ranges 3.09x-12.68x, worse (not better) at smaller k in most rows -- makes
sense, since per-call overhead is a fixed cost per transfer, so it's a
proportionally larger tax when each transfer carries fewer bytes.
`reports/m7_gather_timing.csv` has the full I x k/I grid, all 18 points
naive-slower-than-staged. This is exactly the "compare against per-row
cudaMemcpyAsync (it will be far worse -- show the data)" the spec
predicts (PROJECT_SPEC.md M7 task 2), and it was true without needing a
second round of tuning -- unlike task 1, where the first correctness-first
design missed its performance target and needed real rework.

**Deliberately NOT implemented as Python-level `.copy_(non_blocking=True)`
loops, and that choice is itself worth recording.** M6's Nsight Systems
finding this session (see the 2026-09-XX M6 entry above) showed that at
small-per-op granularity, Python/ATen call dispatch overhead can dominate
measurements and mask the actual hardware-level effect being tested. If
"per-row transfer" had been benchmarked as N separate Python-level
`tensor.copy_()` calls, a measured slowdown could not be cleanly
attributed to "many small PCIe transfers are inherently worse than one
large one" -- it could just as easily be "Python dispatched N kernel
launches instead of 1," a different and much less interesting claim.
Doing both variants as tight C++ loops calling the CUDA runtime API
directly (`gather_rows.cu`, no `__global__` kernel needed -- the "kernel"
here is host-side orchestration of `cudaMemcpyAsync` calls, plus a plain
`std::memcpy` for the staging gather) isolates the actual claim the spec
is making, about transfer pattern, not about which language issues the
calls.

M7 task 2 acceptance is met: bytes transferred are instrumented directly
(`bytes_transferred` column, not inferred from timing), and the naive
baseline is measurably, substantially worse across the whole sweep.
Row gather from a pinned arena into a sparse GEMV's staging buffer is now
demonstrated as the right primitive; task 3 (sparse fused GEMV restricted
to the selected index set) is what actually turns this saved-bytes result
into a saved-bytes-*per-token* system-level number.

### 2026-09-17 -- M7 task 3: sparse fused GEMV (up reuses M4, down needs a new kernel)

**The two directions are not symmetric, and seeing why was the actual
task-3 insight -- the kernel work itself was small once that was clear.**
For token *t*: `gate_proj` must still be computed DENSELY (all I
channels) since selection needs `|gate_out|` for every channel before it
can decide which ones matter. Only `up_proj` and `down_proj` become
sparse. `up_proj` is `[I, H]`, naturally row-indexed by intermediate
channel -- so "restrict to the selected set" is *exactly* task 2's row
gather (already built) feeding straight into the *existing*
`gemv_w4a16_group_lop3` (M4) with `N=k` instead of `N=I`. No new kernel
needed; this was the first real payoff of designing task 2's primitive
generally back in M6/task 2 rather than special-casing it.

`down_proj` is different: standard nn.Linear layout is `[H, I]` (output-
row-major), so "restrict to selected channels" means selecting *columns*,
which are NOT contiguous in that layout -- gathering them would mean
scattered per-element reads inside every one of H rows, not a row gather
at all. Fix: store `down_proj` TRANSPOSED, `[I, H]`, so its rows are ALSO
indexed by intermediate channel, and task 2's row gather works unchanged
here too. But the compute shape flips: instead of "one row per output,
reduce along K" (every other GEMV kernel in this repo), down's transposed
form has no per-row output at all -- the k gathered rows collectively
define ONE H-length output vector, `y = sum_i h[i] * W_T[i,:]`, a
weighted row-sum. `gemv_w4a16_sparse_accumulate.cu` parallelizes by
OUTPUT COLUMN instead of by row: thread `c` owns output column `c` and
loops over the k selected rows, accumulating `h[i] * dequant(W_T[i,c])`.
Consecutive threads (consecutive `c`) read the SAME packed word within an
int4-group-of-8 for a given `i`, so this is a coalesced-broadcast read
pattern -- the literal transpose of `gemv_w4a16_group`'s lane-strided
pattern, matching the transposed shape of the math it computes.

**Correctness: clean on the kernel itself, one real snag in the
integration test -- and the snag taught something.** Two direct tests
(a one-hot basis-vector exactness check, and a reference-matched check
against `formats.dequantize`) passed on the first Colab run. A third
test -- tying task 1's `topk_select` + both sparse GEMVs together against
a "zero everything outside the selected set, then dense matmul" reference
-- failed on the first run (max diff 8.0 on outputs in the thousands).
The two kernel-level tests passing ruled out a kernel bug immediately, so
the question was what the integration test itself was doing differently.
Answer: floating-point addition isn't associative, and the reference
summed all I=512 channels (masked to zero) in natural index order while
the kernel accumulates only the k selected terms in `topk_select`'s
UNORDERED (atomicAdd-race) gather order -- different summation order over
~256 terms with sign cancellation, at output magnitudes in the thousands,
produces few-ULP-scale differences after the fp16 cast on its own, with
no bug anywhere. Fixed by selecting to the kernel's own gather order
*before* the reference matmul too (removes the avoidable "k terms vs 512
terms, most structurally zero" mismatch) and widening that one test's
tolerance to reflect it's comparing two valid summation orders, not a
kernel against a fixed-order reference of the same arithmetic. All 138
tests pass after the fix (`make test`, full suite, not just the new file).

M7 task 3 is functionally complete: both sparse GEMV paths exist and are
verified. Byte-savings-per-token instrumentation (task 4's actual
subject) and the k-sweep Pareto curve (perplexity vs tokens/sec) are the
remaining M7 work.

### 2026-09-17 -- M7 task 4: wiring DIP in surfaced a real M6 bug (non-deterministic greedy decode)

**The wiring itself went smoothly; a smoke test exposed something much
more important.** `run_decoder_layer_dip`/`generate_dip` (gate dense,
top-k select, gather up_proj/down_proj_T's k rows straight from the
pinned arena, both sparse GEMVs) came together with only one real fix
along the way: `generate.py` had hardcoded `GROUP_SIZE = 128` as a module
constant instead of reading whatever group_size a model's arena was
actually quantized with -- silently wrong for anyone loading with a
different group_size (every real caller in this repo happens to use 128,
so this never fired before M7 task 4's synthetic integration test used
32 to get a meaningful multi-group check at its small test dims). Fixed
by adding `group_size` to `StreamingModel` and reading `model.group_size`
throughout. First error: 493.5 max diff. After the fix: still non-trivial
(146.5), but now attributable to something real and different -- see below.

**Then `bench_m7_pareto.py` produced a nonsensical perplexity: 63 million
on Qwen3-1.7B** (worse than random guessing over a 151936-token vocab,
which would be ~150000). Before assuming the new eval code was wrong,
the cheapest differential check was comparing THIS project's own
`generate_streaming` against real HF generation on the same prompt --
Qwen3-1.7B's *streaming* path (`load_streaming_model`) had only ever been
validated end-to-end on the 14B headline model (M6, "coherent text," a
qualitative eyeball check); the smaller model was new territory. First
sign of trouble: printing the SAME greedy call's tokens twice gave
different token IDs. Greedy decode (do_sample=False, no randomness
anywhere in the math) producing different output on identical repeated
calls is impossible if everything is deterministic -- a dead giveaway of
an uncontrolled race, not a numerics problem. Four repeated calls
confirmed it: four different continuations from the identical prompt on
the identical model.

**Root cause: `StreamManager`'s double buffer only enforced one of the two
orderings a shared buffer needs.** `events[buf_idx]` made the compute
stream wait for a buffer's copy to finish before reading it (read-after-write,
handled). But nothing made the COPY stream wait for the PREVIOUS
GEMV's read of that same buffer to finish before overwriting it
(write-after-read -- missing entirely). `WeightPipeline` alternates 2
buffers, so buffer 0 gets read by GEMV call N, then isn't touched again
until the PREFETCH inside call N+2 overwrites it -- with no dependency
edge between "GEMV N reads buf 0" and "prefetch N+2 writes buf 0," the
two independent CUDA streams involved (default/compute stream, copy
stream 0) could race, and whichever finished first each time nondeterministically
decided whether the GEMV read old or new data. This is a genuine, if
narrow, race window: it only matters if the NEXT prefetch's copy can
plausibly start before the PREVIOUS GEMV's read finishes, which requires
copy latency to be comparable to (or shorter than) compute latency for
neighboring weights. Qwen3-14B's per-weight GEMV compute time is large
enough that this apparently never happened to lose the race in practice
(every M6 test and the "coherent text" check all happened to pass by
timing luck, not by correctness); Qwen3-1.7B's much smaller, faster
GEMVs made the race trigger essentially every call.

**This means M6's own acceptance evidence was quietly incomplete** --
"verified" meant "verified at 14B's scale," not "verified regardless of
model size," and the actual synchronization bug had been sitting in
this project's tested, merged, reviewed M6 code the entire time. Nothing
in M6's test suite (`test_offload.py`) caught it because those tests check
StreamManager's OWN correctness (does a prefetched buffer contain the
right bytes) in isolation, never a tight alternating read/write cycle
across many iterations under real timing pressure -- exactly the gap a
unit test's controlled pacing tends to paper over and only a longer,
faster, real workload exposes.

**Fix:** `StreamManager` gains `read_done_events` (one per buffer) and
`mark_read_done(buf_idx)`; `prefetch()` now makes the copy stream wait on
the buffer's `read_done_event` before writing (`stream.wait_event(...)`
on a never-recorded event is a documented no-op, which is exactly right
for a buffer's very first prefetch, before anything has ever read it).
`WeightPipeline.next_gemv` calls `mark_read_done(cur_buf)` immediately
after launching cur_buf's GEMV. Verified: 4/4 repeated greedy calls on
Qwen3-1.7B now give byte-identical output, matching real HF's own greedy
continuation ("The capital of France is Paris, and the capital of the
United States is" -- coherent, matches the shape of HF's own output).
Full test suite (143/143) still passes -- the fix adds ordering, it
doesn't change any buffer's final contents. `make profile-m6-overlap`
re-verification (does the extra wait dependency cost any of the 94%
overlap efficiency M6 reported) is a natural follow-up, not yet done.

**Why this is worth dwelling on:** this is the single most consequential
finding of the M7 session so far -- not because DIP's own kernels had a
bug (they didn't; tasks 1-3's kernel-level tests were never in question),
but because it demonstrates precisely the failure mode this project's own
rule ("verify on real hardware, never claim success without it") exists
to catch, AND shows that hardware verification at ONE scale doesn't
generalize to correctness at another. The lesson generalizes past this
one race: any claim of "verified" from here on should say verified on
WHAT, at WHAT scale, under WHAT load -- not just verified.

### 2026-09-17 -- M7 task 4: the k-sweep Pareto curve, and where the knee actually is

With the race fixed, `bench/bench_m7_pareto.py` on real Qwen3-1.7B gives a
sensible result on the first re-run (`reports/m7_pareto.csv`, teacher-forced
perplexity over a 70-token real passage, dense's own already-verified
streaming path as the reference):

| k/I | bytes/token saved | tokens/sec | perplexity | ratio vs dense |
|---|---|---|---|---|
| 1.0 (dense) | 0% | 12.06 | 16.50 | 1.00x |
| 1.0 (DIP) | 0% | 5.11 | 16.91 | 1.02x |
| 0.75 | 12.5% | 6.78 | 60.32 | 3.65x |
| 0.5 | 25.0% | 7.59 | 251.77 | 15.26x |
| 0.375 | 31.2% | 8.88 | 526.76 | 31.92x |
| 0.25 | 37.5% | 10.61 | 259446 | 15721x |
| 0.125 | 43.8% | 9.72 | 1117671 | 67723x |

Two things stand out, both sensible in hindsight but neither obvious in
advance:

**DIP at k=I (no pruning at all) is SLOWER than dense (5.11 vs 12.06
tok/s), despite moving identical bytes.** The extra cost is entirely
mechanism overhead the dense path doesn't pay: `topk_threshold_select`,
the `.cpu()` device-to-host sync `gather_rows_staged` needs (indices must
be host-resident for its host-side memcpy), and a full
`torch.cuda.synchronize()` per layer per token before the sparse GEMVs can
read the gathered buffers (`run_decoder_layer_dip`'s docstring already
flagged this synchronous, non-overlapped design as an honest limitation,
not an oversight). This is the SAME shape of finding as M6's CPU-dispatch
bottleneck: the mechanism has fixed per-call overhead that has to be
amortized by the bytes it saves, and at k=I there's nothing to amortize it
against.

**The knee is sharp, not gradual, and it's between k/I=0.375 and
k/I=0.25.** Perplexity degrades smoothly and tolerably down to k/I=0.375
(ratio 31.9x -- bad, but the output is presumably still recognizably
related to the passage), then EXPLODES by nearly three more orders of
magnitude at k/I=0.25 (ratio 15721x) and stays catastrophic at 0.125. This
is not a smooth accuracy-latency tradeoff curve with a rounded elbow --
it's closer to a cliff. Picking the knee: **k/I ≈ 0.375-0.5** is the
defensible operating point -- 0.5 keeps perplexity ratio to 15x (bad but
recoverable-looking) while saving 25% of bytes/token; anything below 0.375
is off a cliff this model/passage/k-selection-criterion combination can't
absorb. The honest caveat: this is ONE model (1.7B, small), ONE 70-token
passage, and top-k-by-raw-gate-magnitude is the simplest possible
selection criterion -- PROJECT_SPEC.md's own framing (M8's calibration
pass, hot-channel frequency) suggests real deployments would want
per-channel calibration data informing which channels are safe to prune,
not a fresh per-token magnitude cutoff with no memory of what mattered on
other tokens. The cliff observed here is plausibly THIS criterion's
weakness specifically (a single passage's channel-importance ranking
changing sharply once enough "usually-safe" channels get cut), not
necessarily inherent to DIP as an approach -- exactly what M8's
frequency-aware hot cache is positioned to investigate next.

M7's stated acceptance ("measurable reduction in bytes transferred per
token, instrumented directly... with perplexity degradation quantified
rather than hand-waved") is met: bytes/token are computed directly from
the arena's own bookkeeping (`dip_bytes_per_token`, not inferred from
timing), and perplexity is a real, quantified number at every point,
including the honest finding that it's not a smooth curve. M7 is
functionally complete (tasks 1-4 all done and verified); M8 is next.
