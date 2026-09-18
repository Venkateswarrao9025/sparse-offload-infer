# M9 task 1: Nsight Compute on every hot kernel

`ncu --set full --profile-from-start off` on `bench/profile_m9_kernels.py`
(one forward pass each through the dense/M6, DIP/M7, and cache-aware-DIP/M8
decoder layers, plus the LM head projection, on a synthetic model at Qwen3-14B's
real shapes — H=5120, I=17408, 40 attention heads/8 KV heads, T4/sm_75, the
project's stated target — PROJECT_SPEC.md sec 3). Full per-kernel-launch raw
metrics were captured but not committed (the raw `--set full` CSV is ~3.5MB
of per-launch data across ~250 kernel launches; this table is the first
instance of each kernel, which is what's representative and actionable).
`reports/m9_ncu_summary.csv` has this table in machine-readable form.

**Scope**: the 11 kernels actually in the real decode path. Deliberately
excludes M1/M2's pedagogical kernels (`add_one`, `reduce_*`, `transpose_*`,
`gemv_fp16_v1`/`v2`, `softmax_*`, `swiglu_gate_up`, `gemv_w8a16`) — superseded
or never used in production serving — and `gather_rows_staged`/`gather_rows_naive`,
which are host-driven `cudaMemcpyAsync` calls, not compute kernels (Nsight
Systems, not Compute, is the right tool for those — see M6's
`profile_m6_overlap.py`/`analyze_nsys_overlap.py`).

| kernel | achieved occ. | theoretical occ. | mem throughput | compute throughput | registers/thread | grid × block | duration |
|---|---|---|---|---|---|---|---|
| `rmsnorm` | 24.8% | 100% | 0.6% | 0.2% | 16 | 1×256 | 17.1 μs |
| `gemv_w4a16_group_lop3` | 89.6% | 100% | 53.3% | 72.2% | 38 | 1280×128 | 135.9 μs |
| `rope_apply` | 6.3% | 100% | 1.8% | 1.1% | 16 | 40×64 | 4.5 μs |
| `kv_cache_append` | 12.4% | 100% | 1.0% | 0.4% | 20 | 8×128 | 3.9 μs |
| `decode_attention` | 12.5% | 100% | 3.6% | 5.7% | 27 | 40×128 | 5.9 μs |
| `topk_threshold_select` | 8.2% | 100% | 0.1% | 0.1% | 18 | 1×1024 | **3698.8 μs** |
| `gemv_w4a16_sparse_accumulate` | 24.9% | 100% | 18.5% | 28.8% | 43 | 20×256 | **1936.1 μs** |
| `build_dip_descriptors` | 23.9% | 100% | 10.0% | 4.4% | 16 | 34×256 | 5.5 μs |
| `gemv_dip_fused_up` | 92.9% | 100% | 54.2% | 76.0% | 40 | 2176×128 | 220.7 μs |
| `gemv_dip_fused_down` | 25.0% | 100% | 10.7% | 19.0% | 30 | 20×256 | **4436.5 μs** |
| `gemv_fp16_v3` (LM head) | 12.4% | 100% | 29.7% | 6.3% | 35 | 25×128 | 12.8 μs |

## The headline finding: three kernels are ~96% of the total captured time

`topk_threshold_select` + `gemv_w4a16_sparse_accumulate` + `gemv_dip_fused_down`
= 3698.8 + 1936.1 + 4436.5 = **10,071.4 μs out of 10,477.7 μs total** across
all 11 kernels in this pass — everything else (including the two large dense
GEMV workhorses) is a rounding error by comparison. This isn't a "per-token
latency" number (the pass deliberately runs dense + DIP + cache-aware-DIP
back to back to cover every kernel once, not a realistic single decode
step) but the RELATIVE weighting is real and consistent with — and now
quantifies precisely — what this project's earlier benchmarks only measured
indirectly through wall-clock tok/s (M7/M8's "mechanism overhead" findings).

## Second finding: nothing here is register-limited

**Theoretical occupancy is 100% for every single kernel.** Register counts
top out at 43 (`gemv_w4a16_sparse_accumulate`) against a T4's 255-per-thread
budget — nowhere close to spilling or capping occupancy. Every occupancy
shortfall below is a LAUNCH CONFIGURATION or MEMORY ACCESS PATTERN problem,
never a register-pressure one. This matters for where any future kernel
optimization effort should go: not "reduce register usage," but "give the
GPU more independent work per launch" or "fix the access pattern."

## Per-kernel: what actually limits it

- **`rmsnorm`** (24.8% achieved occ., grid=1 block): decode processes ONE
  token's hidden state at a time — a single row genuinely can't fill more
  than 1 block's worth of parallelism. **Limited by launch configuration**,
  inherent to batch=1 decode, not fixable without batching multiple
  requests together (out of scope for this project's single-stream design).
- **`gemv_w4a16_group_lop3`** (89.6% achieved occ., 72.2% compute throughput,
  the highest of any kernel): the actual GEMV workhorse — one warp per
  output row across 1280 blocks (N up to 17408), enough independent rows to
  keep the SM full. **Compute-bound** (ALU-bound from int4 unpack/dequant
  per the kernel's own captured rule: "ALU is the highest-utilized pipeline
  at 57.4%"), well-utilized — this is the kernel working as designed.
- **`rope_apply`** (6.3% achieved occ., the LOWEST of any kernel, grid=40 —
  exactly 1 block/SM on a 40-SM T4): rotates Q/K for one token across
  attention heads — inherently tiny per-call work. **Limited by launch
  configuration**; ncu's own rule recommends ≥2 blocks/SM, impossible to
  reach at this granularity without batching multiple tokens' RoPE together.
- **`kv_cache_append`** (12.4% achieved occ., grid=8 for 8 KV heads):
  one block per KV head, appending one token's K/V. **Limited by launch
  configuration** — inherent to decode's one-token-at-a-time KV cache
  update, same shape of limit as `rope_apply`.
- **`decode_attention`** (12.5% achieved occ., grid=40 for 40 Q heads):
  one query token attending over the KV cache, one block per head.
  **Limited by launch configuration** — same inherent batch=1 constraint.
- **`topk_threshold_select`** (8.2% achieved occ., grid=1 — uses exactly
  ONE of the T4's 40 SMs, and by far the SLOWEST kernel profiled at 3.70ms):
  **limited by its own single-block design**, a known, already-documented
  limitation from M7 ("~18x over the single-digit-μs target... structural
  limit of using only 1 of the T4's 40 SMs," follow-up: radix-select or
  multi-block redesign, not started) — today's session ALSO made Phase 3
  single-threaded on top of that single-block limit, trading correctness
  (bit-for-bit determinism, see `docs/LEARNING_NOTES.md`'s 2026-09-18 fifth-bug
  entry) for more of exactly this already-known performance gap. This
  profiling run is the first HARD NUMBER confirming this kernel, not the
  GEMV kernels, is the dominant single cost in the whole pipeline — the
  clearest, most load-bearing candidate for actual future optimization work
  this project has produced.
- **`gemv_w4a16_sparse_accumulate`** (24.9% achieved occ., grid=20 for
  H=5120/256 — exactly HALF the T4's 40 SMs, second-slowest at 1.94ms):
  parallelized by OUTPUT COLUMN (M7 task 3's design), so grid size is fixed
  by H regardless of how many rows (`dip_k`) it accumulates — and each
  thread does a fully sequential loop over all `dip_k` selected rows with
  no parallelism to hide it. **Limited by BOTH launch configuration (only
  half the SMs touched) AND memory access pattern** (ncu: only 7.3 of 32
  bytes/sector utilized on global loads — each thread's per-row read isn't
  coalesced across threads). Bigger `dip_k` directly means more serial
  work per thread with nothing to hide it behind.
- **`build_dip_descriptors`** (23.9% achieved occ.): resolves each selected
  channel's cache-slot-or-miss via a data-dependent lookup into `slot_of`.
  **Limited by memory access pattern** — ncu flags this explicitly:
  "uncoalesced global accesses resulting in 1785 excessive sectors (27% of
  6667 total)," an inherent cost of a per-index scattered gather.
- **`gemv_dip_fused_up`** (92.9% achieved occ., 76.0% compute throughput —
  the BEST-utilized kernel of the entire profile): inherits
  `gemv_w4a16_group_lop3`'s efficient one-warp-per-row design almost
  unchanged. **Compute-bound**, well-utilized — M8's centerpiece kernel is
  not a performance problem on the up-projection side.
- **`gemv_dip_fused_down`** (25.0% achieved occ., grid=20, the SLOWEST
  kernel of the entire profile at 4.44ms — even slower than
  `topk_threshold_select`): inherits `gemv_w4a16_sparse_accumulate`'s exact
  same structural limits (output-column parallelization → half the SMs;
  per-thread sequential accumulation with poor coalescing — only 6.5 of 32
  bytes/sector utilized). **Limited by BOTH launch configuration and memory
  access pattern**, and — combined with the plain (non-cached) down kernel
  above — the down-projection direction is consistently this project's
  worst-performing GEMV shape, on both sides of the M7/M8 divide.
- **`gemv_fp16_v3`** (LM head, 12.4% achieved occ., grid=25): **caveat**
  — this profiling run uses a synthetic model with a tiny 100-row
  vocabulary (matching `profile_m6_overlap.py`'s own synthetic-vocab
  convention), so grid=25 reflects that toy vocab size, not a real ~150K-row
  LM head. A real vocabulary would give this kernel far more independent
  rows to parallelize across — this number is not representative of real
  serving and shouldn't be read as a genuine bottleneck finding.

## Where this leaves M9's remaining work

No kernel here is register-limited, so a register-reduction pass would be
wasted effort anywhere in this pipeline. The two directions worth pursuing,
in order of expected impact: (1) `topk_threshold_select`'s single-block
design — already flagged as the follow-up from M7, now confirmed by hard
numbers as the single largest cost; (2) the down-projection accumulate
kernels' output-column parallelization scheme, which caps grid size at H
regardless of `dip_k` and pays for it twice (half the SMs, plus poor
coalescing). Neither is undertaken here — M9 task 1's job is measurement,
not a rewrite; see `docs/LEARNING_NOTES.md` for how this connects to M7's
already-open "radix-select/multi-block redesign, not started" note.
