"""M8 task 2: which intermediate channels stay resident in VRAM ("hot"),
avoiding a host gather for those channels on every token that selects
them.

Two halves, deliberately kept independent:

1. Trace-driven POLICY SIMULATION (this module's *Policy classes below) --
   given a recorded sequence of per-token selected-channel-index sets
   (from generate.calibrate_channel_frequencies, or any online run), each
   policy computes the hit rate it WOULD achieve, with no GPU or kernel
   involvement at all -- a classic page-replacement-style trace
   simulation. This lets three policies (PROJECT_SPEC.md M8 task 2:
   "evaluate static-frequency vs LRU vs LFU-with-decay") be compared on
   pure CPU bookkeeping before committing to which one the real fused
   kernel (M8 task 3) needs to actually support.

2. HotCache -- the GPU-resident cache M8 task 3's fused kernel reads from,
   built once a policy's final channel set is chosen. Needs CUDA (lives in
   this module too, but every function that touches a torch CUDA tensor
   is clearly marked).

Every policy breaks ties by the LOWER channel index (PROJECT_SPEC.md M8
task 2: "ties must break deterministically... so runs are reproducible").
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import torch

if TYPE_CHECKING:
    # Only for type hints -- soinfer.runtime.generate imports soinfer.ops
    # at its own top level, which requires the compiled CUDA extension.
    # A real (non-TYPE_CHECKING) import here would break this module's
    # pure-CPU-simulation half's no-GPU importability.
    from soinfer.runtime.generate import StreamingModel


def _as_sorted_ints(selected: Sequence[int] | torch.Tensor) -> list[int]:
    """Every policy processes one token's selected-channel set in a fixed
    (ascending-index) order before touching any cache state. This matters
    because topk_threshold_select's own output order is UNSPECIFIED
    (atomicAdd-race order, see csrc/kernels/topk_select.cuh) -- if a
    policy's within-token eviction/insertion order depended on that
    incidental order, two runs over the identical logical selection could
    produce different cache contents, which is exactly the
    non-determinism PROJECT_SPEC.md M8 task 2 rules out."""
    if isinstance(selected, torch.Tensor):
        selected = selected.tolist()
    return sorted(int(c) for c in selected)


@dataclass
class SimulationResult:
    policy: str
    cache_size: int
    hits: int
    total: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


class StaticFrequencyPolicy:
    """M8 task 2: a fixed hot set, chosen ONCE from calibration
    frequencies (generate.calibrate_channel_frequencies) and never updated
    during serving. Cheapest of the three (zero per-token bookkeeping at
    serve time) but blind to any shift in which channels matter after
    calibration -- exactly the tradeoff PROJECT_SPEC.md M8 task 2 asks to
    evaluate against the two ADAPTIVE policies below."""

    name = "static"

    def __init__(self, frequencies: torch.Tensor | Sequence[int], cache_size: int):
        if isinstance(frequencies, torch.Tensor):
            frequencies = frequencies.tolist()
        num_channels = len(frequencies)
        cache_size = min(cache_size, num_channels)
        # sort by (-frequency, index): higher frequency first, ties broken
        # by the LOWER index (a smaller index sorts first at equal freq).
        order = sorted(range(num_channels), key=lambda c: (-frequencies[c], c))
        self.hot_set: frozenset[int] = frozenset(order[:cache_size])
        self.cache_size = cache_size

    def simulate(self, trace: Iterable[Sequence[int] | torch.Tensor]) -> SimulationResult:
        hits = total = 0
        for selected in trace:
            for c in _as_sorted_ints(selected):
                total += 1
                if c in self.hot_set:
                    hits += 1
        return SimulationResult(self.name, self.cache_size, hits, total)


class LRUPolicy:
    """M8 task 2: evicts the least-recently-selected channel on a miss
    when the cache is full. Fully online/adaptive -- needs no calibration
    pass at all, unlike StaticFrequencyPolicy.

    Each token's selected set is checked against the cache's state as of
    the END of the PREVIOUS token, all at once, before any of THIS token's
    own misses are inserted -- matching how the real per-token GPU
    HotCache actually works (build_dip_descriptors resolves every one of a
    token's dip_k channels against the cache's current state, then the
    cache updates once for the next token). Checking and mutating one
    channel at a time within a single token (an earlier version of this
    method did) breaks down whenever a token's own selection count
    exceeds cache_size -- real M8 traffic always does (dip_k=3072 vs.
    cache sizes of a few hundred): the token's OWN churn evicts everything
    carried over from the previous token before cross-token reuse is ever
    checked, silently producing a ~0% hit rate regardless of how skewed
    the real access pattern is. Caught by bench_m8_hot_cache.py on real
    Qwen3-1.7B data -- see docs/LEARNING_NOTES.md's M8 entry."""

    name = "lru"

    def __init__(self, cache_size: int):
        self.cache_size = cache_size

    def simulate(self, trace: Iterable[Sequence[int] | torch.Tensor]) -> SimulationResult:
        cache: "OrderedDict[int, None]" = OrderedDict()
        hits = total = 0
        for selected in trace:
            indices = _as_sorted_ints(selected)
            total += len(indices)
            misses = []
            for c in indices:
                if c in cache:
                    hits += 1
                    cache.move_to_end(c)
                else:
                    misses.append(c)
            for c in misses:
                if self.cache_size > 0 and len(cache) >= self.cache_size:
                    cache.popitem(last=False)  # evict least-recently-used
                if self.cache_size > 0:
                    cache[c] = None
        return SimulationResult(self.name, self.cache_size, hits, total)


class LFUDecayPolicy:
    """M8 task 2: evicts the lowest-(decayed)-frequency channel on a miss.
    Every `decay_every` tokens, all tracked frequencies are multiplied by
    `decay` (< 1), so old usage fades and the cache can adapt to a shift
    in which channels matter -- LRU's weakness is a channel that's usually
    hot but briefly quiet gets evicted purely for being untouched
    recently; LFU-with-decay keeps a channel's accumulated (decayed)
    importance in the picture, not just its last-touch time."""

    name = "lfu_decay"

    def __init__(self, cache_size: int, decay: float = 0.98, decay_every: int = 50):
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        self.cache_size = cache_size
        self.decay = decay
        self.decay_every = decay_every

    def simulate(self, trace: Iterable[Sequence[int] | torch.Tensor]) -> SimulationResult:
        freq: dict[int, float] = {}
        cache: set[int] = set()
        hits = total = 0
        for step, selected in enumerate(trace):
            for c in _as_sorted_ints(selected):
                total += 1
                freq[c] = freq.get(c, 0.0) + 1.0
                if c in cache:
                    hits += 1
                elif self.cache_size <= 0:
                    pass
                elif len(cache) < self.cache_size:
                    cache.add(c)
                else:
                    # evict the lowest (decayed) frequency in the cache;
                    # ties broken by evicting the HIGHER index first (i.e.
                    # keep the lower index) -- the same convention
                    # StaticFrequencyPolicy uses.
                    victim = min(cache, key=lambda x: (freq.get(x, 0.0), -x))
                    cache.discard(victim)
                    cache.add(c)
            if self.decay_every > 0 and (step + 1) % self.decay_every == 0:
                for k in list(freq.keys()):
                    freq[k] *= self.decay
        return SimulationResult(self.name, self.cache_size, hits, total)


def compare_policies(
    trace: Sequence[Sequence[int] | torch.Tensor], cache_sizes: Sequence[int],
    frequencies: torch.Tensor | Sequence[int] | None = None, **lfu_kwargs,
) -> list[SimulationResult]:
    """M8 task 2's "evaluate static-frequency vs LRU vs LFU-with-decay":
    runs all three policies over the SAME trace at each cache size and
    returns every result, so callers (bench_m8_hot_cache.py) can just dump
    this to a CSV. `frequencies` (required for the static policy) is
    typically the SAME trace's own aggregate per-channel counts --
    pass generate.calibrate_channel_frequencies's output summed, or reuse
    a separate calibration run; either is a legitimate choice depending on
    whether you want to test the static policy's realistic use (chosen
    from calibration data, evaluated on held-out serving traffic) or its
    best case (chosen from the same trace it's evaluated on)."""
    trace = list(trace)  # policies each need to walk it once; a generator would be exhausted after the first
    if frequencies is None:
        counts: dict[int, int] = {}
        for selected in trace:
            for c in _as_sorted_ints(selected):
                counts[c] = counts.get(c, 0) + 1
        num_channels = (max(counts) + 1) if counts else 0
        frequencies = [counts.get(c, 0) for c in range(num_channels)]

    results = []
    for cache_size in cache_sizes:
        results.append(StaticFrequencyPolicy(frequencies, cache_size).simulate(trace))
        results.append(LRUPolicy(cache_size).simulate(trace))
        results.append(LFUDecayPolicy(cache_size, **lfu_kwargs).simulate(trace))
    return results


# ---------------------------------------------------------------------------
# GPU-resident cache (needs CUDA + the compiled soinfer._C extension --
# NOT YET HARDWARE-VERIFIED as of the commit that adds this, see
# docs/LEARNING_NOTES.md's M8 task 3 entry for why).
# ---------------------------------------------------------------------------


@dataclass
class HotCache:
    """M8 task 2/3: the GPU-resident 'hot' channel cache the fused kernel
    (M8 task 3, ops.gemv_dip_fused_up/_down) reads from. Built ONCE per
    layer from a chosen set of hot channel indices -- typically
    StaticFrequencyPolicy.hot_set, picked by comparing policies via
    compare_policies above (see this module's docstring for why the two
    ADAPTIVE policies, LRU/LFU-with-decay, stay simulation-only for now
    rather than driving a real, continuously-updated GPU-resident cache:
    that would mean evicting/reloading rows mid-serving, a substantially
    bigger system than a cache fixed at calibration time).

    Holds BOTH up_proj's and down_proj_T's rows for every hot channel
    (both are needed to fully avoid a host gather for that channel), plus
    `slot_of` -- a [intermediate_size] int32 GPU tensor mapping a channel
    index to its cache slot (-1 if not cached) that
    ops.build_dip_descriptors (M8 task 3) reads directly.
    """

    slot_of: torch.Tensor  # [I] int32 CUDA, -1 if channel c is not cached
    hot_indices: torch.Tensor  # [C] int64 CUDA, ascending -- which channels are cached
    up_Wq: torch.Tensor  # [C, up_row_nbytes] uint8 CUDA
    up_scale: torch.Tensor  # [C, up_num_groups] float32 CUDA
    down_Wq: torch.Tensor  # [C, down_row_nbytes] uint8 CUDA
    down_scale: torch.Tensor  # [C, down_num_groups] float32 CUDA

    @property
    def cache_size(self) -> int:
        return self.hot_indices.shape[0]

    @staticmethod
    def build(model: "StreamingModel", layer_idx: int, hot_indices: Sequence[int] | torch.Tensor) -> "HotCache":
        """Gathers the given channels' up_proj/down_proj_T rows from the
        pinned host arena into GPU-resident cache tensors, using
        gather_rows_staged (M7 task 2) -- a ONE-TIME cost paid when the
        cache is built, not per token (unlike the per-token staging
        gather M7 task 4's DIP path pays for every selected channel,
        cached or not -- this is exactly the cost M8 removes for
        cache-resident channels)."""
        # Local imports: keep this module's pure-CPU simulation half (the
        # *Policy classes above) importable and runnable with no compiled
        # CUDA extension and no safetensors install at all -- verified on
        # a machine with neither.
        import soinfer.ops as ops
        from .load_hf_checkpoint import DOWN_PROJ_T_SUFFIX

        if isinstance(hot_indices, torch.Tensor):
            hot_indices = torch.sort(hot_indices.long().cpu())[0]
        else:
            hot_indices = torch.tensor(sorted(int(c) for c in hot_indices), dtype=torch.int64)
        C = hot_indices.shape[0]

        I = model.matrices["model.layers.0.mlp.up_proj.weight"].n
        slot_of = torch.full((I,), -1, dtype=torch.int32, device="cuda")
        hot_indices_cuda = hot_indices.cuda()
        slot_of[hot_indices_cuda] = torch.arange(C, dtype=torch.int32, device="cuda")

        up_lm = model.matrices[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
        downT_lm = model.matrices[f"model.layers.{layer_idx}.{DOWN_PROJ_T_SUFFIX}"]

        up_gpu = torch.empty(C, up_lm.handle.row_nbytes, dtype=torch.uint8, device="cuda")
        down_gpu = torch.empty(C, downT_lm.handle.row_nbytes, dtype=torch.uint8, device="cuda")
        if C > 0:
            # C == 0 (nothing cached, e.g. cache_size=0 or before calibration)
            # is a legitimate config -- skip the gather entirely rather than
            # calling gather_rows_staged with a 0-row staging buffer: a
            # freshly-.pin_memory()'d ZERO-element CPU tensor isn't actually
            # registered as pinned (there's nothing to page-lock), so the
            # binding's pinned-memory check rejects it. up_gpu/down_gpu are
            # already the correctly-shaped (0-row) empty CUDA tensors either
            # way, matching the empty-cache-buffer convention
            # test_m8_fused_gemv.py's degenerate all-staging tests use.
            up_host = model.store.matrix_view(up_lm.handle)
            up_staging = torch.empty(C, up_lm.handle.row_nbytes, dtype=torch.uint8).pin_memory()
            ops.gather_rows_staged(up_host, hot_indices, up_staging, up_gpu)

            down_host = model.store.matrix_view(downT_lm.handle)
            down_staging = torch.empty(C, downT_lm.handle.row_nbytes, dtype=torch.uint8).pin_memory()
            ops.gather_rows_staged(down_host, hot_indices, down_staging, down_gpu)
            torch.cuda.synchronize()  # both gathers must land before the cache is considered ready

        up_scale = up_lm.scale.index_select(0, hot_indices_cuda)
        down_scale = downT_lm.scale.index_select(0, hot_indices_cuda)

        return HotCache(
            slot_of=slot_of, hot_indices=hot_indices_cuda, up_Wq=up_gpu, up_scale=up_scale, down_Wq=down_gpu,
            down_scale=down_scale,
        )
