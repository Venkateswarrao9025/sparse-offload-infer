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
from typing import Iterable, Sequence

import torch


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
    pass at all, unlike StaticFrequencyPolicy."""

    name = "lru"

    def __init__(self, cache_size: int):
        self.cache_size = cache_size

    def simulate(self, trace: Iterable[Sequence[int] | torch.Tensor]) -> SimulationResult:
        cache: "OrderedDict[int, None]" = OrderedDict()
        hits = total = 0
        for selected in trace:
            for c in _as_sorted_ints(selected):
                total += 1
                if c in cache:
                    hits += 1
                    cache.move_to_end(c)
                else:
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
