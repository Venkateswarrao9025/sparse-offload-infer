"""M8 task 2: "evaluate static-frequency vs LRU vs LFU-with-decay"
(PROJECT_SPEC.md). Runs a calibration passage through the DIP decode path
with per-token trace capture (generate.calibrate_with_trace), then
replays that trace through all three cache policies
(soinfer.offload.hot_cache.compare_policies) at several cache sizes,
purely as CPU bookkeeping -- no GPU work beyond producing the trace
itself.

Run on the Colab/Kaggle T4 session, not locally (no CUDA here); needs
Qwen3-1.7B already downloaded (see bench_m7_pareto.py).

Writes reports/m8_hot_cache_comparison.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_text import CALIBRATION_TEXT
from harness import write_csv

from soinfer.offload import hot_cache
from soinfer.runtime import generate as gen
from soinfer.runtime.loader import load_streaming_model

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MODEL_NAME = "Qwen/Qwen3-1.7B"
DIP_K_FRACTION = 0.5
LAYER_TO_ANALYZE = 13  # a middle layer -- see bench_m8_calibration.py's PLOT_LAYERS
CACHE_SIZE_FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2]  # fraction of intermediate_size kept resident


def _model_dir_from_cache(model_name: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(model_name)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench_m8_hot_cache.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    print(f"loading {MODEL_NAME}...")
    model_dir = _model_dir_from_cache(MODEL_NAME)
    model = load_streaming_model(model_dir, include_dip=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_ids = tokenizer(CALIBRATION_TEXT, return_tensors=None)["input_ids"]
    I = model.matrices["model.layers.0.mlp.up_proj.weight"].n
    dip_k = max(1, int(round(I * DIP_K_FRACTION)))
    print(f"tracing {len(token_ids)} tokens, dip_k={dip_k}, layer {LAYER_TO_ANALYZE}/{model.num_layers}...")

    channel_counts, trace = gen.calibrate_with_trace(model, token_ids, dip_k)
    layer_trace = trace[LAYER_TO_ANALYZE]
    layer_frequencies = channel_counts[LAYER_TO_ANALYZE].cpu().tolist()

    cache_sizes = [max(1, int(round(I * frac))) for frac in CACHE_SIZE_FRACTIONS]
    results = hot_cache.compare_policies(layer_trace, cache_sizes, frequencies=layer_frequencies)

    rows = [
        {
            "layer": LAYER_TO_ANALYZE, "cache_size": r.cache_size, "cache_frac": r.cache_size / I,
            "policy": r.policy, "hits": r.hits, "total": r.total, "hit_rate": r.hit_rate,
        }
        for r in results
    ]
    for row in rows:
        print(row)
    write_csv(rows, REPORTS_DIR / "m8_hot_cache_comparison.csv")

    print("\n[Hit rate by policy and cache size, layer", LAYER_TO_ANALYZE, "]")
    for cache_size in cache_sizes:
        line = f"  cache={cache_size:5d} ({cache_size/I:.1%} of I):"
        for r in results:
            if r.cache_size == cache_size:
                line += f"  {r.policy}={r.hit_rate:.1%}"
        print(line)


if __name__ == "__main__":
    main()
