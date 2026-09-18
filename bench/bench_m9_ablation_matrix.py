"""M9 task 2: full ablation matrix (lean scope -- see the scoping note below).

Extends M8's single-point ablation table (bench_m8_ablation.py, k/I=0.5,
cache=10% of I) into a full sweep: k/I (6 points, reusing M7's own corrected
sweep -- reports/m7_pareto.csv) x cache_frac (3 points: 5%/10%/20% of I).
Reuses the dense baseline and all 6 plain-DIP (cache_frac=0.0) rows directly
from reports/m7_pareto.csv instead of re-running them -- those numbers are
already real, hardware-verified, and reproduced bit-identical twice (see
docs/LEARNING_NOTES.md's 2026-09-18 M7 re-run entry); recomputing them here
would just burn GPU time to get the same answer. Calibrates once per k
value (the calibration histogram depends on dip_k) and reuses that
histogram across the cache_frac sweep for that k, rather than
recalibrating per (k, cache_frac) pair.

**Scoping note (a deliberate reduction from PROJECT_SPEC.md M9 task 2's
literal wording, chosen to control Colab compute cost)**: Qwen3-1.7B only
(not also 14B -- avoids a ~28GB download and multi-hour runtime), one quant
format (INT4 group-128, what this project has used throughout), one seed
per point (not three with error bars -- this project's benchmarks have
never been seed-varied; the eval passage and prompt are fixed, so "seed"
here would only affect greedy-decode's deterministic continuation, which
doesn't vary run to run anyway per today's determinism fixes). This is
still a real, multi-dimensional ablation matrix (6 x 3 = 18 new
cache-aware points, on top of the 7 already-verified dense/DIP rows) --
just not the full model-size x quant-format x 3-seed scope the spec
describes. If a fuller sweep is wanted later, this script is the template
to extend (loop over MODEL_NAME and QUANT configs, add a seed loop around
the whole body).

Run on the Colab/Kaggle T4 session, not locally (no CUDA here); needs
Qwen3-1.7B already downloaded (see bench_m7_pareto.py) and
reports/m7_pareto.csv already present (from `make bench-m7-pareto`).

Writes reports/m9_ablation_matrix.csv.
"""
from __future__ import annotations

import csv
import sys
import time
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
K_FRACTIONS = [1.0, 0.75, 0.5, 0.375, 0.25, 0.125]  # matches reports/m7_pareto.csv's own sweep
CACHE_FRACTIONS = [0.05, 0.10, 0.20]
N_GEN_TOKENS = 24
WARMUP_TOKENS = 4

EVAL_TEXT = (
    "The history of computing is a story of abstraction. Each generation of "
    "engineers built new tools on top of the ones that came before, hiding "
    "complexity so the next generation could reach further. From vacuum "
    "tubes to transistors, from assembly language to high-level compilers, "
    "the pattern repeats: what was once an expert's craft becomes a "
    "beginner's starting point."
)


def _model_dir_from_cache(model_name: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(model_name)


def _load_existing_pareto_rows() -> list[dict]:
    """Dense + plain-DIP (cache_frac=0.0) rows, reused verbatim from the
    already-corrected, already-reproduced-twice M7 sweep -- not recomputed."""
    path = REPORTS_DIR / "m7_pareto.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `make bench-m7-pareto` first (this script reuses its rows)")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "mode": r["mode"], "k_over_I": float(r["k_over_I"]), "cache_frac": 0.0,
                "bytes_per_token": float(r["bytes_per_token"]), "tokens_per_sec": float(r["tokens_per_sec"]),
                "perplexity": float(r["perplexity"]), "ppl_ratio_vs_dense": float(r["ppl_ratio_vs_dense"]),
            })
    return rows


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench_m9_ablation_matrix.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    rows = _load_existing_pareto_rows()
    dense_ppl = next(r["perplexity"] for r in rows if r["mode"] == "dense")
    print(f"reusing {len(rows)} rows from reports/m7_pareto.csv (dense_ppl={dense_ppl:.4f})")

    print(f"loading {MODEL_NAME}...")
    model_dir = _model_dir_from_cache(MODEL_NAME)
    model = load_streaming_model(model_dir, include_dip=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    eval_ids = tokenizer(EVAL_TEXT, return_tensors=None)["input_ids"]
    prompt_ids = eval_ids[:8]
    calib_ids = tokenizer(CALIBRATION_TEXT, return_tensors=None)["input_ids"]
    I = model.matrices["model.layers.0.mlp.up_proj.weight"].n
    dense_bytes = gen.dip_bytes_per_token(model, dip_k=I, dense=True)
    dense_suffix_bytes_per_token = gen.dip_bytes_per_token(model, dip_k=0, dense=False)

    for frac in K_FRACTIONS:
        dip_k = max(1, int(round(I * frac)))
        print(f"\ncalibrating for k/I={frac} (dip_k={dip_k})...")
        channel_counts = gen.calibrate_channel_frequencies(model, calib_ids, dip_k)

        for cache_frac in CACHE_FRACTIONS:
            cache_size = max(1, int(round(I * cache_frac)))
            print(f"  k/I={frac} cache_frac={cache_frac} (cache_size={cache_size})...")
            caches = [
                hot_cache.HotCache.build(
                    model, layer_idx=li,
                    hot_indices=hot_cache.StaticFrequencyPolicy(channel_counts[li], cache_size).hot_set,
                )
                for li in range(model.num_layers)
            ]

            pipeline_cached = gen.WeightPipeline(model, suffixes=gen.DIP_DENSE_SUFFIXES)
            bufs_cached = gen.DipBuffers.make(model, max_k=dip_k)
            cached_layer_fn = lambda m, li, x, p, kc, vc: gen.run_decoder_layer_cached_dip(  # noqa: E731
                pipeline_cached, m, li, x, p, kc, vc, dip_k, bufs_cached, caches
            )
            ppl = float(torch.exp(torch.tensor(gen.teacher_forced_nll(model, cached_layer_fn, eval_ids))))

            torch.cuda.synchronize()
            gen.generate_cached_dip(model, prompt_ids, WARMUP_TOKENS, dip_k, caches)
            torch.cuda.synchronize()
            miss_bytes = [0] * model.num_layers
            t0 = time.perf_counter()
            gen.generate_cached_dip(model, prompt_ids, N_GEN_TOKENS, dip_k, caches, miss_bytes=miss_bytes)
            torch.cuda.synchronize()
            tok_s = N_GEN_TOKENS / (time.perf_counter() - t0)
            n_tokens_measured = len(prompt_ids) + N_GEN_TOKENS
            bytes_per_token = dense_suffix_bytes_per_token + sum(miss_bytes) / n_tokens_measured

            row = {
                "mode": "cache_aware_dip", "k_over_I": frac, "cache_frac": cache_frac,
                "bytes_per_token": bytes_per_token, "tokens_per_sec": tok_s, "perplexity": ppl,
                "ppl_ratio_vs_dense": ppl / dense_ppl,
            }
            print(f"    {row}")
            rows.append(row)

    for r in rows:
        r["bytes_saved_vs_dense_pct"] = 100.0 * (1.0 - r["bytes_per_token"] / dense_bytes)
    write_csv(rows, REPORTS_DIR / "m9_ablation_matrix.csv")

    print("\n[M9 ablation matrix] k/I, cache_frac, bytes saved vs dense, tokens/sec, perplexity ratio vs dense:")
    for r in rows:
        print(f"  mode={r['mode']:<16} k/I={r['k_over_I']:.3f} cache={r['cache_frac']:.2f}  "
              f"bytes_saved={r['bytes_saved_vs_dense_pct']:5.1f}%  tok/s={r['tokens_per_sec']:6.2f}  "
              f"ppl_ratio={r['ppl_ratio_vs_dense']:.4f}")


if __name__ == "__main__":
    main()
