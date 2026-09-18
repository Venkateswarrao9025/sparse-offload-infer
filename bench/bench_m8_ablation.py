"""M8 task 4: the ablation table -- PROJECT_SPEC.md calls this "the single
most important artifact in the repo." Three rows, same model/kernels/
hardware throughout (isolating the contribution of each stage, not
comparing against a different implementation):

  1. dense offload (M6): every weight streamed in full, every token.
  2. +DIP (M7): up_proj/down_proj_T reduced to the top-k selected channels
     per token, gathered fresh from the host arena every time.
  3. +cache-aware DIP (M8): the same top-k selection, but channels
     resident in a GPU HotCache (built once, from a calibration pass on a
     SEPARATE passage from the eval text below -- realistic use, not the
     policy's best case) cost zero transfer; only the staging misses are
     gathered.

For each row: bytes/token (instrumented directly -- dip_bytes_per_token
for rows 1-2, since bytes there are a pure function of dip_k; a real
miss_bytes accumulator for row 3, since cache-aware bytes/token is
data-dependent on this token's actual selection vs. the fixed cache and
can't be computed in closed form), tokens/sec (timed generation), and
perplexity (teacher-forced NLL, exponentiated, over the SAME eval
passage bench_m7_pareto.py uses, for comparability with that Pareto
curve's own dense_ppl number).

Run on the Colab/Kaggle T4 session, not locally (no CUDA here); needs
Qwen3-1.7B already downloaded (see bench_m7_pareto.py).

Writes reports/m8_ablation.csv.
"""
from __future__ import annotations

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
DIP_K_FRACTION = 0.5  # PROJECT_SPEC.md sec 7's own example claim uses k/I=0.5; matches M7's knee region
CACHE_FRACTION = 0.10  # 10% of I resident -- within bench_m8_hot_cache.py's already-verified 1%-20% sweep
N_GEN_TOKENS = 24
WARMUP_TOKENS = 4

# Identical to bench_m7_pareto.py's EVAL_TEXT -- duplicated (not imported;
# that script doesn't export it as a shared module the way
# calibration_text.py is) so this table's dense/DIP rows are directly
# comparable to that Pareto curve's own numbers, on the same passage.
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


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench_m8_ablation.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    print(f"loading {MODEL_NAME}...")
    model_dir = _model_dir_from_cache(MODEL_NAME)
    model = load_streaming_model(model_dir, include_dip=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    eval_ids = tokenizer(EVAL_TEXT, return_tensors=None)["input_ids"]
    prompt_ids = eval_ids[:8]
    I = model.matrices["model.layers.0.mlp.up_proj.weight"].n
    dip_k = max(1, int(round(I * DIP_K_FRACTION)))
    cache_size = max(1, int(round(I * CACHE_FRACTION)))
    print(f"eval passage: {len(eval_ids)} tokens; dip_k={dip_k} (k/I={DIP_K_FRACTION}); "
          f"cache_size={cache_size} (={CACHE_FRACTION:.0%} of I)")

    rows = []

    # --- Row 1: dense offload (M6) -------------------------------------
    print("\n[1/3] dense offload...")
    pipeline_dense = gen.WeightPipeline(model)
    dense_layer_fn = lambda m, li, x, p, kc, vc: gen.run_decoder_layer_streaming(pipeline_dense, m, li, x, p, kc, vc)  # noqa: E731
    dense_nll = gen.teacher_forced_nll(model, dense_layer_fn, eval_ids)
    dense_ppl = float(torch.exp(torch.tensor(dense_nll)))

    torch.cuda.synchronize()
    gen.generate_streaming(model, prompt_ids, WARMUP_TOKENS)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen.generate_streaming(model, prompt_ids, N_GEN_TOKENS)
    torch.cuda.synchronize()
    dense_tok_s = N_GEN_TOKENS / (time.perf_counter() - t0)
    dense_bytes = gen.dip_bytes_per_token(model, dip_k=I, dense=True)

    row_dense = {
        "mode": "dense", "k_over_I": 1.0, "cache_frac": 0.0, "bytes_per_token": dense_bytes,
        "tokens_per_sec": dense_tok_s, "perplexity": dense_ppl, "ppl_ratio_vs_dense": 1.0,
        "bytes_saved_vs_dense_pct": 0.0,
    }
    print(row_dense)
    rows.append(row_dense)

    # --- Row 2: +DIP (M7), no caching -----------------------------------
    print(f"\n[2/3] +DIP (k/I={DIP_K_FRACTION})...")
    pipeline_dip = gen.WeightPipeline(model, suffixes=gen.DIP_DENSE_SUFFIXES)
    bufs_dip = gen.DipBuffers.make(model, max_k=dip_k)
    dip_layer_fn = lambda m, li, x, p, kc, vc: gen.run_decoder_layer_dip(pipeline_dip, m, li, x, p, kc, vc, dip_k, bufs_dip)  # noqa: E731
    dip_nll = gen.teacher_forced_nll(model, dip_layer_fn, eval_ids)
    dip_ppl = float(torch.exp(torch.tensor(dip_nll)))

    torch.cuda.synchronize()
    gen.generate_dip(model, prompt_ids, WARMUP_TOKENS, dip_k)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen.generate_dip(model, prompt_ids, N_GEN_TOKENS, dip_k)
    torch.cuda.synchronize()
    dip_tok_s = N_GEN_TOKENS / (time.perf_counter() - t0)
    dip_bytes = gen.dip_bytes_per_token(model, dip_k=dip_k, dense=False)

    row_dip = {
        "mode": "dip", "k_over_I": DIP_K_FRACTION, "cache_frac": 0.0, "bytes_per_token": dip_bytes,
        "tokens_per_sec": dip_tok_s, "perplexity": dip_ppl, "ppl_ratio_vs_dense": dip_ppl / dense_ppl,
        "bytes_saved_vs_dense_pct": 100.0 * (1.0 - dip_bytes / dense_bytes),
    }
    print(row_dip)
    rows.append(row_dip)

    # --- Row 3: +cache-aware DIP (M8) -----------------------------------
    print(f"\n[3/3] +cache-aware DIP (k/I={DIP_K_FRACTION}, cache={CACHE_FRACTION:.0%} of I)...")
    print(f"  calibrating hot set on a SEPARATE passage ({len(CALIBRATION_TEXT.split())} words)...")
    calib_ids = tokenizer(CALIBRATION_TEXT, return_tensors=None)["input_ids"]
    channel_counts = gen.calibrate_channel_frequencies(model, calib_ids, dip_k)
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
    cached_nll = gen.teacher_forced_nll(model, cached_layer_fn, eval_ids)
    cached_ppl = float(torch.exp(torch.tensor(cached_nll)))

    torch.cuda.synchronize()
    gen.generate_cached_dip(model, prompt_ids, WARMUP_TOKENS, dip_k, caches)
    torch.cuda.synchronize()
    miss_bytes = [0] * model.num_layers
    t0 = time.perf_counter()
    gen.generate_cached_dip(model, prompt_ids, N_GEN_TOKENS, dip_k, caches, miss_bytes=miss_bytes)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    cached_tok_s = N_GEN_TOKENS / elapsed
    n_tokens_measured = len(prompt_ids) + N_GEN_TOKENS  # miss_bytes accumulated over prompt + generated steps
    dense_suffix_bytes_per_token = gen.dip_bytes_per_token(model, dip_k=0, dense=False)  # q/k/v/o/gate only
    cached_bytes = dense_suffix_bytes_per_token + sum(miss_bytes) / n_tokens_measured

    row_cached = {
        "mode": "cache_aware_dip", "k_over_I": DIP_K_FRACTION, "cache_frac": CACHE_FRACTION,
        "bytes_per_token": cached_bytes, "tokens_per_sec": cached_tok_s, "perplexity": cached_ppl,
        "ppl_ratio_vs_dense": cached_ppl / dense_ppl,
        "bytes_saved_vs_dense_pct": 100.0 * (1.0 - cached_bytes / dense_bytes),
    }
    print(row_cached)
    rows.append(row_cached)

    write_csv(rows, REPORTS_DIR / "m8_ablation.csv")

    print("\n[M8 ablation] dense offload -> +DIP -> +cache-aware DIP:")
    print(f"  {'mode':<18} {'bytes/token':>12} {'bytes saved':>12} {'tok/s':>8} {'ppl':>8} {'ppl ratio':>10}")
    for r in rows:
        print(f"  {r['mode']:<18} {r['bytes_per_token']:>12,.0f} {r['bytes_saved_vs_dense_pct']:>11.1f}% "
              f"{r['tokens_per_sec']:>8.2f} {r['perplexity']:>8.3f} {r['ppl_ratio_vs_dense']:>10.4f}")


if __name__ == "__main__":
    main()
