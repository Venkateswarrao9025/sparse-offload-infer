"""M7 task 4: k-sweep accuracy-vs-speedup Pareto curve. For k/I in
PROJECT_SPEC.md's own list, measures (a) bytes transferred per token,
instrumented DIRECTLY from the arena's own bookkeeping -- not inferred
from timing (M7's acceptance wording); (b) tokens/sec, timed generation;
(c) perplexity, teacher-forced negative log-likelihood over a short real
text passage, exponentiated -- against the dense (unpruned) path as the
reference, not an external ground truth, matching this project's own
established validation chain (the dense soinfer path was itself verified
token-for-token against real HF generation back in M5/test_end_to_end.py).

Uses Qwen3-1.7B (not the 14B headline model) -- small enough that a 6-way
k sweep with real generation + perplexity at each point finishes in a
reasonable time on a single T4 session. Run on the Colab/Kaggle T4
session, not locally (no CUDA here); needs the model already downloaded
(see bench_m6_headline_model.py / test_layer_parity.py, which use the
same model and populate the HF cache).

Writes reports/m7_pareto.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import write_csv

from soinfer.runtime import generate as gen
from soinfer.runtime.loader import load_streaming_model

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MODEL_NAME = "Qwen/Qwen3-1.7B"
K_FRACTIONS = [1.0, 0.75, 0.5, 0.375, 0.25, 0.125]  # PROJECT_SPEC.md M7 task 4's own sweep
N_GEN_TOKENS = 24  # tokens/sec measurement window
WARMUP_TOKENS = 4

# Short, fixed passage for the teacher-forced perplexity measurement -- see
# module docstring for why this project scores DIP against its own
# already-verified dense path rather than an external corpus/ground truth.
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
        raise SystemExit("bench_m7_pareto.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    print(f"loading {MODEL_NAME}...")
    model_dir = _model_dir_from_cache(MODEL_NAME)
    model = load_streaming_model(model_dir, include_dip=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    eval_ids = tokenizer(EVAL_TEXT, return_tensors=None)["input_ids"]
    prompt_ids = eval_ids[:8]
    print(f"eval passage: {len(eval_ids)} tokens; prompt for tokens/sec timing: {len(prompt_ids)} tokens")

    I = model.matrices["model.layers.0.mlp.up_proj.weight"].n
    rows = []

    # Dense baseline (k/I = 1.0's bytes are identical to dense, but the
    # DENSE path itself -- WeightPipeline over all 7 suffixes, no gather --
    # is the actual M6 code path, so time and score it directly rather than
    # assuming DIP at k=I behaves identically).
    print("dense baseline...")
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

    row = {
        "k_over_I": 1.0, "k": I, "mode": "dense", "bytes_per_token": dense_bytes,
        "tokens_per_sec": dense_tok_s, "perplexity": dense_ppl, "ppl_ratio_vs_dense": 1.0,
    }
    print(row)
    rows.append(row)

    for frac in K_FRACTIONS:
        dip_k = max(1, int(round(I * frac)))
        print(f"DIP k={dip_k} (k/I={frac})...")

        pipeline_dip = gen.WeightPipeline(model, suffixes=gen.DIP_DENSE_SUFFIXES)
        bufs = gen.DipBuffers.make(model, max_k=dip_k)
        dip_layer_fn = lambda m, li, x, p, kc, vc, _k=dip_k, _b=bufs: gen.run_decoder_layer_dip(
            pipeline_dip, m, li, x, p, kc, vc, _k, _b
        )
        nll = gen.teacher_forced_nll(model, dip_layer_fn, eval_ids)
        ppl = float(torch.exp(torch.tensor(nll)))

        torch.cuda.synchronize()
        gen.generate_dip(model, prompt_ids, WARMUP_TOKENS, dip_k)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen.generate_dip(model, prompt_ids, N_GEN_TOKENS, dip_k)
        torch.cuda.synchronize()
        tok_s = N_GEN_TOKENS / (time.perf_counter() - t0)

        bytes_per_token = gen.dip_bytes_per_token(model, dip_k=dip_k, dense=False)
        row = {
            "k_over_I": frac, "k": dip_k, "mode": "dip", "bytes_per_token": bytes_per_token,
            "tokens_per_sec": tok_s, "perplexity": ppl, "ppl_ratio_vs_dense": ppl / dense_ppl,
        }
        print(row)
        rows.append(row)

    write_csv(rows, REPORTS_DIR / "m7_pareto.csv")

    dip_rows = [r for r in rows if r["mode"] == "dip"]
    print("\n[Pareto] k/I, bytes saved vs dense, tokens/sec, perplexity ratio vs dense:")
    for r in dip_rows:
        bytes_saved_pct = 100.0 * (1.0 - r["bytes_per_token"] / dense_bytes)
        print(f"  k/I={r['k_over_I']:.3f}  bytes_saved={bytes_saved_pct:5.1f}%  "
              f"tok/s={r['tokens_per_sec']:6.2f}  ppl_ratio={r['ppl_ratio_vs_dense']:.4f}")


if __name__ == "__main__":
    main()
