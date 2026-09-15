"""M0 task 5: baseline decode throughput for the dev model.

Everything soinfer builds from M4 onward is judged against these numbers on
the same hardware: HF FP16, HF + torch.compile, and (if it builds) bitsandbytes
INT8. Run on the Colab/Kaggle T4 session, not locally (no CUDA here).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import write_csv

# Must be a SwiGLU (gate/up/down) MLP model -- verify before swapping this.
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
PROMPT = "The history of artificial intelligence began"
NEW_TOKENS = 128


def decode_tokens_per_sec(model, input_ids, new_tokens: int) -> float:
    gen_kwargs = dict(max_new_tokens=new_tokens, do_sample=False, pad_token_id=model.config.eos_token_id)

    # warmup (compilation, cudnn autotune, etc. should not count toward the timed run)
    with torch.inference_mode():
        model.generate(input_ids, **gen_kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(input_ids, **gen_kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = out.shape[1] - input_ids.shape[1]
    return generated / elapsed


def run_variant(name: str, model, tokenizer, rows: list[dict]) -> None:
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)
    torch.cuda.reset_peak_memory_stats()
    tok_per_s = decode_tokens_per_sec(model, input_ids, NEW_TOKENS)
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    row = {"variant": name, "tokens_per_sec": tok_per_s, "peak_vram_gb": peak_mem_gb, "new_tokens": NEW_TOKENS}
    print(row)
    rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench_baseline.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows: list[dict] = []

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    run_variant("hf_fp16", model, tokenizer, rows)

    compiled = torch.compile(model)
    run_variant("hf_fp16_compiled", compiled, tokenizer, rows)
    del model, compiled
    torch.cuda.empty_cache()

    try:
        bnb_model = AutoModelForCausalLM.from_pretrained(args.model, load_in_8bit=True, device_map="cuda")
        run_variant("bnb_int8", bnb_model, tokenizer, rows)
    except ImportError:
        print("bitsandbytes not installed -- skipping INT8 baseline (recorded as absent, not faked)")

    out_path = Path(__file__).resolve().parent.parent / "reports" / "m0_baseline.csv"
    write_csv(rows, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
