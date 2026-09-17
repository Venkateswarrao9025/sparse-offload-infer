"""M6 task 4 + acceptance: run a real 14B+ model, streaming its INT4-
quantized decoder-layer weights from pinned host memory, and check it
"generates coherent text on a 15 GB GPU" (PROJECT_SPEC.md M6 acceptance).

Downloads Qwen3-14B from HuggingFace (~28GB, apache-2.0, ungated) if not
already cached, then streams it into a pinned arena one tensor at a time
(soinfer.runtime.loader -- never materializes the full checkpoint in host
RAM, which is what makes a 14B+ model loadable on Colab free tier's
~10GB RAM budget). Coherence is inherently qualitative -- this writes the
generated text to reports/ for a human to read, same as the project's own
convention of never asserting a number without a CSV backing it, just
applied to text instead of a number.

Run on the Colab/Kaggle T4 session, not locally (no CUDA here, and no
safetensors installed -- see docs/DESIGN.md). Takes several minutes
(mostly the streaming-quantize pass, ~8 minutes for Qwen3-14B on a T4).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import write_csv

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MODEL_ID = "Qwen/Qwen3-14B"
GROUP_SIZE = 128
PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
]
N_NEW_TOKENS = 12


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench_m6_headline_model.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    import soinfer.runtime.loader as loader
    import soinfer.runtime.generate as generate

    print(f"downloading/locating {MODEL_ID}...")
    model_dir = snapshot_download(MODEL_ID, allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    t0 = time.time()
    model = loader.load_streaming_model(model_dir, group_size=GROUP_SIZE)
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s -- pinned arena {model.store.bytes_used / 1e9:.2f} GB")

    rows = []
    for prompt in PROMPTS:
        prompt_ids = tokenizer.encode(prompt)
        t0 = time.time()
        tokens = generate.generate_streaming(model, prompt_ids, N_NEW_TOKENS, eos_token_id=tokenizer.eos_token_id)
        gen_s = time.time() - t0
        text = tokenizer.decode(tokens)
        n_generated = len(tokens) - len(prompt_ids)
        tok_per_s = n_generated / gen_s if gen_s > 0 else 0.0
        row = {"prompt": prompt, "generated_text": text, "n_generated_tokens": n_generated,
               "gen_seconds": gen_s, "tokens_per_sec": tok_per_s}
        print(row)
        rows.append(row)

    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak VRAM: {mem:.2f} GB (of 15 GB on a T4)")

    write_csv(rows, REPORTS_DIR / "m6_headline_generation.csv")
    with open(REPORTS_DIR / "m6_headline_generation.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL_ID}\nLoad time: {load_s:.1f}s\nPinned arena: {model.store.bytes_used / 1e9:.2f} GB\n")
        f.write(f"Peak VRAM: {mem:.2f} GB\n\n")
        for row in rows:
            f.write(f"Prompt: {row['prompt']!r}\nGenerated: {row['generated_text']!r}\n")
            f.write(f"({row['n_generated_tokens']} tokens in {row['gen_seconds']:.2f}s = {row['tokens_per_sec']:.2f} tok/s)\n\n")
    print(f"wrote reports/m6_headline_generation.{{csv,txt}}")


if __name__ == "__main__":
    main()
