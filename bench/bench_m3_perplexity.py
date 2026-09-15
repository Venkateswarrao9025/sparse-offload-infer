"""M3 task 4 (PROJECT_SPEC.md sec 6): fake-quant WikiText-2 perplexity for
every (format, bit-width) pair on the dev model. Needs transformers +
datasets + a model download -- run on Colab
(notebooks/colab_bootstrap.ipynb), not on the GPU-less local dev machine.

Perplexity methodology: WikiText-2 raw test split, concatenated into one
string, chunked into non-overlapping windows of MAX_LENGTH tokens (no
sliding-window overlap -- simpler and faster than the overlapping-stride
recipe some perplexity guides use; stated here so the number is
reproducible and comparable to itself across formats, which is all this
project's ablation actually needs -- PROJECT_SPEC.md sec 7 only requires
comparing against your own baseline under the same methodology, not
matching a external published number exactly).
"""
import copy
import csv
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from soinfer.quant import formats
from soinfer.quant.quantize_model import fake_quantize_linear_layers

MODEL_ID = "Qwen/Qwen3-1.7B"
MAX_LENGTH = 1024
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer


def verify_swiglu(model) -> None:
    """PROJECT_SPEC.md sec 4: verify the model uses a SwiGLU (gate/up/down)
    MLP before trusting any DIP-related result later in the project."""
    mlp = model.model.layers[0].mlp
    for attr in ("gate_proj", "up_proj", "down_proj"):
        if not hasattr(mlp, attr):
            raise RuntimeError(f"{MODEL_ID} MLP has no {attr} -- not a SwiGLU MLP, DIP would not apply")
    print(f"[OK] {MODEL_ID} confirmed SwiGLU MLP (gate_proj/up_proj/down_proj present)")


@torch.no_grad()
def perplexity(model, tokenizer, text: str, max_length: int = MAX_LENGTH) -> float:
    device = next(model.parameters()).device
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    seq_len = input_ids.size(1)

    total_nll = torch.zeros((), dtype=torch.float64)
    total_tokens = 0
    for begin in range(0, seq_len, max_length):
        chunk = input_ids[:, begin : begin + max_length]
        if chunk.size(1) < 2:
            continue
        loss = model(chunk, labels=chunk).loss.double()
        n = chunk.size(1) - 1  # HF's internal label shift predicts n tokens from n+1 inputs
        total_nll += loss * n
        total_tokens += n
    return torch.exp(total_nll / total_tokens).item()


def main() -> None:
    model, tokenizer = load_model_and_tokenizer()
    verify_swiglu(model)

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    print(f"WikiText-2 test: {len(text)} chars")

    pristine_state = copy.deepcopy(model.state_dict())

    results = []
    fp16_ppl = perplexity(model, tokenizer, text)
    results.append({"format": "fp16_baseline", "bits": 16, "perplexity": fp16_ppl, "layers_quantized": 0})
    print(results[-1])

    format_configs = [
        ("per_tensor", "per_tensor", None),
        ("per_channel", "per_channel", None),
        ("group128", "group", 128),
        ("block32", "block32", None),
        ("mx_e8m0", "mx_e8m0", None),
    ]
    for bits in (8, 4):
        for name, granularity, group_size in format_configs:
            model.load_state_dict(pristine_state)
            config = formats.QuantConfig(bits=bits, granularity=granularity, group_size=group_size)
            n = fake_quantize_linear_layers(model, config)
            ppl = perplexity(model, tokenizer, text)
            results.append({"format": name, "bits": bits, "perplexity": ppl, "layers_quantized": n})
            print(results[-1])

    model.load_state_dict(pristine_state)  # leave the model clean for anything downstream

    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "m3_perplexity.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["format", "bits", "perplexity", "layers_quantized"])
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
