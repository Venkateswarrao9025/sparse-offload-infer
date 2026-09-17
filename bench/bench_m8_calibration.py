"""M8 task 1: calibration pass -- run a few hundred calibration tokens
through the DIP decode path, record per-channel selection frequency per
layer, and plot the skew ("the skew is the story," PROJECT_SPEC.md M8
task 1: channel selection is not uniformly random across tokens -- some
channels are chosen far more often than others; M8 exploits that by
keeping the most-selected ones resident on the GPU).

Run on the Colab/Kaggle T4 session, not locally (no CUDA here); needs
Qwen3-1.7B already downloaded (see bench_m7_pareto.py).

Writes reports/m8_channel_frequencies.csv (long format: layer, channel,
count, frequency) and reports/m8_skew.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_text import CALIBRATION_TEXT
from harness import write_csv

from soinfer.runtime import generate as gen
from soinfer.runtime.loader import load_streaming_model

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MODEL_NAME = "Qwen/Qwen3-1.7B"
DIP_K_FRACTION = 0.5  # representative operating point (M7's chosen knee region)
PLOT_LAYERS = [0, 13, 27]  # early / middle / late -- does skew change with depth?


def _model_dir_from_cache(model_name: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(model_name)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench_m8_calibration.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    print(f"loading {MODEL_NAME}...")
    model_dir = _model_dir_from_cache(MODEL_NAME)
    model = load_streaming_model(model_dir, include_dip=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_ids = tokenizer(CALIBRATION_TEXT, return_tensors=None)["input_ids"]
    I = model.matrices["model.layers.0.mlp.up_proj.weight"].n
    dip_k = max(1, int(round(I * DIP_K_FRACTION)))
    print(f"calibrating on {len(token_ids)} tokens, dip_k={dip_k} (k/I={DIP_K_FRACTION}), {model.num_layers} layers...")

    channel_counts = gen.calibrate_channel_frequencies(model, token_ids, dip_k)
    n_tokens = len(token_ids)

    rows = []
    skew_summary = []
    for layer_idx, counts in enumerate(channel_counts):
        counts_cpu = counts.cpu()
        sorted_counts, _ = torch.sort(counts_cpu, descending=True)
        total_selections = sorted_counts.sum().item()  # == n_tokens * dip_k, sanity-checkable

        for c in range(I):
            cnt = int(counts_cpu[c].item())
            if cnt > 0:
                rows.append({"layer": layer_idx, "channel": c, "count": cnt, "frequency": cnt / n_tokens})

        # "the skew is the story": what fraction of all selections did the
        # top 10%/25% most-selected channels of THIS layer account for?
        top10_frac = sorted_counts[: I // 10].sum().item() / total_selections if total_selections else 0.0
        top25_frac = sorted_counts[: I // 4].sum().item() / total_selections if total_selections else 0.0
        never_selected = int((counts_cpu == 0).sum().item())
        skew_summary.append({
            "layer": layer_idx, "top10pct_channels_share": top10_frac, "top25pct_channels_share": top25_frac,
            "never_selected_channels": never_selected, "never_selected_frac": never_selected / I,
        })
        print(f"  layer {layer_idx:2d}: top 10% of channels account for {top10_frac:.1%} of selections, "
              f"top 25% account for {top25_frac:.1%}, {never_selected}/{I} channels never selected")

    write_csv(rows, REPORTS_DIR / "m8_channel_frequencies.csv")
    write_csv(skew_summary, REPORTS_DIR / "m8_skew_summary.csv")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    for layer_idx in PLOT_LAYERS:
        counts_sorted, _ = torch.sort(channel_counts[layer_idx].cpu(), descending=True)
        ax.plot(counts_sorted.numpy(), label=f"layer {layer_idx}")
    ax.set_xlabel("channel rank (most- to least-selected)")
    ax.set_ylabel("times selected (out of "
                   f"{n_tokens} calibration tokens)")
    ax.set_title(f"M8 task 1: per-channel selection frequency is highly skewed\n"
                 f"Qwen3-1.7B, dip_k={dip_k} (k/I={DIP_K_FRACTION}), {n_tokens} calibration tokens")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    out_path = REPORTS_DIR / "m8_skew.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
