"""M3 accuracy table (PROJECT_SPEC.md sec 6, M3 acceptance): for every
(format, bit-width) pair, report the fraction of weights forced to exact
zero and the relative reconstruction error.

This uses SYNTHETIC weight tensors (Gaussian base + injected per-channel
outliers, the qualitative shape of a real transformer weight matrix), not a
real model -- it demonstrates the per-tensor INT4 collapse mechanism
honestly without claiming a result it didn't measure. The real WikiText-2
perplexity sweep on the dev model (PROJECT_SPEC.md M3 task 4) is a separate,
larger step (needs transformers/datasets + a model download) tracked
separately; do not read this CSV as that result.
"""
import csv
import os

import torch

from soinfer.quant import formats

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def make_synthetic_weight(rows: int, cols: int, seed: int) -> torch.Tensor:
    """Gaussian base with a handful of injected large-magnitude outlier
    channels -- real LLM weight matrices reliably show a small number of
    channels with much larger magnitude than the bulk (the whole reason
    per-channel/group granularity and AWQ-style scaling exist)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g)
    num_outlier_cols = max(1, cols // 200)
    outlier_cols = torch.randperm(cols, generator=g)[:num_outlier_cols]
    w[:, outlier_cols] *= 25.0
    return w


def sweep() -> list[dict]:
    torch.manual_seed(0)
    w = make_synthetic_weight(rows=512, cols=4096, seed=0)
    configs = [
        ("per_tensor", formats.QuantConfig(bits=0, granularity="per_tensor")),
        ("per_channel", formats.QuantConfig(bits=0, granularity="per_channel")),
        ("group128", formats.QuantConfig(bits=0, granularity="group", group_size=128)),
        ("block32", formats.QuantConfig(bits=0, granularity="block32")),
        ("mx_e8m0", formats.QuantConfig(bits=0, granularity="mx_e8m0")),
    ]
    rows = []
    for bits in (4, 8):
        for name, base_config in configs:
            config = formats.QuantConfig(bits=bits, granularity=base_config.granularity, group_size=base_config.group_size)
            qt = formats.quantize(w, config)
            recon = formats.dequantize(qt)
            frac_zero = formats.fraction_exact_zero(qt)
            rel_err = ((recon - w).abs().sum() / w.abs().sum()).item()
            rows.append(
                {
                    "format": name,
                    "bits": bits,
                    "frac_exact_zero": frac_zero,
                    "rel_l1_recon_error": rel_err,
                }
            )
            print(rows[-1])
    return rows


def main() -> None:
    rows = sweep()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "m3_quant_accuracy.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["format", "bits", "frac_exact_zero", "rel_l1_recon_error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")

    pt4 = next(r for r in rows if r["format"] == "per_tensor" and r["bits"] == 4)
    g128_4 = next(r for r in rows if r["format"] == "group128" and r["bits"] == 4)
    assert pt4["frac_exact_zero"] > g128_4["frac_exact_zero"], (
        "expected per-tensor INT4 to force more weights to exact zero than group-128 INT4 "
        f"(got {pt4['frac_exact_zero']} vs {g128_4['frac_exact_zero']})"
    )
    print(
        f"[PASS] per-tensor INT4 collapse reproduced: "
        f"{pt4['frac_exact_zero']:.1%} exact-zero vs group128's {g128_4['frac_exact_zero']:.1%}"
    )


if __name__ == "__main__":
    main()
