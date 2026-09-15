"""Apply fake (quantize-then-immediately-dequantize) weight quantization to
every nn.Linear layer of a loaded HF model, in place. This is "fake quant":
the model still computes in float, but every weight has been rounded to
exactly what a real quantized kernel would produce, so a perplexity eval
measures the format's accuracy cost without needing an actual INT4/INT8
GEMM kernel (that's M4+).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from . import formats


def fake_quantize_linear_layers(
    model: nn.Module,
    config: formats.QuantConfig,
    scale_fn=None,
    skip_names: Sequence[str] = ("lm_head", "embed_tokens"),
) -> int:
    """Mutates model in place: every nn.Linear whose qualified name doesn't
    contain any of skip_names gets its .weight replaced by
    dequantize(quantize(weight, config, scale_fn)). Embeddings/lm_head are
    skipped by default -- this project's quantization story is about the
    attention/MLP projection weights, not the vocabulary embedding, and
    conflating the two would muddy the accuracy comparison.

    Returns the number of layers touched.
    """
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(skip in name for skip in skip_names):
            continue
        with torch.no_grad():
            w = module.weight.data
            qt = formats.quantize(w.float(), config, scale_fn=scale_fn)
            module.weight.data = formats.dequantize(qt).to(w.dtype).to(w.device)
        count += 1
    return count
