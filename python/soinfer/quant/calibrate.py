"""Calibration strategies for choosing a quantization scale.

Each scale_fn has the signature

    scale_fn(grouped, amax, qmax) -> scale

where `grouped` holds the raw values being quantized together (shape
[..., group_size] -- the whole tensor flattened for per-tensor, one row for
per-channel, one group for group/block32/mx_e8m0) and `amax` is
`grouped.abs().amax(dim=-1)` (i.e. `amax.shape == grouped.shape[:-1]`,
computed by the caller since it's needed either way). Plug one in via
`formats.quantize(w, config, scale_fn=...)`; `min_max` is the default when
scale_fn is omitted.
"""
from __future__ import annotations

import torch

_EPS = torch.finfo(torch.float32).tiny


def min_max(grouped: torch.Tensor, amax: torch.Tensor, qmax: int) -> torch.Tensor:
    """scale = max(|w|) / qmax. The simplest, unbiased choice -- and the most
    sensitive to a single outlier setting the scale for an entire
    group/tensor (see formats.fraction_exact_zero for how badly that bites
    at per-tensor granularity)."""
    return (amax / qmax).clamp(min=_EPS)


def percentile(grouped: torch.Tensor, amax: torch.Tensor, qmax: int, pct: float = 99.9) -> torch.Tensor:
    """Clip the max magnitude to the pct-th percentile of |w| before scaling.
    Trades exact representation of the most extreme outliers (they now
    saturate at qmax instead of defining the scale) for a smaller scale --
    and thus finer rounding -- on the bulk of the distribution."""
    clipped_amax = torch.quantile(grouped.abs().float(), pct / 100.0, dim=-1)
    clipped_amax = torch.minimum(clipped_amax, amax)  # never let clipping *raise* the scale
    return (clipped_amax / qmax).clamp(min=_EPS)


def mse_optimal(
    grouped: torch.Tensor, amax: torch.Tensor, qmax: int, num_candidates: int = 32, min_frac: float = 0.5
) -> torch.Tensor:
    """Grid-search candidate scales in [min_frac, 1.0] x (amax/qmax) and keep
    whichever minimizes reconstruction MSE per group. The min-max scale
    (frac=1.0) is always one of the candidates but is rarely MSE-optimal: a
    smaller scale clips more outliers in exchange for finer rounding on
    everything else, and for most real weight distributions that trade is
    worth it below some point."""
    minmax_scale = (amax / qmax).clamp(min=_EPS)
    best_scale = minmax_scale.clone()
    best_mse = torch.full_like(minmax_scale, float("inf"))
    for frac in torch.linspace(min_frac, 1.0, num_candidates):
        cand = minmax_scale * frac.item()
        q = torch.clamp(torch.round(grouped / cand.unsqueeze(-1)), -qmax, qmax)
        mse = (q * cand.unsqueeze(-1) - grouped).pow(2).mean(dim=-1)
        improved = mse < best_mse
        best_scale = torch.where(improved, cand, best_scale)
        best_mse = torch.where(improved, mse, best_mse)
    return best_scale


def awq_scale(w: torch.Tensor, act_abs_mean: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """AWQ-style activation-aware per-input-channel scaling (Lin et al.,
    2023). Quantization error on weight channel k matters in proportion to
    how large the activations multiplying it are, so scale that channel
    *up* by s_k = mean(|x_k|)**alpha before quantizing, and divide
    activation channel k by the same s_k at inference time --
    (W * s) @ (x / s) == W @ x exactly, so this is a pure reparameterization
    that shifts quantization error away from the channels that matter most.

    w: [..., N, K]. act_abs_mean: [K], average |activation| per input
    channel from a calibration pass. Returns s: [K] (apply with
    apply_awq_scale(w, s), and divide the matching activations by s).
    """
    if act_abs_mean.shape[-1] != w.shape[-1]:
        raise ValueError(f"act_abs_mean has K={act_abs_mean.shape[-1]}, w has K={w.shape[-1]}")
    s = act_abs_mean.float().clamp(min=_EPS).pow(alpha)
    return s / s.mean()  # normalize so overall weight magnitude doesn't drift


def apply_awq_scale(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """w: [..., N, K], s: [K] from awq_scale(). Scale weights up before
    quantizing; divide the matching activations by the same s beforehand."""
    return w * s
