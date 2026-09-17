"""M6 task 4 support: stream a HF safetensors checkpoint's per-layer linear
weights directly into a PinnedWeightStore, quantizing ONE TENSOR AT A TIME.

This deliberately never materializes the full checkpoint in host RAM the
way `transformers.AutoModelForCausalLM.from_pretrained` does -- for a 14B+
model that's 28+ GB in bf16, which doesn't fit in this project's Colab
free-tier ~10GB RAM budget. Peak extra RAM here is roughly one weight
tensor (at most a few hundred MB for Qwen3-14B's largest per-layer matrix)
plus the pinned arena itself, which is sized to the much smaller quantized
total (~6-7 GB for Qwen3-14B at INT4-group128) -- that's what actually
makes a headline-sized model loadable at all on this hardware.

Embedding, LM head, and norm weights are NOT handled here: embed_tokens is
a lookup (no GEMV to quantize), the norms are tiny, and lm_head, while
GEMV-shaped, is used identically every step (no benefit from per-layer
streaming) -- all three are simple enough to load directly to GPU as FP16
in the calling script, not through this streaming/quantizing path.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from safetensors import safe_open

from ..quant import formats, pack
from .weight_store import PinnedWeightStore, WeightHandle

# Qwen3 decoder layer's linear weights (see transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer).
LINEAR_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


@dataclass
class LoadedMatrix:
    handle: WeightHandle
    scale: torch.Tensor  # [N, num_groups] fp32, already on GPU
    packed_k: int
    n: int
    k: int


def _quantize_and_pack(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    config = formats.QuantConfig(bits=4, granularity="group", group_size=group_size)
    qt = formats.quantize(w.float(), config)
    packed, packed_k = pack.pack_int4(qt.qweight)
    return packed, qt.scale.contiguous(), packed_k


def _packed_int4_bytes(n: int, k: int) -> int:
    return n * (-(-k // 8) * 4)  # n * ceil(k/8)*4


def plan_arena_bytes(
    model_dir: str, num_layers: int, suffixes: tuple[str, ...] = LINEAR_SUFFIXES, transpose: bool = False
) -> int:
    """First pass: reads each linear weight's shape from its safetensors
    header only (no tensor data loaded -- `get_slice(name).get_shape()` is
    a metadata read) to compute the exact pinned arena size needed.
    PinnedWeightStore needs this upfront (one allocation, not a growable
    one -- see weight_store.py). `transpose` swaps (n, k) for a planned
    down_proj_T registration (M7's include_down_proj_t)."""
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    shard_cache: dict[str, object] = {}
    total_bytes = 0
    for layer in range(num_layers):
        for suffix in suffixes:
            name = f"model.layers.{layer}.{suffix}"
            shard_name = weight_map[name]
            if shard_name not in shard_cache:
                shard_cache[shard_name] = safe_open(os.path.join(model_dir, shard_name), framework="pt")
            n, k = shard_cache[shard_name].get_slice(name).get_shape()
            if transpose:
                n, k = k, n
            total_bytes += _packed_int4_bytes(n, k)
    return total_bytes


DOWN_PROJ_T_SUFFIX = "mlp.down_proj_T.weight"  # synthetic name, not in any HF checkpoint -- see include_down_proj_t


def stream_load_layers(
    model_dir: str, num_layers: int, group_size: int = 128, log_every: int = 8, include_down_proj_t: bool = False
) -> tuple[PinnedWeightStore, dict[str, LoadedMatrix]]:
    """Quantizes and registers every decoder layer's 7 linear weights, one
    tensor at a time, straight from the safetensors shards. Returns the
    populated store and a name -> LoadedMatrix map (name format:
    "model.layers.{i}.{suffix}", matching LINEAR_SUFFIXES).

    include_down_proj_t (M7): ALSO registers down_proj TRANSPOSED, under
    the synthetic name "model.layers.{i}.mlp.down_proj_T.weight" --
    down_proj's HF layout is [hidden_size, intermediate_size]
    (out-feature-major), so selecting a subset of intermediate channels
    means selecting COLUMNS, which aren't contiguous and can't be row-
    gathered. Storing it transposed too ([intermediate_size, hidden_size])
    makes it row-indexed by intermediate channel, same as up_proj -- see
    csrc/kernels/gemv_sparse_accumulate.cuh's module docstring for the
    kernel that consumes this layout. Doubles down_proj's arena footprint
    (both orientations coexist) -- acceptable for the M7 DIP experiment,
    not something a real deployment would do (it would pick ONE mode)."""
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    total_bytes = plan_arena_bytes(model_dir, num_layers)
    if include_down_proj_t:
        total_bytes += plan_arena_bytes(model_dir, num_layers, suffixes=("mlp.down_proj.weight",), transpose=True)
    store = PinnedWeightStore(total_bytes=total_bytes)
    matrices: dict[str, LoadedMatrix] = {}
    shard_cache: dict[str, object] = {}

    def _shard(shard_name: str):
        if shard_name not in shard_cache:
            shard_cache[shard_name] = safe_open(os.path.join(model_dir, shard_name), framework="pt")
        return shard_cache[shard_name]

    for layer in range(num_layers):
        for suffix in LINEAR_SUFFIXES:
            name = f"model.layers.{layer}.{suffix}"
            w = _shard(weight_map[name]).get_tensor(name)  # materializes just this one tensor
            n, k = w.shape
            packed, scale, packed_k = _quantize_and_pack(w, group_size)
            handle = store.register(name, packed)
            matrices[name] = LoadedMatrix(handle=handle, scale=scale.cuda(), packed_k=packed_k, n=n, k=k)

            if include_down_proj_t and suffix == "mlp.down_proj.weight":
                wT = w.t().contiguous()  # [intermediate_size, hidden_size]
                nT, kT = wT.shape
                packedT, scaleT, packed_kT = _quantize_and_pack(wT, group_size)
                del wT
                nameT = f"model.layers.{layer}.{DOWN_PROJ_T_SUFFIX}"
                handleT = store.register(nameT, packedT)
                matrices[nameT] = LoadedMatrix(handle=handleT, scale=scaleT.cuda(), packed_k=packed_kT, n=nT, k=kT)
            del w  # free the bf16 tensor before the next one
        if (layer + 1) % log_every == 0 or layer == num_layers - 1:
            print(f"  loaded layer {layer + 1}/{num_layers} ({store.bytes_used / 1e9:.2f} GB in arena)")

    return store, matrices
