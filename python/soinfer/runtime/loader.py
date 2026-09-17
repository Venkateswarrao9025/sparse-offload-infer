"""M6 task 4: assembles a StreamingModel (generate.py) from a HF checkpoint
directory -- streams the decoder layers' linear weights into a
PinnedWeightStore via load_hf_checkpoint (never materializing the full
FP16/BF16 checkpoint in host RAM), and loads the small always-resident
pieces (embeddings, LM head, norms) directly to GPU.
"""
from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open

from soinfer.offload.load_hf_checkpoint import stream_load_layers
from .generate import StreamingModel


def _load_small_to_gpu(model_dir: str, weight_map: dict[str, str], name: str, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Loads one tensor straight to GPU, freeing the intermediate host
    copy immediately -- for embeddings/LM head/norms, which are small
    enough (relative to the ~10GB host RAM budget this project targets on
    Colab free tier) to just load directly rather than route through the
    streaming machinery meant for the much larger per-layer weights."""
    shard = weight_map[name]
    with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
        t = f.get_tensor(name)
    g = t.to(dtype=dtype, device="cuda")
    del t
    return g


def load_streaming_model(model_dir: str, group_size: int = 128) -> StreamingModel:
    """Reads config.json for architecture dims, streams every decoder
    layer's linear weights into a pinned, INT4-quantized arena, and loads
    embeddings/LM head/norms directly to GPU as FP16. Prints progress
    every few layers -- for a 14B-class model this takes minutes, not
    seconds, and silent multi-minute calls are a bad experience."""
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    num_layers = config["num_hidden_layers"]

    print(f"streaming {num_layers} decoder layers into a pinned INT4 arena...")
    store, matrices = stream_load_layers(model_dir, num_layers, group_size=group_size)

    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    print("loading embeddings/LM head/norms to GPU...")
    embed_tokens = _load_small_to_gpu(model_dir, weight_map, "model.embed_tokens.weight")
    lm_head = _load_small_to_gpu(model_dir, weight_map, "lm_head.weight")
    final_norm = _load_small_to_gpu(model_dir, weight_map, "model.norm.weight")
    layer_norms = [
        {
            "input_layernorm": _load_small_to_gpu(model_dir, weight_map, f"model.layers.{i}.input_layernorm.weight"),
            "post_attention_layernorm": _load_small_to_gpu(model_dir, weight_map, f"model.layers.{i}.post_attention_layernorm.weight"),
            "q_norm": _load_small_to_gpu(model_dir, weight_map, f"model.layers.{i}.self_attn.q_norm.weight"),
            "k_norm": _load_small_to_gpu(model_dir, weight_map, f"model.layers.{i}.self_attn.k_norm.weight"),
        }
        for i in range(num_layers)
    ]

    return StreamingModel(
        store=store,
        matrices=matrices,
        num_layers=num_layers,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        final_norm=final_norm,
        layer_norms=layer_norms,
        hidden_size=config["hidden_size"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        rms_norm_eps=config["rms_norm_eps"],
        rope_theta=float(config["rope_theta"]),
    )
