"""M6 task 4's decode loop: greedy generation over a model whose decoder-
layer weights are streamed from a PinnedWeightStore just-in-time, one
matrix at a time, instead of sitting resident on the GPU.

This is deliberately the "correctness first" version (PROJECT_SPEC.md's
own framing, echoed throughout this project's M2/M4/M5 progression):
`StreamManager` is used with a single buffer per weight role (prefetch
immediately followed by wait, no cross-layer overlap yet) and Q/K/V are
three separate GEMVs rather than M5's fused_qkv_projection. Both are real,
measured follow-ups (the M6 roofline already shows the overlap
*opportunity*; realizing it in this actual loop, and re-fusing QKV against
per-tensor-quantized weights, are the natural next optimization passes),
not correctness gaps.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

import soinfer.ops as ops
from soinfer.offload.load_hf_checkpoint import LINEAR_SUFFIXES, LoadedMatrix
from soinfer.offload.stream_manager import StreamManager
from soinfer.offload.weight_store import PinnedWeightStore

GROUP_SIZE = 128


@dataclass
class StreamingModel:
    """Everything generate_streaming needs, bundled: the pinned arena and
    its per-matrix handles (from load_hf_checkpoint.stream_load_layers),
    the small always-resident GPU tensors (embeddings/lm_head/norms, none
    of which benefit from per-layer streaming -- see load_hf_checkpoint.py's
    module docstring for why), and the streaming machinery itself."""

    store: PinnedWeightStore
    matrices: dict[str, LoadedMatrix]  # "model.layers.{i}.{suffix}" -> LoadedMatrix
    num_layers: int
    embed_tokens: torch.Tensor  # [vocab, H] half, GPU-resident
    lm_head: torch.Tensor  # [vocab, H] half, GPU-resident
    final_norm: torch.Tensor  # [H] half, GPU-resident
    layer_norms: list[dict[str, torch.Tensor]]  # per layer: input_layernorm/post_attention_layernorm/q_norm/k_norm
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float

    def __post_init__(self) -> None:
        self.stream_manager = StreamManager(num_buffers=1)
        self.gpu_bufs: dict[str, torch.Tensor] = {}
        for suffix in LINEAR_SUFFIXES:
            m = self.matrices[f"model.layers.0.{suffix}"]
            self.gpu_bufs[suffix] = torch.empty(m.n, m.handle.row_nbytes, dtype=torch.uint8, device="cuda")


def _load_and_gemv(model: StreamingModel, layer_idx: int, suffix: str, x_in: torch.Tensor) -> torch.Tensor:
    m = model.matrices[f"model.layers.{layer_idx}.{suffix}"]
    buf = model.gpu_bufs[suffix]
    model.stream_manager.prefetch(0, buf, model.store.matrix_view(m.handle))
    model.stream_manager.wait(0)
    return ops.gemv_w4a16_group_lop3(buf, m.scale, x_in, m.packed_k, GROUP_SIZE)


def run_decoder_layer_streaming(
    model: StreamingModel, layer_idx: int, x_flat: torch.Tensor, pos: int, k_cache: torch.Tensor, v_cache: torch.Tensor
) -> torch.Tensor:
    """One decoder layer's forward for a single token at position `pos`,
    streaming its 7 linear weights from the pinned arena just-in-time.
    Mutates k_cache/v_cache in place (append at `pos`). Same block
    structure as tests/test_layer_parity.py and test_end_to_end.py's
    _run_one_layer, but with quantized-streamed weights instead of
    resident FP16 ones."""
    NQ, NKV, HD = model.num_attention_heads, model.num_key_value_heads, model.head_dim
    eps, theta = model.rms_norm_eps, model.rope_theta
    norms = model.layer_norms[layer_idx]

    residual = x_flat
    h = ops.rmsnorm(x_flat.unsqueeze(0), norms["input_layernorm"], eps).squeeze(0).contiguous()

    q = _load_and_gemv(model, layer_idx, "self_attn.q_proj.weight", h).view(NQ, HD).contiguous()
    k = _load_and_gemv(model, layer_idx, "self_attn.k_proj.weight", h).view(NKV, HD).contiguous()
    v = _load_and_gemv(model, layer_idx, "self_attn.v_proj.weight", h).view(NKV, HD).contiguous()

    q = ops.qk_norm(q, norms["q_norm"], eps)
    k = ops.qk_norm(k, norms["k_norm"], eps)
    cos_vals, sin_vals = ops.precompute_rope_cos_sin(HD, theta, pos, device="cuda")
    q = ops.apply_rope(q, cos_vals, sin_vals)
    k = ops.apply_rope(k, cos_vals, sin_vals)

    ops.kv_cache_append(k_cache, v_cache, k, v, pos)
    attn_out = ops.decode_attention(q, k_cache, v_cache, pos + 1).reshape(NQ * HD).contiguous()

    o = _load_and_gemv(model, layer_idx, "self_attn.o_proj.weight", attn_out)
    h2 = residual + o

    residual2 = h2
    h3 = ops.rmsnorm(h2.unsqueeze(0), norms["post_attention_layernorm"], eps).squeeze(0).contiguous()

    gate = _load_and_gemv(model, layer_idx, "mlp.gate_proj.weight", h3)
    up = _load_and_gemv(model, layer_idx, "mlp.up_proj.weight", h3)
    mlp_h = (torch.nn.functional.silu(gate.float()) * up.float()).half()
    down = _load_and_gemv(model, layer_idx, "mlp.down_proj.weight", mlp_h)

    return residual2 + down


def generate_streaming(model: StreamingModel, prompt_ids: list[int], n_new: int, eos_token_id: int | None = None, max_seq_len: int = 512) -> list[int]:
    """Greedy decode: prefills `prompt_ids` one token at a time (this
    project targets decode, not batched prefill -- see PROJECT_SPEC.md sec
    2 -- so the prompt is just decode steps with no output taken until
    it's consumed), then generates up to `n_new` more tokens, stopping
    early on `eos_token_id` if given."""
    NKV, HD = model.num_key_value_heads, model.head_dim
    k_caches = [torch.zeros(NKV, max_seq_len, HD, device="cuda", dtype=torch.float16) for _ in range(model.num_layers)]
    v_caches = [torch.zeros(NKV, max_seq_len, HD, device="cuda", dtype=torch.float16) for _ in range(model.num_layers)]

    all_tokens = list(prompt_ids)
    x = None
    for pos, tok in enumerate(prompt_ids):
        x = model.embed_tokens[tok].contiguous()
        for layer_idx in range(model.num_layers):
            x = run_decoder_layer_streaming(model, layer_idx, x, pos, k_caches[layer_idx], v_caches[layer_idx])

    pos = len(prompt_ids)
    for _ in range(n_new):
        xn = ops.rmsnorm(x.unsqueeze(0), model.final_norm, model.rms_norm_eps).squeeze(0).contiguous()
        logits = ops.gemv_fp16_v3(model.lm_head, xn)
        next_tok = int(torch.argmax(logits).item())
        all_tokens.append(next_tok)
        if eos_token_id is not None and next_tok == eos_token_id:
            break
        x = model.embed_tokens[next_tok].contiguous()
        for layer_idx in range(model.num_layers):
            x = run_decoder_layer_streaming(model, layer_idx, x, pos, k_caches[layer_idx], v_caches[layer_idx])
        pos += 1

    return all_tokens
