"""M6 task 4's decode loop: greedy generation over a model whose decoder-
layer weights are streamed from a PinnedWeightStore just-in-time, one
matrix at a time, instead of sitting resident on the GPU.

`WeightPipeline` double-buffers this: while the default (compute) stream
is running the GEMV for weight i, a second CUDA stream is already copying
weight i+1 into the other buffer, continuously across the ENTIRE
generation (not reset per layer or per token -- the sequence of weight
identities is static and known in advance regardless of which tokens end
up being generated, so there's nothing to gain from ever letting the
pipeline drain). This is PROJECT_SPEC.md M6 task 2's "prefetch layer i+1
while computing layer i," generalized to weight-level granularity.

Q/K/V are still three separate GEMVs rather than M5's fused
`fused_qkv_projection` -- re-fusing them against per-tensor-quantized
weights (each needs its own scale) is a separate follow-up, not required
for the overlap this file adds.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

import soinfer.ops as ops
from soinfer.offload.load_hf_checkpoint import LINEAR_SUFFIXES, LoadedMatrix
from soinfer.offload.stream_manager import StreamManager
from soinfer.offload.weight_store import PinnedWeightStore

GROUP_SIZE = 128


@dataclass
class StreamingModel:
    """Everything the decode loop needs, bundled: the pinned arena and its
    per-matrix handles (from load_hf_checkpoint.stream_load_layers), the
    small always-resident GPU tensors (embeddings/lm_head/norms, none of
    which benefit from per-layer streaming -- see load_hf_checkpoint.py's
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
        self.stream_manager = StreamManager(num_buffers=2)


class WeightPipeline:
    """Walks the flat, cyclic sequence of per-token weight names
    (layer0.q, layer0.k, ..., layer0.down, layer1.q, ..., wrapping back to
    layer0.q for the next token) one step ahead of compute. Two generic
    byte buffers, each sized to the single largest weight in the model --
    reused across every role/layer rather than one pair per role, since
    only one weight is ever "in flight" at a time regardless of its shape.
    """

    def __init__(self, model: StreamingModel):
        self.model = model
        self.sm = model.stream_manager
        max_bytes = max(m.n * m.handle.row_nbytes for m in model.matrices.values())
        self.bufs = [torch.empty(max_bytes, dtype=torch.uint8, device="cuda") for _ in range(2)]
        per_token_sequence = [
            f"model.layers.{i}.{suffix}" for i in range(model.num_layers) for suffix in LINEAR_SUFFIXES
        ]
        self._names = itertools.cycle(per_token_sequence)
        self.cur_buf = 0
        self.cur_name = next(self._names)
        self._prefetch_into(self.cur_buf, self.cur_name)  # prime: get the first weight in flight

    def _prefetch_into(self, buf_idx: int, name: str) -> None:
        m = self.model.matrices[name]
        nbytes = m.n * m.handle.row_nbytes
        view = self.bufs[buf_idx][:nbytes].view(m.n, m.handle.row_nbytes)
        self.sm.prefetch(buf_idx, view, self.model.store.matrix_view(m.handle))

    def next_gemv(self, x_in: torch.Tensor) -> torch.Tensor:
        """Waits for the currently-in-flight weight (issued either by
        __init__ or the previous call), runs its GEMV, and immediately
        kicks off the NEXT weight's prefetch into the other buffer before
        returning -- so that transfer runs concurrently with whatever
        compute the caller does with this call's result."""
        name = self.cur_name
        m = self.model.matrices[name]
        self.sm.wait(self.cur_buf)
        nbytes = m.n * m.handle.row_nbytes
        view = self.bufs[self.cur_buf][:nbytes].view(m.n, m.handle.row_nbytes)
        result = ops.gemv_w4a16_group_lop3(view, m.scale, x_in, m.packed_k, GROUP_SIZE)

        next_buf = 1 - self.cur_buf
        next_name = next(self._names)
        self._prefetch_into(next_buf, next_name)
        self.cur_buf, self.cur_name = next_buf, next_name
        return result


def run_decoder_layer_streaming(
    pipeline: WeightPipeline, model: StreamingModel, layer_idx: int, x_flat: torch.Tensor, pos: int,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
) -> torch.Tensor:
    """One decoder layer's forward for a single token at position `pos`,
    pulling its 7 linear weights from `pipeline` (already in flight,
    overlapped with the previous weight's compute). Mutates k_cache/
    v_cache in place (append at `pos`). Same block structure as
    tests/test_layer_parity.py and test_end_to_end.py's _run_one_layer,
    but with quantized-streamed weights instead of resident FP16 ones."""
    NQ, NKV, HD = model.num_attention_heads, model.num_key_value_heads, model.head_dim
    eps, theta = model.rms_norm_eps, model.rope_theta
    norms = model.layer_norms[layer_idx]

    residual = x_flat
    h = ops.rmsnorm(x_flat.unsqueeze(0), norms["input_layernorm"], eps).squeeze(0).contiguous()

    q = pipeline.next_gemv(h).view(NQ, HD).contiguous()
    k = pipeline.next_gemv(h).view(NKV, HD).contiguous()
    v = pipeline.next_gemv(h).view(NKV, HD).contiguous()

    q = ops.qk_norm(q, norms["q_norm"], eps)
    k = ops.qk_norm(k, norms["k_norm"], eps)
    cos_vals, sin_vals = ops.precompute_rope_cos_sin(HD, theta, pos, device="cuda")
    q = ops.apply_rope(q, cos_vals, sin_vals)
    k = ops.apply_rope(k, cos_vals, sin_vals)

    ops.kv_cache_append(k_cache, v_cache, k, v, pos)
    attn_out = ops.decode_attention(q, k_cache, v_cache, pos + 1).reshape(NQ * HD).contiguous()

    o = pipeline.next_gemv(attn_out)
    h2 = residual + o

    residual2 = h2
    h3 = ops.rmsnorm(h2.unsqueeze(0), norms["post_attention_layernorm"], eps).squeeze(0).contiguous()

    gate = pipeline.next_gemv(h3)
    up = pipeline.next_gemv(h3)
    mlp_h = (torch.nn.functional.silu(gate.float()) * up.float()).half()
    down = pipeline.next_gemv(mlp_h)

    return residual2 + down


def generate_streaming(model: StreamingModel, prompt_ids: list[int], n_new: int, eos_token_id: int | None = None, max_seq_len: int = 512) -> list[int]:
    """Greedy decode: prefills `prompt_ids` one token at a time (this
    project targets decode, not batched prefill -- see PROJECT_SPEC.md sec
    2 -- so the prompt is just decode steps with no output taken until
    it's consumed), then generates up to `n_new` more tokens, stopping
    early on `eos_token_id` if given. One WeightPipeline runs continuously
    across the whole call -- prompt tokens and generated tokens alike --
    since the weight-load sequence never depends on which tokens are
    actually produced."""
    NKV, HD = model.num_key_value_heads, model.head_dim
    k_caches = [torch.zeros(NKV, max_seq_len, HD, device="cuda", dtype=torch.float16) for _ in range(model.num_layers)]
    v_caches = [torch.zeros(NKV, max_seq_len, HD, device="cuda", dtype=torch.float16) for _ in range(model.num_layers)]
    pipeline = WeightPipeline(model)

    all_tokens = list(prompt_ids)
    x = None
    for pos, tok in enumerate(prompt_ids):
        x = model.embed_tokens[tok].contiguous()
        for layer_idx in range(model.num_layers):
            x = run_decoder_layer_streaming(pipeline, model, layer_idx, x, pos, k_caches[layer_idx], v_caches[layer_idx])

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
            x = run_decoder_layer_streaming(pipeline, model, layer_idx, x, pos, k_caches[layer_idx], v_caches[layer_idx])
        pos += 1

    return all_tokens
