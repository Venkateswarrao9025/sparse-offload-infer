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
import torch.nn.functional as F

import soinfer.ops as ops
from soinfer.offload.load_hf_checkpoint import DOWN_PROJ_T_SUFFIX, LINEAR_SUFFIXES, LoadedMatrix
from soinfer.offload.stream_manager import StreamManager
from soinfer.offload.weight_store import PinnedWeightStore

GROUP_SIZE = 128

# M7: the 5 weights that stay fully dense-streamed even in the DIP decode
# path (gate_proj must run dense -- selection needs |gate_out| for every
# channel -- and attention has no channel-selection concept at all). up_proj
# and down_proj_T are deliberately excluded: DIP path 3.a gathers only their
# k selected rows directly from the pinned arena instead of streaming the
# whole matrix through a WeightPipeline.
DIP_DENSE_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
)


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

    def __init__(self, model: StreamingModel, suffixes: tuple[str, ...] = LINEAR_SUFFIXES):
        """suffixes (M7): the per-layer weight names this pipeline cycles
        through. Defaults to all 7 linear weights (the dense M6 path); the
        DIP decode path passes DIP_DENSE_SUFFIXES (5 weights -- up_proj and
        down_proj_T are excluded, fetched instead by an ad hoc row gather
        sized to the selected k, not streamed whole)."""
        self.model = model
        self.sm = model.stream_manager
        per_token_sequence = [f"model.layers.{i}.{suffix}" for i in range(model.num_layers) for suffix in suffixes]
        max_bytes = max(model.matrices[name].n * model.matrices[name].handle.row_nbytes for name in per_token_sequence)
        self.bufs = [torch.empty(max_bytes, dtype=torch.uint8, device="cuda") for _ in range(2)]
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


@dataclass
class DipBuffers:
    """M7: reusable scratch buffers for run_decoder_layer_dip's up_proj/
    down_proj_T row gathers -- allocated once per model (sized to the
    largest k the caller will ever request across a whole run) and reused
    across every layer/token, matching WeightPipeline's "allocate once,
    reuse" pattern rather than paying pinned-memory allocation cost per
    call."""

    staging_up: torch.Tensor  # pinned CPU uint8 [max_k * up_row_nbytes]
    staging_down: torch.Tensor  # pinned CPU uint8 [max_k * downT_row_nbytes]
    gpu_up: torch.Tensor  # CUDA uint8 [max_k * up_row_nbytes]
    gpu_down: torch.Tensor  # CUDA uint8 [max_k * downT_row_nbytes]
    up_row_nbytes: int
    down_row_nbytes: int

    @staticmethod
    def make(model: StreamingModel, max_k: int) -> "DipBuffers":
        up_row_nbytes = model.matrices["model.layers.0.mlp.up_proj.weight"].handle.row_nbytes
        down_row_nbytes = model.matrices[f"model.layers.0.{DOWN_PROJ_T_SUFFIX}"].handle.row_nbytes
        return DipBuffers(
            staging_up=torch.empty(max_k * up_row_nbytes, dtype=torch.uint8).pin_memory(),
            staging_down=torch.empty(max_k * down_row_nbytes, dtype=torch.uint8).pin_memory(),
            gpu_up=torch.empty(max_k * up_row_nbytes, dtype=torch.uint8, device="cuda"),
            gpu_down=torch.empty(max_k * down_row_nbytes, dtype=torch.uint8, device="cuda"),
            up_row_nbytes=up_row_nbytes,
            down_row_nbytes=down_row_nbytes,
        )


def run_decoder_layer_dip(
    pipeline: WeightPipeline, model: StreamingModel, layer_idx: int, x_flat: torch.Tensor, pos: int,
    k_cache: torch.Tensor, v_cache: torch.Tensor, dip_k: int, bufs: DipBuffers,
) -> torch.Tensor:
    """M7: same attention block as run_decoder_layer_streaming (q/k/v/o
    still fully dense-streamed through `pipeline`), but the MLP only
    touches `dip_k` of the intermediate_size channels. gate_proj also
    stays dense (through `pipeline`, same as q/k/v/o) -- selecting the top
    dip_k channels needs |gate_out| for every channel first (PROJECT_SPEC.md
    M7's opening paragraph). up_proj and down_proj_T are NOT streamed
    through `pipeline` at all: their dip_k selected rows are gathered
    directly from the pinned arena (gather_rows_staged, M7 task 2) into
    `bufs`, one synchronous round-trip per layer per token (no cross-token
    lookahead for these two -- which rows are needed is only known after
    this token's gate_proj GEMV runs, unlike q/k/v/o/gate's fixed,
    known-in-advance sequence that `pipeline` prefetches ahead of time;
    see LEARNING_NOTES.md's M7 task 4 entry for why that's an honest,
    documented limitation rather than an oversight). Consumed by M7 task
    3's two sparse GEMVs: up reuses gemv_w4a16_group_lop3 (M4) with
    N=dip_k; down uses the new gemv_w4a16_sparse_accumulate."""
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

    gate = pipeline.next_gemv(h3)  # dense: |gate_out| drives selection below
    abs_g = gate.float().abs().contiguous()
    idx = ops.topk_threshold_select(abs_g, dip_k)  # [dip_k] int32, CUDA
    idx_long = idx.long()
    idx_cpu = idx_long.cpu()  # gather_rows_staged's host-side memcpy needs host-resident indices

    up_lm = model.matrices[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
    up_host = model.store.matrix_view(up_lm.handle)
    up_nbytes = dip_k * bufs.up_row_nbytes
    up_staging = bufs.staging_up[:up_nbytes].view(dip_k, bufs.up_row_nbytes)
    up_gpu = bufs.gpu_up[:up_nbytes].view(dip_k, bufs.up_row_nbytes)
    ops.gather_rows_staged(up_host, idx_cpu, up_staging, up_gpu)

    downT_lm = model.matrices[f"model.layers.{layer_idx}.{DOWN_PROJ_T_SUFFIX}"]
    downT_host = model.store.matrix_view(downT_lm.handle)
    down_nbytes = dip_k * bufs.down_row_nbytes
    down_staging = bufs.staging_down[:down_nbytes].view(dip_k, bufs.down_row_nbytes)
    down_gpu = bufs.gpu_down[:down_nbytes].view(dip_k, bufs.down_row_nbytes)
    ops.gather_rows_staged(downT_host, idx_cpu, down_staging, down_gpu)
    torch.cuda.synchronize()  # both gathers must land before the sparse GEMVs below read them

    up_scale_sel = up_lm.scale.index_select(0, idx_long)
    u_selected = ops.gemv_w4a16_group_lop3(up_gpu, up_scale_sel, h3, up_lm.packed_k, GROUP_SIZE)

    g_selected = gate.index_select(0, idx_long)
    h_selected = (F.silu(g_selected.float()) * u_selected.float()).half()

    downT_scale_sel = downT_lm.scale.index_select(0, idx_long)
    down_out = ops.gemv_w4a16_sparse_accumulate(down_gpu, downT_scale_sel, h_selected, model.hidden_size, GROUP_SIZE)

    return residual2 + down_out


def dip_bytes_per_token(model: StreamingModel, dip_k: int, dense: bool = False) -> int:
    """M7 acceptance ("measurable reduction in bytes transferred per token
    -- instrument it directly, count bytes, don't infer from timing"):
    total per-token weight bytes, computed directly from the arena's own
    row_nbytes/row-count bookkeeping. dense=True reports the M6 baseline
    (every one of the 7 linear weights streamed in full, matching
    WeightPipeline(model, LINEAR_SUFFIXES)); dense=False reports the DIP
    path (up_proj and down_proj_T reduced to dip_k rows each; everything
    else -- q/k/v/o/gate -- unchanged, since only the MLP's channel-
    selected weights are ever pruned)."""
    total = 0
    for layer in range(model.num_layers):
        for suffix in DIP_DENSE_SUFFIXES:
            m = model.matrices[f"model.layers.{layer}.{suffix}"]
            total += m.n * m.handle.row_nbytes
        up = model.matrices[f"model.layers.{layer}.mlp.up_proj.weight"]
        downT = model.matrices[f"model.layers.{layer}.{DOWN_PROJ_T_SUFFIX}"]
        total += (up.n if dense else dip_k) * up.handle.row_nbytes
        total += (downT.n if dense else dip_k) * downT.handle.row_nbytes
    return total


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


def generate_dip(model: StreamingModel, prompt_ids: list[int], n_new: int, dip_k: int,
                  eos_token_id: int | None = None, max_seq_len: int = 512) -> list[int]:
    """M7: same greedy-decode structure as generate_streaming, but every
    layer's MLP runs run_decoder_layer_dip instead of
    run_decoder_layer_streaming -- q/k/v/o/gate still stream densely
    through a WeightPipeline (built with DIP_DENSE_SUFFIXES, 5 weights
    instead of 7: up_proj/down_proj_T are excluded since DIP fetches only
    their dip_k selected rows per token, not the whole matrix)."""
    NKV, HD = model.num_key_value_heads, model.head_dim
    k_caches = [torch.zeros(NKV, max_seq_len, HD, device="cuda", dtype=torch.float16) for _ in range(model.num_layers)]
    v_caches = [torch.zeros(NKV, max_seq_len, HD, device="cuda", dtype=torch.float16) for _ in range(model.num_layers)]
    pipeline = WeightPipeline(model, suffixes=DIP_DENSE_SUFFIXES)
    bufs = DipBuffers.make(model, max_k=dip_k)

    all_tokens = list(prompt_ids)
    x = None
    for pos, tok in enumerate(prompt_ids):
        x = model.embed_tokens[tok].contiguous()
        for layer_idx in range(model.num_layers):
            x = run_decoder_layer_dip(pipeline, model, layer_idx, x, pos, k_caches[layer_idx], v_caches[layer_idx], dip_k, bufs)

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
            x = run_decoder_layer_dip(pipeline, model, layer_idx, x, pos, k_caches[layer_idx], v_caches[layer_idx], dip_k, bufs)
        pos += 1

    return all_tokens


def teacher_forced_nll(model: StreamingModel, layer_fn, token_ids: list[int], max_seq_len: int = 512) -> float:
    """M7 task 4's perplexity building block: mean negative log-likelihood
    of each REAL next token in `token_ids`, teacher-forced (the ground-
    truth prefix is fed at every step, never the model's own prediction --
    standard perplexity methodology, distinct from generate_streaming's/
    generate_dip's greedy-sampling loops). perplexity = exp(the returned
    value).

    `layer_fn(model, layer_idx, x, pos, k_cache, v_cache) -> x_out` lets
    this one eval loop score either decode path: pass
    `lambda m, li, x, p, kc, vc: run_decoder_layer_streaming(pipeline, m, li, x, p, kc, vc)`
    for the dense (M6) path, or the equivalent closure over
    run_decoder_layer_dip for M7's DIP path -- whichever pipeline/DipBuffers
    state a specific call needs is captured by the caller's closure, not by
    this function."""
    NKV, HD = model.num_key_value_heads, model.head_dim
    k_caches = [torch.zeros(NKV, max_seq_len, HD, device="cuda", dtype=torch.float16) for _ in range(model.num_layers)]
    v_caches = [torch.zeros(NKV, max_seq_len, HD, device="cuda", dtype=torch.float16) for _ in range(model.num_layers)]

    total_nll = 0.0
    count = 0
    for pos, tok in enumerate(token_ids[:-1]):
        x = model.embed_tokens[tok].contiguous()
        for layer_idx in range(model.num_layers):
            x = layer_fn(model, layer_idx, x, pos, k_caches[layer_idx], v_caches[layer_idx])
        xn = ops.rmsnorm(x.unsqueeze(0), model.final_norm, model.rms_norm_eps).squeeze(0).contiguous()
        logits = ops.gemv_fp16_v3(model.lm_head, xn)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        total_nll += -log_probs[token_ids[pos + 1]].item()
        count += 1

    return total_nll / count
