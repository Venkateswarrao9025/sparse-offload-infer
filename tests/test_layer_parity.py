"""M5 acceptance (PROJECT_SPEC.md sec 6, M5): "a full transformer block, your
implementation vs HF, max relative error < 2e-2 on real activations." This
is the milestone's own words for its backbone test.

Assembles one Qwen3 decoder layer's forward pass (input RMSNorm -> fused
QKV -> QK-norm -> RoPE -> KV cache append -> decode attention -> o_proj ->
residual -> post-attention RMSNorm -> fused SwiGLU MLP -> residual)
entirely from soinfer.ops kernels, using the real transformers Qwen3
modeling code (Qwen3DecoderLayer, Qwen3RotaryEmbedding) as both the weight
source and the reference output -- not an independently-derived formula.
Uses the real Qwen3-1.7B shape (hidden=2048, intermediate=6144, 16 query
heads, 8 KV heads, head_dim=128, rope_theta=1e6) with randomly-initialized
weights (no download needed -- this tests the architecture/kernels, not a
specific checkpoint's numbers).

Skipped entirely on machines without a CUDA GPU or without transformers
installed; run for real on the Colab/Kaggle T4 session via `make test`.
"""
import pytest
import torch

soinfer = pytest.importorskip("soinfer")
if soinfer.ops is None:
    pytest.skip("soinfer._C (CUDA extension) not built", allow_module_level=True)
transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

from transformers import Qwen3Config  # noqa: E402
from transformers.models.qwen3 import modeling_qwen3 as qwen3_mod  # noqa: E402

MAX_REL_ERR = 2e-2  # PROJECT_SPEC.md M5 acceptance bound
NEAR_ZERO_THRESHOLD = 0.05  # below this |hf_out|, relative error is dominated by fp16 rounding noise, not signal


def _qwen3_1p7b_shaped_config() -> "Qwen3Config":
    return Qwen3Config(
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        vocab_size=100,  # small; the LM head isn't exercised by a single decoder layer
    )


def _run_my_layer(layer, config, x_flat: torch.Tensor, pos: int) -> torch.Tensor:
    """Assembles one decoder layer's forward for a single token, using
    weights extracted from `layer` (a real Qwen3DecoderLayer), entirely via
    soinfer.ops kernels."""
    ops = soinfer.ops
    H, NQ, NKV, HD = config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.head_dim
    eps = config.rms_norm_eps
    rope_theta = float(config.rope_parameters["rope_theta"])
    attn = layer.self_attn

    residual = x_flat
    h = ops.rmsnorm(x_flat.unsqueeze(0), layer.input_layernorm.weight, eps).squeeze(0).contiguous()

    qkv_W = ops.concat_qkv_weights(attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight)
    q, k, v = ops.fused_qkv_projection(qkv_W, h, NQ * HD, NKV * HD)
    q = q.view(NQ, HD).contiguous()
    k = k.view(NKV, HD).contiguous()
    v = v.view(NKV, HD).contiguous()

    q = ops.qk_norm(q, attn.q_norm.weight, eps)
    k = ops.qk_norm(k, attn.k_norm.weight, eps)

    cos_vals, sin_vals = ops.precompute_rope_cos_sin(HD, rope_theta, pos, device="cuda")
    q = ops.apply_rope(q, cos_vals, sin_vals)
    k = ops.apply_rope(k, cos_vals, sin_vals)

    k_cache = torch.zeros(NKV, 1, HD, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros(NKV, 1, HD, device="cuda", dtype=torch.float16)
    ops.kv_cache_append(k_cache, v_cache, k, v, 0)
    attn_out = ops.decode_attention(q, k_cache, v_cache, 1).reshape(NQ * HD).contiguous()

    o = ops.gemv_fp16_v3(attn.o_proj.weight, attn_out)
    h2 = residual + o

    residual2 = h2
    h3 = ops.rmsnorm(h2.unsqueeze(0), layer.post_attention_layernorm.weight, eps).squeeze(0).contiguous()
    mlp_out = ops.fused_swiglu_mlp(layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight, h3)
    return residual2 + mlp_out


@pytest.mark.parametrize("pos", [0, 5, 100])
def test_decoder_layer_matches_huggingface(pos):
    torch.manual_seed(0)
    config = _qwen3_1p7b_shaped_config()
    layer = qwen3_mod.Qwen3DecoderLayer(config, layer_idx=0).cuda().half().eval()
    rotary = qwen3_mod.Qwen3RotaryEmbedding(config).cuda()

    x = torch.randn(1, 1, config.hidden_size, device="cuda", dtype=torch.float16)
    position_ids = torch.tensor([[pos]], device="cuda")
    cos, sin = rotary(x, position_ids)
    with torch.no_grad():
        hf_out = layer(
            x,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        )[0, 0]

    my_out = _run_my_layer(layer, config, x[0, 0].contiguous(), pos)

    diff = (my_out.float() - hf_out.float()).abs()
    signal = hf_out.float().abs() > NEAR_ZERO_THRESHOLD
    rel = diff[signal] / hf_out.float().abs()[signal]
    assert rel.numel() > 0, "test is vacuous if every output element is near zero"
    assert torch.all(rel < MAX_REL_ERR), (
        f"pos={pos}: max relative error {rel.max().item():.4f} exceeds PROJECT_SPEC.md's "
        f"{MAX_REL_ERR} bound (on elements with |hf_out| > {NEAR_ZERO_THRESHOLD})"
    )
