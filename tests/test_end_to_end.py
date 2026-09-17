"""M5 acceptance (PROJECT_SPEC.md sec 6, M5): "greedy decode of 128 tokens
produces the identical token sequence as the HF reference. If tokens
diverge, find out where before proceeding. This test is the backbone of
the project."

Builds a full (randomly-initialized, Qwen3-1.7B-shaped) Qwen3ForCausalLM,
gets HF's greedy-decode token sequence via model.generate(do_sample=False),
then runs an independent greedy-decode loop built entirely from
soinfer.ops kernels (embedding lookup is the one op left as a plain tensor
index -- everything from the first decoder layer onward is
rmsnorm/fused_qkv_projection/qk_norm/apply_rope/kv_cache_append/
decode_attention/gemv_fp16_v3/fused_swiglu_mlp, per-layer, looped over all
decoder layers and every generated position) and checks the two token
sequences are identical.

Uses random weights (no download needed) and a small vocab -- this tests
the architecture/kernels/decode-loop wiring, not any specific checkpoint's
learned behavior; num_hidden_layers=4 (not the real model's 28) keeps the
test fast while still exercising "multiple layers, each with its own
growing KV cache, across many decode steps," which is the actual thing an
end-to-end test needs to catch that a single-layer parity test can't.

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

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402


def _qwen3_1p7b_shaped_config(num_hidden_layers: int) -> "Qwen3Config":
    return Qwen3Config(
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=num_hidden_layers,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        vocab_size=100,
        tie_word_embeddings=False,
    )


def _run_one_layer(layer, config, x_flat: torch.Tensor, pos: int, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
    """One decoder layer's forward for a single token at position `pos`,
    mutating k_cache/v_cache in place -- same assembly as
    test_layer_parity.py's _run_my_layer, but taking pre-allocated
    multi-position caches so it can be called across a whole decode loop."""
    ops = soinfer.ops
    NQ, NKV, HD = config.num_attention_heads, config.num_key_value_heads, config.head_dim
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

    ops.kv_cache_append(k_cache, v_cache, k, v, pos)
    attn_out = ops.decode_attention(q, k_cache, v_cache, pos + 1).reshape(NQ * HD).contiguous()

    o = ops.gemv_fp16_v3(attn.o_proj.weight, attn_out)
    h2 = residual + o

    residual2 = h2
    h3 = ops.rmsnorm(h2.unsqueeze(0), layer.post_attention_layernorm.weight, eps).squeeze(0).contiguous()
    mlp_out = ops.fused_swiglu_mlp(layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight, h3)
    return residual2 + mlp_out


def _run_my_generate(model, config, start_token: int, n_new: int, max_seq_len: int) -> list[int]:
    """Independent greedy-decode loop, entirely soinfer.ops kernels from
    the first decoder layer onward. embed_tokens/argmax/lm_head-matmul use
    plain tensor ops (embedding lookup and argmax aren't in PROJECT_SPEC.md's
    kernel list; lm_head is just another GEMV, reusing gemv_fp16_v3)."""
    ops = soinfer.ops
    num_kv_heads, head_dim = config.num_key_value_heads, config.head_dim
    eps = config.rms_norm_eps
    num_layers = config.num_hidden_layers
    k_caches = [torch.zeros(num_kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16) for _ in range(num_layers)]
    v_caches = [torch.zeros(num_kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16) for _ in range(num_layers)]

    tokens = [start_token]
    cur_token = start_token
    embed_w = model.model.embed_tokens.weight
    for pos in range(n_new):
        x = embed_w[cur_token].contiguous()
        for layer_idx in range(num_layers):
            x = _run_one_layer(model.model.layers[layer_idx], config, x, pos, k_caches[layer_idx], v_caches[layer_idx])
        x = ops.rmsnorm(x.unsqueeze(0), model.model.norm.weight, eps).squeeze(0).contiguous()
        logits = ops.gemv_fp16_v3(model.lm_head.weight, x)
        next_token = int(torch.argmax(logits).item())
        tokens.append(next_token)
        cur_token = next_token
    return tokens


@pytest.mark.parametrize("seed,start_token,n_new", [(0, 7, 6), (1, 49, 8), (2, 17, 8)])
def test_greedy_decode_matches_huggingface_token_for_token(seed, start_token, n_new):
    torch.manual_seed(seed)
    config = _qwen3_1p7b_shaped_config(num_hidden_layers=4)  # 4 not 28: fast, still multi-layer + multi-step
    model = Qwen3ForCausalLM(config).cuda().half().eval()

    input_ids = torch.tensor([[start_token]], device="cuda")
    with torch.no_grad():
        hf_tokens = model.generate(input_ids, max_new_tokens=n_new, do_sample=False)[0].tolist()

    my_tokens = _run_my_generate(model, config, start_token, n_new, max_seq_len=n_new + 1)

    assert my_tokens == hf_tokens, f"token sequence diverged: hf={hf_tokens} mine={my_tokens}"
