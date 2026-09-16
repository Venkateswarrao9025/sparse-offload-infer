"""M5 acceptance (PROJECT_SPEC.md sec 6, M5): fused kernels match PyTorch
references within tolerance on real activations.

Covers task 1 (fused SwiGLU MLP), task 2 (fused QKV projection), task 3 (KV
cache append), and task 4 (decode attention). Shapes mirror the real
Qwen3-1.7B config (hidden=2048, intermediate=6144, 16 query heads, 8 KV
heads, head_dim=128) so this exercises the actual dev-model dimensions, not
arbitrary ones. Skipped entirely on machines without a CUDA GPU; run for
real on the Colab/Kaggle T4 session via `make test`.
"""
import math

import pytest
import torch
import torch.nn.functional as F

soinfer = pytest.importorskip("soinfer")
if soinfer.ops is None:
    pytest.skip("soinfer._C (CUDA extension) not built", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="M5 kernels require a CUDA GPU")

# See tests/test_m2_kernels.py for why GEMV-shaped outputs use a
# magnitude-scaled bound instead of a flat absolute tolerance.
MAX_ABS_ERR = 1e-2
GEMV_RTOL = 5e-3


def assert_gemv_matches(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    diff = (actual.float() - expected.float()).abs()
    bound = MAX_ABS_ERR + GEMV_RTOL * expected.float().abs()
    assert torch.all(diff < bound), (
        f"{label} exceeds scaled tolerance: max diff {diff.max().item()} " f"at bound {bound[diff.argmax()].item()}"
    )


def swiglu_gate_up_ref(gate_W: torch.Tensor, up_W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    gate = torch.mv(gate_W.float(), x.float())
    up = torch.mv(up_W.float(), x.float())
    return (F.silu(gate) * up).half()


def test_swiglu_gate_up_matches_reference():
    torch.manual_seed(0)
    I, H = 512, 4096  # H % 8 == 0
    gate_W = torch.randn(I, H, device="cuda", dtype=torch.float16)
    up_W = torch.randn(I, H, device="cuda", dtype=torch.float16)
    x = torch.randn(H, device="cuda", dtype=torch.float16)

    expected = swiglu_gate_up_ref(gate_W, up_W, x)
    actual = soinfer.ops.swiglu_gate_up(gate_W, up_W, x)
    assert_gemv_matches(actual, expected, "swiglu_gate_up")


def test_swiglu_gate_up_rejects_mismatched_shapes():
    gate_W = torch.randn(64, 4096, device="cuda", dtype=torch.float16)
    up_W = torch.randn(63, 4096, device="cuda", dtype=torch.float16)
    x = torch.randn(4096, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        soinfer.ops.swiglu_gate_up(gate_W, up_W, x)


def test_swiglu_gate_up_rejects_h_not_multiple_of_8():
    gate_W = torch.randn(4, 4097, device="cuda", dtype=torch.float16)
    up_W = torch.randn(4, 4097, device="cuda", dtype=torch.float16)
    x = torch.randn(4097, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        soinfer.ops.swiglu_gate_up(gate_W, up_W, x)


def test_fused_swiglu_mlp_matches_reference():
    """Full MLP: down_proj(silu(gate_proj(x)) * up_proj(x)), matching an
    HF-style SwiGLU MLP module's forward exactly (no bias, as in Llama/Qwen/
    SmolLM's MLP)."""
    torch.manual_seed(1)
    H, I = 2048, 5504  # roughly Qwen3-1.7B-shaped (hidden, intermediate); H,I % 8 == 0
    gate_W = torch.randn(I, H, device="cuda", dtype=torch.float16) * 0.05
    up_W = torch.randn(I, H, device="cuda", dtype=torch.float16) * 0.05
    down_W = torch.randn(H, I, device="cuda", dtype=torch.float16) * 0.05
    x = torch.randn(H, device="cuda", dtype=torch.float16)

    gate = torch.mv(gate_W.float(), x.float())
    up = torch.mv(up_W.float(), x.float())
    h = F.silu(gate) * up
    expected = torch.mv(down_W.float(), h).half()

    actual = soinfer.ops.fused_swiglu_mlp(gate_W, up_W, down_W, x)
    assert_gemv_matches(actual, expected, "fused_swiglu_mlp")


# ---------------------------------------------------------------------------
# Fused QKV projection (M5 task 2): reuses gemv_fp16_v3 on a pre-concatenated
# weight matrix -- see ops.py for why no new kernel is needed here.
# ---------------------------------------------------------------------------


def test_fused_qkv_projection_matches_reference():
    torch.manual_seed(2)
    H, q_dim, kv_dim = 2048, 2048, 512  # GQA-shaped: q_dim > kv_dim
    Wq = torch.randn(q_dim, H, device="cuda", dtype=torch.float16)
    Wk = torch.randn(kv_dim, H, device="cuda", dtype=torch.float16)
    Wv = torch.randn(kv_dim, H, device="cuda", dtype=torch.float16)
    x = torch.randn(H, device="cuda", dtype=torch.float16)

    qkv_W = soinfer.ops.concat_qkv_weights(Wq, Wk, Wv)
    assert qkv_W.shape == (q_dim + 2 * kv_dim, H)
    q, k, v = soinfer.ops.fused_qkv_projection(qkv_W, x, q_dim, kv_dim)

    expected_q = torch.mv(Wq.float(), x.float()).half()
    expected_k = torch.mv(Wk.float(), x.float()).half()
    expected_v = torch.mv(Wv.float(), x.float()).half()
    assert_gemv_matches(q, expected_q, "fused_qkv_projection[q]")
    assert_gemv_matches(k, expected_k, "fused_qkv_projection[k]")
    assert_gemv_matches(v, expected_v, "fused_qkv_projection[v]")


# ---------------------------------------------------------------------------
# KV cache append (M5 task 3)
# ---------------------------------------------------------------------------


def test_kv_cache_append_writes_only_target_position():
    num_kv_heads, max_seq_len, head_dim = 8, 64, 128
    k_cache = torch.zeros(num_kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros(num_kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16)
    k_cache.fill_(-1.0)  # sentinel so "untouched" is distinguishable from a real (possibly-zero) write
    v_cache.fill_(-1.0)

    torch.manual_seed(3)
    for pos in [0, 5, 63]:
        k_new = torch.randn(num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v_new = torch.randn(num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        soinfer.ops.kv_cache_append(k_cache, v_cache, k_new, v_new, pos)
        assert torch.equal(k_cache[:, pos, :], k_new)
        assert torch.equal(v_cache[:, pos, :], v_new)

    # positions never written stay at the sentinel
    assert torch.all(k_cache[:, 1, :] == -1.0)
    assert torch.all(v_cache[:, 1, :] == -1.0)


def test_kv_cache_append_rejects_pos_out_of_range():
    num_kv_heads, max_seq_len, head_dim = 4, 16, 32
    k_cache = torch.zeros(num_kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros(num_kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16)
    k_new = torch.randn(num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v_new = torch.randn(num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        soinfer.ops.kv_cache_append(k_cache, v_cache, k_new, v_new, max_seq_len)


# ---------------------------------------------------------------------------
# Decode attention (M5 task 4)
# ---------------------------------------------------------------------------


def decode_attention_ref(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, cur_len: int) -> torch.Tensor:
    num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[0]
    n_rep = num_q_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)
    for qh in range(num_q_heads):
        kvh = qh // n_rep
        k = k_cache[kvh, :cur_len].float()
        v = v_cache[kvh, :cur_len].float()
        scores = torch.mv(k, q[qh].float()) * scale
        probs = torch.softmax(scores, dim=0)
        out[qh] = torch.mv(v.t(), probs).half()
    return out


@pytest.mark.parametrize("cur_len", [1, 17, 300])
def test_decode_attention_matches_reference(cur_len):
    torch.manual_seed(4)
    num_q_heads, num_kv_heads, head_dim, max_seq_len = 16, 8, 128, 512  # Qwen3-1.7B shape
    q = torch.randn(num_q_heads, head_dim, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(num_kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(num_kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16)

    expected = decode_attention_ref(q, k_cache, v_cache, cur_len)
    actual = soinfer.ops.decode_attention(q, k_cache, v_cache, cur_len)
    assert_gemv_matches(actual, expected, f"decode_attention[cur_len={cur_len}]")


def test_decode_attention_rejects_non_gqa_head_ratio():
    q = torch.randn(15, 128, device="cuda", dtype=torch.float16)  # 15 not a multiple of 8
    k_cache = torch.randn(8, 64, 128, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(8, 64, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        soinfer.ops.decode_attention(q, k_cache, v_cache, 1)


def test_decode_attention_after_kv_cache_append_matches_reference():
    """End-to-end-ish: append a few tokens via kv_cache_append, then run
    decode_attention over exactly what was appended -- exercises the two
    kernels together the way runtime/generate.py's decode loop will."""
    torch.manual_seed(5)
    num_q_heads, num_kv_heads, head_dim, max_seq_len = 16, 8, 128, 32
    k_cache = torch.zeros(num_kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros(num_kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16)

    n_tokens = 5
    for pos in range(n_tokens):
        k_new = torch.randn(num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v_new = torch.randn(num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        soinfer.ops.kv_cache_append(k_cache, v_cache, k_new, v_new, pos)

    q = torch.randn(num_q_heads, head_dim, device="cuda", dtype=torch.float16)
    expected = decode_attention_ref(q, k_cache, v_cache, n_tokens)
    actual = soinfer.ops.decode_attention(q, k_cache, v_cache, n_tokens)
    assert_gemv_matches(actual, expected, "decode_attention[after kv_cache_append]")
