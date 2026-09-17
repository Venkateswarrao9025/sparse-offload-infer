"""M7 task 4 / integration: wires the already-verified DIP primitives
(topk_select, gather_rows, both sparse GEMVs -- test_m7_kernels.py,
test_gather_rows.py, test_m7_sparse_gemv.py) into the actual streaming
decode loop (runtime/generate.py's run_decoder_layer_dip/generate_dip).
Uses a small SYNTHETIC model (same construction as
bench/profile_m6_overlap.py's build_synthetic_model, but tiny for test
speed, with down_proj ALSO registered transposed) -- no real HF checkpoint
needed, so this runs without the multi-minute download bench_m7_pareto.py
needs.

Skipped entirely on machines without a CUDA GPU; run for real on the
Colab/Kaggle T4 session via `make test`.
"""
import pytest
import torch

soinfer = pytest.importorskip("soinfer")
if soinfer.ops is None:
    pytest.skip("soinfer._C (CUDA extension) not built", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="M7 DIP pipeline requires a CUDA GPU")

from soinfer.offload import load_hf_checkpoint, weight_store  # noqa: E402
from soinfer.quant import formats, pack  # noqa: E402
from soinfer.runtime import generate as gen  # noqa: E402

CFG = dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
NUM_LAYERS = 2
GROUP_SIZE = 32


def _quantize_and_pack(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    config = formats.QuantConfig(bits=4, granularity="group", group_size=GROUP_SIZE)
    qt = formats.quantize(w.float(), config)
    packed, packed_k = pack.pack_int4(qt.qweight)
    return packed, qt.scale.contiguous(), packed_k


def _build_synthetic_dip_model(seed: int = 0) -> gen.StreamingModel:
    """Same shape/registration pattern as profile_m6_overlap.py's
    build_synthetic_model, plus a down_proj_T entry per layer (what
    load_hf_checkpoint.stream_load_layers(include_down_proj_t=True)
    would produce from a real checkpoint)."""
    torch.manual_seed(seed)
    H, I = CFG["hidden_size"], CFG["intermediate_size"]
    shapes = {
        "self_attn.q_proj.weight": (CFG["num_attention_heads"] * CFG["head_dim"], H),
        "self_attn.k_proj.weight": (CFG["num_key_value_heads"] * CFG["head_dim"], H),
        "self_attn.v_proj.weight": (CFG["num_key_value_heads"] * CFG["head_dim"], H),
        "self_attn.o_proj.weight": (H, CFG["num_attention_heads"] * CFG["head_dim"]),
        "mlp.gate_proj.weight": (I, H),
        "mlp.up_proj.weight": (I, H),
        "mlp.down_proj.weight": (H, I),
    }
    total_bytes = sum(n * (-(-k // 8) * 4) for n, k in shapes.values()) * NUM_LAYERS
    total_bytes += I * (-(-H // 8) * 4) * NUM_LAYERS  # down_proj_T: [I, H]
    store = weight_store.PinnedWeightStore(total_bytes=total_bytes)
    matrices = {}
    for i in range(NUM_LAYERS):
        for suffix, (n, k) in shapes.items():
            w = torch.randn(n, k)
            packed, scale, packed_k = _quantize_and_pack(w)
            handle = store.register(f"model.layers.{i}.{suffix}", packed)
            matrices[f"model.layers.{i}.{suffix}"] = load_hf_checkpoint.LoadedMatrix(
                handle=handle, scale=scale.cuda(), packed_k=packed_k, n=n, k=k
            )
            if suffix == "mlp.down_proj.weight":
                wT = w.t().contiguous()
                packedT, scaleT, packed_kT = _quantize_and_pack(wT)
                handleT = store.register(f"model.layers.{i}.{load_hf_checkpoint.DOWN_PROJ_T_SUFFIX}", packedT)
                matrices[f"model.layers.{i}.{load_hf_checkpoint.DOWN_PROJ_T_SUFFIX}"] = load_hf_checkpoint.LoadedMatrix(
                    handle=handleT, scale=scaleT.cuda(), packed_k=packed_kT, n=I, k=H
                )

    embed_tokens = torch.randn(50, H, device="cuda", dtype=torch.float16)
    lm_head = torch.randn(50, H, device="cuda", dtype=torch.float16)
    final_norm = torch.ones(H, device="cuda", dtype=torch.float16)
    layer_norms = [
        {
            "input_layernorm": torch.ones(H, device="cuda", dtype=torch.float16),
            "post_attention_layernorm": torch.ones(H, device="cuda", dtype=torch.float16),
            "q_norm": torch.ones(CFG["head_dim"], device="cuda", dtype=torch.float16),
            "k_norm": torch.ones(CFG["head_dim"], device="cuda", dtype=torch.float16),
        }
        for _ in range(NUM_LAYERS)
    ]
    return gen.StreamingModel(
        store=store, matrices=matrices, num_layers=NUM_LAYERS, embed_tokens=embed_tokens, lm_head=lm_head,
        final_norm=final_norm, layer_norms=layer_norms, hidden_size=H,
        num_attention_heads=CFG["num_attention_heads"], num_key_value_heads=CFG["num_key_value_heads"],
        head_dim=CFG["head_dim"], rms_norm_eps=1e-6, rope_theta=1_000_000.0,
    )


def test_dip_at_full_k_matches_dense_reference():
    """dip_k == intermediate_size: topk_select keeps every channel, so
    run_decoder_layer_dip's output should closely match
    run_decoder_layer_streaming's (M6, unchanged, already verified) on the
    SAME underlying weights -- the one structural difference is down_proj
    vs down_proj_T being independently quantized (different per-group
    scale groupings along a transposed axis), so this checks numerical
    closeness, not bit-exactness."""
    I = CFG["intermediate_size"]
    NKV, HD = CFG["num_key_value_heads"], CFG["head_dim"]

    model_dense = _build_synthetic_dip_model(seed=7)
    model_dip = _build_synthetic_dip_model(seed=7)  # same seed -> identical weights, both representations

    x = torch.randn(CFG["hidden_size"], device="cuda", dtype=torch.float16)
    k_cache_d = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)
    v_cache_d = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)
    k_cache_s = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)
    v_cache_s = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)

    pipeline_dense = gen.WeightPipeline(model_dense)
    out_dense = gen.run_decoder_layer_streaming(pipeline_dense, model_dense, 0, x, 0, k_cache_d, v_cache_d)

    pipeline_dip = gen.WeightPipeline(model_dip, suffixes=gen.DIP_DENSE_SUFFIXES)
    bufs = gen.DipBuffers.make(model_dip, max_k=I)
    out_dip = gen.run_decoder_layer_dip(pipeline_dip, model_dip, 0, x, 0, k_cache_s, v_cache_s, I, bufs)

    diff = (out_dip.float() - out_dense.float()).abs()
    bound = 0.5 + 2e-2 * out_dense.float().abs()
    assert torch.all(diff < bound), f"max diff {diff.max().item()} at bound {bound[diff.argmax()].item()}"


def test_dip_at_partial_k_runs_and_stays_finite():
    I = CFG["intermediate_size"]
    NKV, HD = CFG["num_key_value_heads"], CFG["head_dim"]
    model = _build_synthetic_dip_model(seed=3)
    x = torch.randn(CFG["hidden_size"], device="cuda", dtype=torch.float16)
    k_cache = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)

    pipeline = gen.WeightPipeline(model, suffixes=gen.DIP_DENSE_SUFFIXES)
    bufs = gen.DipBuffers.make(model, max_k=I)
    for k in (I, I // 2, I // 8, 1):
        out = gen.run_decoder_layer_dip(pipeline, model, 0, x, 0, k_cache, v_cache, k, bufs)
        assert out.shape == (CFG["hidden_size"],)
        assert torch.isfinite(out).all()


def test_generate_dip_produces_valid_tokens():
    model = _build_synthetic_dip_model(seed=11)
    vocab = model.embed_tokens.shape[0]
    tokens = gen.generate_dip(model, prompt_ids=[1, 2, 3], n_new=4, dip_k=CFG["intermediate_size"] // 2)
    assert len(tokens) >= 3
    assert all(0 <= t < vocab for t in tokens)


def test_dip_bytes_per_token_matches_hand_computed():
    model = _build_synthetic_dip_model(seed=1)
    I = CFG["intermediate_size"]
    dip_k = 16

    dense_total = gen.dip_bytes_per_token(model, dip_k=I, dense=True)
    dip_total = gen.dip_bytes_per_token(model, dip_k=dip_k, dense=False)

    up0 = model.matrices["model.layers.0.mlp.up_proj.weight"]
    downT0 = model.matrices[f"model.layers.0.{load_hf_checkpoint.DOWN_PROJ_T_SUFFIX}"]
    expected_saved_per_layer = (up0.n - dip_k) * up0.handle.row_nbytes + (downT0.n - dip_k) * downT0.handle.row_nbytes
    assert dense_total - dip_total == expected_saved_per_layer * NUM_LAYERS
    assert dip_total < dense_total


def test_teacher_forced_nll_is_finite_and_positive():
    model = _build_synthetic_dip_model(seed=5)
    vocab = model.embed_tokens.shape[0]
    tokens = [1, 5, 2, 8, 3]

    pipeline = gen.WeightPipeline(model)
    layer_fn = lambda m, li, x, p, kc, vc: gen.run_decoder_layer_streaming(pipeline, m, li, x, p, kc, vc)  # noqa: E731
    nll = gen.teacher_forced_nll(model, layer_fn, tokens)
    assert nll > 0.0 and nll < 50.0  # sanity range: random small model, small vocab (50)
