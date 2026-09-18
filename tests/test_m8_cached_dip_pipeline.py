"""M8 task 4 / integration: wires HotCache (M8 task 2, verified in
test_m8_hot_cache_gpu.py) and the fused cache-or-stream GEMVs (M8 task 3,
verified in test_m8_fused_gemv.py) into the actual streaming decode loop
(runtime/generate.py's run_decoder_layer_cached_dip/generate_cached_dip) --
the piece bench/bench_m8_ablation.py's ablation table actually calls.

The central correctness claim: for the SAME model, token, and dip_k,
run_decoder_layer_cached_dip must produce the EXACT SAME output as
run_decoder_layer_dip (M7, already verified) regardless of which channels
happen to be cache-resident -- caching changes WHERE a selected channel's
weight bytes come from, never the arithmetic. Uses a small SYNTHETIC model
(same construction as test_m7_dip_pipeline.py's) -- no real HF checkpoint
needed.

Skipped entirely on machines without a CUDA GPU; run for real on the
Colab/Kaggle T4 session via `make test`.
"""
import pytest
import torch

soinfer = pytest.importorskip("soinfer")
if soinfer.ops is None:
    pytest.skip("soinfer._C (CUDA extension) not built", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="M8 cached-DIP pipeline requires a CUDA GPU")

from soinfer.offload import load_hf_checkpoint, weight_store  # noqa: E402
from soinfer.offload.hot_cache import HotCache  # noqa: E402
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
    """Identical construction to test_m7_dip_pipeline.py's helper of the
    same name (duplicated per this project's established per-file
    convention, see e.g. test_m8_calibration.py's own copy)."""
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
    total_bytes += I * (-(-H // 8) * 4) * NUM_LAYERS
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
        head_dim=CFG["head_dim"], rms_norm_eps=1e-6, rope_theta=1_000_000.0, group_size=GROUP_SIZE,
    )


def _build_caches(model: gen.StreamingModel, hot_indices) -> list[HotCache]:
    return [HotCache.build(model, layer_idx=li, hot_indices=hot_indices) for li in range(model.num_layers)]


@pytest.mark.parametrize("cache_frac", [1.0, 0.0, 0.5])
def test_cached_dip_matches_dip_reference_regardless_of_cache_split(cache_frac):
    """The core claim: caching changes WHERE bytes come from, never the
    arithmetic. cache_frac=1.0 (everything cached, miss_count always 0)
    and 0.0 (nothing cached, always the staging path) are the degenerate
    cases already covered at the kernel level by test_m8_fused_gemv.py;
    0.5 is the realistic mixed case. All three must be EXACTLY (not
    approximately) equal to run_decoder_layer_dip's output -- both paths
    select channels via the same (now-deterministic, see topk_select.cu's
    tie-break fix) top-k kernel on the identical gate values, so there is
    no source of legitimate numerical divergence between them."""
    I = CFG["intermediate_size"]
    NKV, HD = CFG["num_key_value_heads"], CFG["head_dim"]
    dip_k = 24

    model_ref = _build_synthetic_dip_model(seed=17)
    model_cached = _build_synthetic_dip_model(seed=17)  # same seed -> identical weights

    x = torch.randn(CFG["hidden_size"], device="cuda", dtype=torch.float16)
    k_cache_r = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)
    v_cache_r = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)
    k_cache_c = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)
    v_cache_c = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)

    pipeline_ref = gen.WeightPipeline(model_ref, suffixes=gen.DIP_DENSE_SUFFIXES)
    bufs_ref = gen.DipBuffers.make(model_ref, max_k=dip_k)
    out_ref = gen.run_decoder_layer_dip(pipeline_ref, model_ref, 0, x, 0, k_cache_r, v_cache_r, dip_k, bufs_ref)

    hot_indices = sorted(torch.randperm(I)[: int(I * cache_frac)].tolist())
    caches = _build_caches(model_cached, hot_indices)
    pipeline_cached = gen.WeightPipeline(model_cached, suffixes=gen.DIP_DENSE_SUFFIXES)
    bufs_cached = gen.DipBuffers.make(model_cached, max_k=dip_k)
    out_cached = gen.run_decoder_layer_cached_dip(pipeline_cached, model_cached, 0, x, 0, k_cache_c, v_cache_c,
                                                   dip_k, bufs_cached, caches)

    assert torch.equal(out_cached, out_ref)


def test_cached_dip_miss_bytes_are_zero_with_a_full_cache():
    I = CFG["intermediate_size"]
    NKV, HD = CFG["num_key_value_heads"], CFG["head_dim"]
    dip_k = 16
    model = _build_synthetic_dip_model(seed=23)
    caches = _build_caches(model, list(range(I)))  # everything cached

    x = torch.randn(CFG["hidden_size"], device="cuda", dtype=torch.float16)
    k_cache = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)
    pipeline = gen.WeightPipeline(model, suffixes=gen.DIP_DENSE_SUFFIXES)
    bufs = gen.DipBuffers.make(model, max_k=dip_k)
    miss_bytes = [0] * model.num_layers

    gen.run_decoder_layer_cached_dip(pipeline, model, 0, x, 0, k_cache, v_cache, dip_k, bufs, caches, miss_bytes)
    assert miss_bytes[0] == 0
    assert miss_bytes[1] == 0  # untouched layer stays at its init value


def test_cached_dip_miss_bytes_match_dense_dip_bytes_with_an_empty_cache():
    I = CFG["intermediate_size"]
    NKV, HD = CFG["num_key_value_heads"], CFG["head_dim"]
    dip_k = 16
    model = _build_synthetic_dip_model(seed=29)
    caches = _build_caches(model, [])  # nothing cached -> every selection is a miss

    x = torch.randn(CFG["hidden_size"], device="cuda", dtype=torch.float16)
    k_cache = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)
    pipeline = gen.WeightPipeline(model, suffixes=gen.DIP_DENSE_SUFFIXES)
    bufs = gen.DipBuffers.make(model, max_k=dip_k)
    miss_bytes = [0] * model.num_layers

    gen.run_decoder_layer_cached_dip(pipeline, model, 0, x, 0, k_cache, v_cache, dip_k, bufs, caches, miss_bytes)

    up0 = model.matrices["model.layers.0.mlp.up_proj.weight"]
    downT0 = model.matrices[f"model.layers.0.{load_hf_checkpoint.DOWN_PROJ_T_SUFFIX}"]
    expected = dip_k * (up0.handle.row_nbytes + downT0.handle.row_nbytes)
    assert miss_bytes[0] == expected


def test_generate_cached_dip_produces_valid_tokens():
    I = CFG["intermediate_size"]
    model = _build_synthetic_dip_model(seed=31)
    vocab = model.embed_tokens.shape[0]
    hot_indices = sorted(torch.randperm(I)[: I // 2].tolist())
    caches = _build_caches(model, hot_indices)

    tokens = gen.generate_cached_dip(model, prompt_ids=[1, 2, 3], n_new=4, dip_k=I // 4, caches=caches)
    assert len(tokens) >= 3
    assert all(0 <= t < vocab for t in tokens)
