"""M8 task 1: calibrate_channel_frequencies' bookkeeping, on a small
synthetic model (same construction as test_m7_dip_pipeline.py's, and
bench/profile_m6_overlap.py's before that) -- no real HF checkpoint
needed.

Skipped entirely on machines without a CUDA GPU; run for real on the
Colab/Kaggle T4 session via `make test`.
"""
import pytest
import torch

soinfer = pytest.importorskip("soinfer")
if soinfer.ops is None:
    pytest.skip("soinfer._C (CUDA extension) not built", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="M8 calibration requires a CUDA GPU")

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


def test_calibrate_channel_frequencies_counts_are_consistent():
    I = CFG["intermediate_size"]
    dip_k = 16
    model = _build_synthetic_dip_model(seed=42)
    token_ids = [1, 5, 2, 8, 3, 7, 4, 9]

    channel_counts = gen.calibrate_channel_frequencies(model, token_ids, dip_k)

    assert len(channel_counts) == NUM_LAYERS
    for counts in channel_counts:
        assert counts.shape == (I,)
        assert counts.dtype == torch.int64
        # every one of len(token_ids) forward passes selects EXACTLY dip_k
        # (unique, per topk_threshold_select's set semantics) channels, so
        # counts must sum to exactly len(token_ids) * dip_k, and no channel
        # can be selected more than once per token (count per token step is
        # implicitly <= 1 per channel since indices within one call are
        # unique) -- but across len(token_ids) calls a channel CAN exceed 1.
        assert int(counts.sum().item()) == len(token_ids) * dip_k
        assert (counts >= 0).all()
        assert counts.max().item() <= len(token_ids)  # can't be selected more often than there are tokens


def test_calibrate_channel_frequencies_is_deterministic():
    I = CFG["intermediate_size"]
    dip_k = 16
    model_a = _build_synthetic_dip_model(seed=9)
    model_b = _build_synthetic_dip_model(seed=9)
    token_ids = [2, 4, 6, 1]

    counts_a = gen.calibrate_channel_frequencies(model_a, token_ids, dip_k)
    counts_b = gen.calibrate_channel_frequencies(model_b, token_ids, dip_k)

    for ca, cb in zip(counts_a, counts_b):
        assert torch.equal(ca, cb)
