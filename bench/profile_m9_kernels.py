"""M9 task 1: Nsight Compute on every hot kernel -- the ones actually in the
real decode path (dense M6, DIP M7, cache-aware DIP M8), not the M1/M2
pedagogical exercises (add_one, reduce_*, transpose_*, gemv_fp16_v1/v2,
softmax_*, swiglu_gate_up, gemv_w8a16) that were superseded or never used in
production serving.

Runs ONE forward pass through each of the three decode-layer variants plus
one LM-head projection, back to back, wrapped in cudaProfilerStart/Stop --
covers all 12 hot kernels in a single profiling pass instead of one ncu
invocation per kernel:

  rmsnorm, gemv_w4a16_group_lop3, apply_rope, kv_cache_append,
  decode_attention, topk_threshold_select, gather_rows_staged,
  build_dip_descriptors, gemv_dip_fused_up, gemv_dip_fused_down,
  gemv_w4a16_sparse_accumulate, gemv_fp16_v3

Uses a synthetic model at Qwen3-14B's REAL shapes (PROJECT_SPEC.md's
target model/GPU) with random weights, not a downloaded checkpoint -- same
reasoning as profile_m6_overlap.py: kernel occupancy/throughput/stall
characteristics depend on SHAPE, not on which weights are actually loaded,
and this way profiling costs no extra download/compute-unit budget beyond
the profiling run itself.

Usage (Colab/Kaggle T4; ncu ships with the CUDA toolkit, possibly not on
PATH -- e.g. under /opt/nvidia/nsight-compute/<version>/ncu):

    ncu --set basic --capture-range=cudaProfilerApi --csv \\
        --log-file reports/m9_ncu_report.csv \\
        python bench/profile_m9_kernels.py
"""
from __future__ import annotations

import torch

import soinfer.ops as ops
from soinfer.offload import hot_cache, load_hf_checkpoint, weight_store
from soinfer.quant import formats, pack
from soinfer.runtime import generate as gen

CFG = dict(hidden_size=5120, intermediate_size=17408, num_attention_heads=40, num_key_value_heads=8, head_dim=128)
NUM_LAYERS = 1  # one forward pass is enough to launch every hot kernel at least once
GROUP_SIZE = 128
DIP_K = CFG["intermediate_size"] // 2  # k/I=0.5, this project's own chosen operating point
CACHE_FRAC = 0.10


def _quantize_and_pack(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    config = formats.QuantConfig(bits=4, granularity="group", group_size=GROUP_SIZE)
    qt = formats.quantize(w.float(), config)
    packed, packed_k = pack.pack_int4(qt.qweight)
    return packed, qt.scale.contiguous(), packed_k


def build_synthetic_dip_model() -> gen.StreamingModel:
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
    total_bytes += I * (-(-H // 8) * 4) * NUM_LAYERS  # down_proj_T
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

    embed_tokens = torch.randn(100, H, device="cuda", dtype=torch.float16)
    lm_head = torch.randn(100, H, device="cuda", dtype=torch.float16)
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


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("profile_m9_kernels.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    model = build_synthetic_dip_model()
    NKV, HD = CFG["num_key_value_heads"], CFG["head_dim"]
    x = torch.randn(CFG["hidden_size"], device="cuda", dtype=torch.float16)

    hot_indices = sorted(torch.randperm(CFG["intermediate_size"])[: int(CFG["intermediate_size"] * CACHE_FRAC)].tolist())
    caches = [hot_cache.HotCache.build(model, layer_idx=0, hot_indices=hot_indices)]

    pipeline_dense = gen.WeightPipeline(model)
    pipeline_dip = gen.WeightPipeline(model, suffixes=gen.DIP_DENSE_SUFFIXES)
    bufs_dip = gen.DipBuffers.make(model, max_k=DIP_K)
    pipeline_cached = gen.WeightPipeline(model, suffixes=gen.DIP_DENSE_SUFFIXES)
    bufs_cached = gen.DipBuffers.make(model, max_k=DIP_K)

    k_cache = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros(NKV, 8, HD, device="cuda", dtype=torch.float16)

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()  # ncu --capture-range=cudaProfilerApi starts recording here

    # rmsnorm, gemv_w4a16_group_lop3, apply_rope, kv_cache_append, decode_attention
    gen.run_decoder_layer_streaming(pipeline_dense, model, 0, x, 0, k_cache, v_cache)

    # + topk_threshold_select, gather_rows_staged, gemv_w4a16_sparse_accumulate
    gen.run_decoder_layer_dip(pipeline_dip, model, 0, x, 0, k_cache, v_cache, DIP_K, bufs_dip)

    # + build_dip_descriptors, gemv_dip_fused_up, gemv_dip_fused_down
    gen.run_decoder_layer_cached_dip(pipeline_cached, model, 0, x, 0, k_cache, v_cache, DIP_K, bufs_cached, caches)

    # + gemv_fp16_v3 (LM head projection, not inside any decoder layer)
    xn = ops.rmsnorm(x.unsqueeze(0), model.final_norm, model.rms_norm_eps).squeeze(0).contiguous()
    ops.gemv_fp16_v3(model.lm_head, xn)

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("profiled one forward pass through dense, DIP, and cache-aware-DIP decoder layers, plus the LM head")


if __name__ == "__main__":
    main()
