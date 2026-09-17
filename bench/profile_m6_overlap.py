"""M6 acceptance: "Nsight Systems timeline shows copy and compute kernels
genuinely overlapping, not serialized. Report the overlap efficiency
(achieved vs ideal)."

This is the WORKLOAD half of that check -- run it under `nsys profile`
(not directly; it needs no CUDA GPU work of its own besides what's
profiled) to produce a .nsys-rep trace, then feed that trace's
`cuda_gpu_trace` report to analyze_nsys_overlap.py for the actual
achieved/ideal overlap number. Uses a small synthetic model (same
per-layer shapes as Qwen3-14B, random weights) rather than the real
downloaded checkpoint -- this profiles the streaming *mechanism*
(soinfer.runtime.generate.WeightPipeline), which doesn't depend on which
weights are actually being moved.

Usage (on the Colab/Kaggle T4 session; nsys ships with the CUDA toolkit,
possibly not on PATH -- e.g. under
/opt/nvidia/nsight-compute/<version>/host/target-linux-x64/nsys):

    nsys profile --trace=cuda --capture-range=cudaProfilerApi \\
        -o reports/m6_overlap_profile bench/profile_m6_overlap.py
    nsys stats --report cuda_gpu_trace --format csv \\
        --output reports/m6_overlap_profile reports/m6_overlap_profile.nsys-rep
    python bench/analyze_nsys_overlap.py reports/m6_overlap_profile_cuda_gpu_trace.csv
"""
from __future__ import annotations

import torch

from soinfer.offload import load_hf_checkpoint, weight_store
from soinfer.quant import formats, pack
from soinfer.runtime.generate import StreamingModel, WeightPipeline, run_decoder_layer_streaming

CFG = dict(hidden_size=5120, intermediate_size=17408, num_attention_heads=40, num_key_value_heads=8, head_dim=128)
NUM_LAYERS = 2  # small: this profiles the pipelining mechanism, not a full model
GROUP_SIZE = 128
N_LAYER_FORWARDS = 6


def _make_synth_matrix(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    w = torch.randn(n, k)
    config = formats.QuantConfig(bits=4, granularity="group", group_size=GROUP_SIZE)
    qt = formats.quantize(w, config)
    packed, packed_k = pack.pack_int4(qt.qweight)
    return packed, qt.scale.contiguous(), packed_k


def build_synthetic_model() -> StreamingModel:
    shapes = {
        "self_attn.q_proj.weight": (CFG["num_attention_heads"] * CFG["head_dim"], CFG["hidden_size"]),
        "self_attn.k_proj.weight": (CFG["num_key_value_heads"] * CFG["head_dim"], CFG["hidden_size"]),
        "self_attn.v_proj.weight": (CFG["num_key_value_heads"] * CFG["head_dim"], CFG["hidden_size"]),
        "self_attn.o_proj.weight": (CFG["hidden_size"], CFG["num_attention_heads"] * CFG["head_dim"]),
        "mlp.gate_proj.weight": (CFG["intermediate_size"], CFG["hidden_size"]),
        "mlp.up_proj.weight": (CFG["intermediate_size"], CFG["hidden_size"]),
        "mlp.down_proj.weight": (CFG["hidden_size"], CFG["intermediate_size"]),
    }
    total_bytes = sum(n * (-(-k // 8) * 4) for n, k in shapes.values()) * NUM_LAYERS
    store = weight_store.PinnedWeightStore(total_bytes=total_bytes)
    matrices = {}
    for i in range(NUM_LAYERS):
        for suffix, (n, k) in shapes.items():
            packed, scale, packed_k = _make_synth_matrix(n, k)
            handle = store.register(f"model.layers.{i}.{suffix}", packed)
            matrices[f"model.layers.{i}.{suffix}"] = load_hf_checkpoint.LoadedMatrix(
                handle=handle, scale=scale.cuda(), packed_k=packed_k, n=n, k=k
            )

    H = CFG["hidden_size"]
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

    return StreamingModel(
        store=store, matrices=matrices, num_layers=NUM_LAYERS, embed_tokens=embed_tokens, lm_head=lm_head,
        final_norm=final_norm, layer_norms=layer_norms, hidden_size=H,
        num_attention_heads=CFG["num_attention_heads"], num_key_value_heads=CFG["num_key_value_heads"],
        head_dim=CFG["head_dim"], rms_norm_eps=1e-6, rope_theta=1_000_000.0, group_size=GROUP_SIZE,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("profile_m6_overlap.py requires a CUDA GPU (run this on the Colab/Kaggle T4 session)")

    model = build_synthetic_model()
    k_cache = torch.zeros(CFG["num_key_value_heads"], 32, CFG["head_dim"], device="cuda", dtype=torch.float16)
    v_cache = torch.zeros(CFG["num_key_value_heads"], 32, CFG["head_dim"], device="cuda", dtype=torch.float16)
    x = torch.randn(CFG["hidden_size"], device="cuda", dtype=torch.float16)

    pipeline = WeightPipeline(model)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()  # nsys --capture-range=cudaProfilerApi starts recording here
    for i in range(N_LAYER_FORWARDS):
        x = run_decoder_layer_streaming(pipeline, model, i % NUM_LAYERS, x, i % 32, k_cache, v_cache)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"profiled {N_LAYER_FORWARDS} layer forwards")


if __name__ == "__main__":
    main()
