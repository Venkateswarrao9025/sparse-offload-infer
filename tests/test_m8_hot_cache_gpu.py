"""M8 task 2/3: HotCache, the GPU-resident hot-channel cache class
(soinfer.offload.hot_cache.HotCache) that the real per-token DIP path
(gemv_dip_fused_up/_down) actually reads from. hot_cache.py's
policy-SIMULATION half (StaticFrequencyPolicy/LRUPolicy/LFUDecayPolicy) is
pure CPU/Python and already covered by test_hot_cache.py -- HotCache.build
itself needs CUDA (it calls gather_rows_staged against a real pinned
arena, M7 task 2's machinery) and was, as of the commit introducing it,
entirely unverified. See docs/LEARNING_NOTES.md's M8 entries.

Skipped entirely on machines without a CUDA GPU; run for real on the
Colab/Kaggle T4 session via `make test`.
"""
import pytest
import torch

soinfer = pytest.importorskip("soinfer")
if soinfer.ops is None:
    pytest.skip("soinfer._C (CUDA extension) not built", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="HotCache requires a CUDA GPU")

from soinfer.offload import hot_cache, load_hf_checkpoint, weight_store  # noqa: E402
from soinfer.quant import formats, pack  # noqa: E402
from soinfer.runtime import generate as gen  # noqa: E402

CFG = dict(hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
NUM_LAYERS = 1
GROUP_SIZE = 32


def _quantize_and_pack(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    config = formats.QuantConfig(bits=4, granularity="group", group_size=GROUP_SIZE)
    qt = formats.quantize(w.float(), config)
    packed, packed_k = pack.pack_int4(qt.qweight)
    return packed, qt.scale, packed_k


def _build_synthetic_model_with_raw_weights(seed: int = 0):
    """Same construction as test_m8_calibration.py's _build_synthetic_dip_model,
    but also returns the RAW (pre-quantization) up_proj/down_proj_T weight
    tensors so tests here can compute an independent dense reference,
    rather than cross-checking HotCache against another gather call on
    the same store (which would just confirm self-consistency, not
    correctness)."""
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
    raw = {}
    for i in range(NUM_LAYERS):
        for suffix, (n, k) in shapes.items():
            w = torch.randn(n, k)
            packed, scale, packed_k = _quantize_and_pack(w)
            handle = store.register(f"model.layers.{i}.{suffix}", packed)
            matrices[f"model.layers.{i}.{suffix}"] = load_hf_checkpoint.LoadedMatrix(
                handle=handle, scale=scale.cuda(), packed_k=packed_k, n=n, k=k
            )
            if suffix == "mlp.up_proj.weight":
                raw["up_W"] = w
            if suffix == "mlp.down_proj.weight":
                raw["down_W"] = w
                wT = w.t().contiguous()  # [I, H], down_proj_T convention
                packedT, scaleT, packed_kT = _quantize_and_pack(wT)
                handleT = store.register(f"model.layers.{i}.{load_hf_checkpoint.DOWN_PROJ_T_SUFFIX}", packedT)
                matrices[f"model.layers.{i}.{load_hf_checkpoint.DOWN_PROJ_T_SUFFIX}"] = load_hf_checkpoint.LoadedMatrix(
                    handle=handleT, scale=scaleT.cuda(), packed_k=packed_kT, n=I, k=H,
                )
                raw["downT_W"] = wT

    embed_tokens = torch.randn(10, H, device="cuda", dtype=torch.float16)
    lm_head = torch.randn(10, H, device="cuda", dtype=torch.float16)
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
    model = gen.StreamingModel(
        store=store, matrices=matrices, num_layers=NUM_LAYERS, embed_tokens=embed_tokens, lm_head=lm_head,
        final_norm=final_norm, layer_norms=layer_norms, hidden_size=H,
        num_attention_heads=CFG["num_attention_heads"], num_key_value_heads=CFG["num_key_value_heads"],
        head_dim=CFG["head_dim"], rms_norm_eps=1e-6, rope_theta=1_000_000.0, group_size=GROUP_SIZE,
    )
    return model, raw


# ---------------------------------------------------------------------------
# HotCache.build correctness: does it gather the RIGHT bytes?
# ---------------------------------------------------------------------------


def test_hot_cache_build_slot_of_matches_hot_indices():
    model, _raw = _build_synthetic_model_with_raw_weights(seed=0)
    I = CFG["intermediate_size"]
    hot_indices = [3, 7, 8, 15, 40, 41, 90]  # arbitrary, unsorted-on-purpose input order below

    cache = hot_cache.HotCache.build(model, layer_idx=0, hot_indices=[90, 3, 41, 8, 15, 7, 40])
    assert cache.cache_size == len(hot_indices)
    assert torch.equal(cache.hot_indices.cpu(), torch.tensor(sorted(hot_indices), dtype=torch.int64))

    slot_of = cache.slot_of.cpu()
    assert slot_of.shape == (I,)
    expected_slot = {c: slot for slot, c in enumerate(sorted(hot_indices))}
    for c in range(I):
        assert int(slot_of[c].item()) == expected_slot.get(c, -1)


def test_hot_cache_build_with_empty_hot_set():
    """Regression: C=0 (nothing cached, e.g. cache_size=0 or before a
    calibration pass has run) used to crash HotCache.build --
    gather_rows_staged rejected a freshly-.pin_memory()'d ZERO-element CPU
    tensor as 'not pinned' (nothing to page-lock for an empty allocation).
    Found by test_m8_cached_dip_pipeline.py's cache_frac=0.0 case."""
    model, _raw = _build_synthetic_model_with_raw_weights(seed=4)
    I = CFG["intermediate_size"]
    cache = hot_cache.HotCache.build(model, layer_idx=0, hot_indices=[])
    assert cache.cache_size == 0
    assert cache.up_Wq.shape[0] == 0
    assert cache.down_Wq.shape[0] == 0
    assert torch.all(cache.slot_of == -1)


def test_hot_cache_build_gathers_correct_quantized_bytes_and_scales():
    """The real correctness question: does HotCache.build's gather_rows_staged
    round-trip actually pull out the SAME bytes/scales that a direct
    index_select on the full (already-known, since this is synthetic)
    weight matrix would -- not just self-consistent with another read of
    the same store, but matching an INDEPENDENTLY tracked ground truth."""
    model, raw = _build_synthetic_model_with_raw_weights(seed=1)
    I, H = CFG["intermediate_size"], CFG["hidden_size"]
    hot_indices = sorted(torch.randperm(I)[: I // 3].tolist())

    cache = hot_cache.HotCache.build(model, layer_idx=0, hot_indices=hot_indices)

    # Independent ground truth: re-quantize the RAW weight (not read back
    # through the store at all) and index_select the same rows.
    up_Wq_full, up_scale_full, _ = _quantize_and_pack(raw["up_W"])
    down_Wq_full, down_scale_full, _ = _quantize_and_pack(raw["downT_W"])
    up_qt = formats.quantize(raw["up_W"].float(), formats.QuantConfig(bits=4, granularity="group", group_size=GROUP_SIZE))
    idx = torch.tensor(hot_indices, dtype=torch.int64)

    assert torch.equal(cache.up_Wq.cpu(), up_Wq_full.index_select(0, idx))
    assert torch.equal(cache.up_scale.cpu(), up_scale_full.index_select(0, idx))
    assert torch.equal(cache.down_Wq.cpu(), down_Wq_full.index_select(0, idx))
    assert torch.equal(cache.down_scale.cpu(), down_scale_full.index_select(0, idx))

    # And the dequantized values actually round-trip to something close to
    # the raw float weight, for the specific rows HotCache chose to cache
    # (sanity on the quantization itself, not just byte-for-byte plumbing).
    up_dequant_hot = formats.dequantize(up_qt)[idx]
    assert torch.allclose(up_dequant_hot, raw["up_W"][idx], atol=0.5)  # int4: coarse but not garbage


# ---------------------------------------------------------------------------
# End-to-end: a real HotCache feeding the real fused GEMVs, cross-checked
# against the M7 (non-cached) dense reference -- ties HotCache.build,
# ops.build_dip_descriptors, ops.gather_rows_staged (for the miss rows,
# exactly like the real per-token DIP path), and both fused GEMV kernels
# together for the first time.
# ---------------------------------------------------------------------------


def test_hot_cache_feeds_real_fused_gemv_pipeline_matches_m7_dense_reference():
    model, raw = _build_synthetic_model_with_raw_weights(seed=2)
    I, H = CFG["intermediate_size"], CFG["hidden_size"]
    dip_k = 20

    # Build the cache from an arbitrary hot set (a real run would use
    # StaticFrequencyPolicy.hot_set from a calibration pass; the choice
    # doesn't matter for correctness -- any hit/miss split must compute
    # the same answer as the dense reference).
    hot_indices = sorted(torch.randperm(I)[: I // 2].tolist())
    cache = hot_cache.HotCache.build(model, layer_idx=0, hot_indices=hot_indices)

    # A token's top-k selection: deliberately mixes channels inside and
    # outside the hot set, so both the cache-hit and staging-miss paths of
    # gemv_dip_fused_* get exercised in the SAME call.
    selected = torch.randperm(I)[:dip_k].int().cuda()
    selected_long = selected.long()

    up_lm = model.matrices["model.layers.0.mlp.up_proj.weight"]
    downT_lm = model.matrices[f"model.layers.0.{load_hf_checkpoint.DOWN_PROJ_T_SUFFIX}"]

    # Real per-token miss-gather, exactly like run_decoder_layer_dip does
    # via DipBuffers -- not an index_select shortcut.
    descriptors, miss_channels, miss_count = soinfer.ops.build_dip_descriptors(selected, cache.slot_of)
    mc = int(miss_count.item())
    miss_idx_cpu = miss_channels[:mc].cpu().long()  # gather_rows_staged needs a 1D int64 CPU tensor

    up_staging = torch.empty(mc, up_lm.handle.row_nbytes, dtype=torch.uint8).pin_memory()
    up_staging_gpu = torch.empty(mc, up_lm.handle.row_nbytes, dtype=torch.uint8, device="cuda")
    soinfer.ops.gather_rows_staged(model.store.matrix_view(up_lm.handle), miss_idx_cpu, up_staging, up_staging_gpu)

    down_staging = torch.empty(mc, downT_lm.handle.row_nbytes, dtype=torch.uint8).pin_memory()
    down_staging_gpu = torch.empty(mc, downT_lm.handle.row_nbytes, dtype=torch.uint8, device="cuda")
    soinfer.ops.gather_rows_staged(model.store.matrix_view(downT_lm.handle), miss_idx_cpu, down_staging,
                                    down_staging_gpu)
    torch.cuda.synchronize()

    staging_up_scale = up_lm.scale.index_select(0, miss_channels[:mc].long())
    staging_down_scale = downT_lm.scale.index_select(0, miss_channels[:mc].long())

    x = torch.randn(H, device="cuda", dtype=torch.float16)
    u_fused = soinfer.ops.gemv_dip_fused_up(cache.up_Wq, cache.up_scale, up_staging_gpu, staging_up_scale,
                                             descriptors, x, H, GROUP_SIZE)

    h_selected = torch.randn(dip_k, device="cuda", dtype=torch.float16)
    down_fused = soinfer.ops.gemv_dip_fused_down(cache.down_Wq, cache.down_scale, down_staging_gpu,
                                                  staging_down_scale, descriptors, h_selected, H, GROUP_SIZE)

    # M7 (non-cached) dense reference: quantize the SAME raw weights fresh
    # (independent of the store/cache machinery entirely) and index_select
    # + the plain M4/M7 kernels on the selected rows directly.
    up_Wq_full, up_scale_full, _ = _quantize_and_pack(raw["up_W"])
    down_Wq_full, down_scale_full, _ = _quantize_and_pack(raw["downT_W"])
    up_sel_Wq = up_Wq_full.index_select(0, selected_long.cpu()).cuda()
    up_sel_scale = up_scale_full.index_select(0, selected_long.cpu()).cuda()
    down_sel_Wq = down_Wq_full.index_select(0, selected_long.cpu()).cuda()
    down_sel_scale = down_scale_full.index_select(0, selected_long.cpu()).cuda()

    u_ref = soinfer.ops.gemv_w4a16_group_lop3(up_sel_Wq, up_sel_scale, x, H, GROUP_SIZE)
    down_ref = soinfer.ops.gemv_w4a16_sparse_accumulate(down_sel_Wq, down_sel_scale, h_selected, H, GROUP_SIZE)

    assert torch.equal(u_fused, u_ref)
    assert torch.equal(down_fused, down_ref)


def test_hot_cache_all_selected_channels_hit_needs_no_staging():
    """Degenerate but realistic case: every one of this token's selected
    channels happens to be cache-resident (miss_count == 0) -- the
    staging buffers passed to the fused GEMVs are legitimately empty."""
    model, raw = _build_synthetic_model_with_raw_weights(seed=3)
    I, H = CFG["intermediate_size"], CFG["hidden_size"]
    dip_k = 12

    hot_indices = list(range(I))  # everything cached
    cache = hot_cache.HotCache.build(model, layer_idx=0, hot_indices=hot_indices)

    selected = torch.randperm(I)[:dip_k].int().cuda()
    descriptors, _miss_channels, miss_count = soinfer.ops.build_dip_descriptors(selected, cache.slot_of)
    assert int(miss_count.item()) == 0

    empty_up_Wq = torch.empty(0, cache.up_Wq.shape[1], dtype=torch.uint8, device="cuda")
    empty_up_scale = torch.empty(0, cache.up_scale.shape[1], dtype=torch.float32, device="cuda")
    empty_down_Wq = torch.empty(0, cache.down_Wq.shape[1], dtype=torch.uint8, device="cuda")
    empty_down_scale = torch.empty(0, cache.down_scale.shape[1], dtype=torch.float32, device="cuda")

    x = torch.randn(H, device="cuda", dtype=torch.float16)
    u_fused = soinfer.ops.gemv_dip_fused_up(cache.up_Wq, cache.up_scale, empty_up_Wq, empty_up_scale, descriptors,
                                             x, H, GROUP_SIZE)

    up_Wq_full, up_scale_full, _ = _quantize_and_pack(raw["up_W"])
    up_sel_Wq = up_Wq_full.index_select(0, selected.long().cpu()).cuda()
    up_sel_scale = up_scale_full.index_select(0, selected.long().cpu()).cuda()
    u_ref = soinfer.ops.gemv_w4a16_group_lop3(up_sel_Wq, up_sel_scale, x, H, GROUP_SIZE)

    assert torch.equal(u_fused, u_ref)
