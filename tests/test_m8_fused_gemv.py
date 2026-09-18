"""M8 task 3 (PROJECT_SPEC.md sec 6, M8): the descriptor-driven cache-or-
stream fused GEMV -- build_dip_descriptors, gemv_dip_fused_up,
gemv_dip_fused_down. Everything here compares against a dense reference
computed from the SAME already-packed bytes just redistributed across two
physical buffers (cache/staging), not two independently-quantized
representations -- unlike M7 task 3's down_proj vs down_proj_T comparison
(see test_m7_sparse_gemv.py), so tolerances here can be as tight as any
plain M4 GEMV test.

NOT YET HARDWARE-VERIFIED as of the commit that adds this file -- no GPU
access this session (see docs/LEARNING_NOTES.md's M8 task 3 entry).
Written and reasoned through carefully, but treat as unverified until a
real `make test` run confirms it, same as gemv_dip_fused.cu itself.

Skipped entirely on machines without a CUDA GPU; run for real on the
Colab/Kaggle T4 session via `make test`.
"""
import torch
import pytest

from soinfer.quant import formats, pack

soinfer = pytest.importorskip("soinfer")
if soinfer.ops is None:
    pytest.skip("soinfer._C (CUDA extension) not built", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="M8 kernels require a CUDA GPU")

# Tighter than M7 task 3's cross-quantization tests (MAX_ABS_ERR=1e-2,
# GEMV_RTOL=5e-3) -- see module docstring for why that's justified here.
MAX_ABS_ERR = 1e-2
GEMV_RTOL = 5e-3


def assert_gemv_matches(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    diff = (actual.float() - expected.float()).abs()
    bound = MAX_ABS_ERR + GEMV_RTOL * expected.float().abs()
    assert torch.all(diff < bound), (
        f"{label} exceeds scaled tolerance: max diff {diff.max().item()} at bound {bound[diff.argmax()].item()}"
    )


def _quantize_pack(W: torch.Tensor, group_size: int):
    qt = formats.quantize(W.float(), formats.QuantConfig(bits=4, granularity="group", group_size=group_size))
    Wq_packed, _ = pack.pack_int4(qt.qweight)
    return Wq_packed.cuda(), qt.scale.cuda(), qt


def _is_staging(desc: int) -> bool:
    return (desc & 0x80000000) != 0


def _offset(desc: int) -> int:
    return desc & 0x7FFFFFFF


def _split_logical_rows(k: int, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic pseudo-random split of [0, k) into a 'cache' subset
    and a 'staging' subset, each in a SHUFFLED (non-logical-order) order
    -- so a test can't accidentally pass just because offsets happen to
    equal logical positions."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(k, generator=g).tolist()
    half = k // 2
    cache_logical = sorted(perm[:half])  # kept ascending for a readable descriptor build below
    staging_logical = perm[half:]  # deliberately NOT sorted -- offsets must not assume order
    return cache_logical, staging_logical


def _build_descriptors(k: int, cache_logical: list[int], staging_logical: list[int]) -> torch.Tensor:
    descriptors = torch.empty(k, dtype=torch.int32)
    for slot, j in enumerate(cache_logical):
        descriptors[j] = slot
    for pos, j in enumerate(staging_logical):
        # pos | (1 << 31) is >= 2**31, out of signed int32 range as a plain
        # Python int; subtract 2**32 to get the same bit pattern as the
        # negative int32 the kernel produces (see gemv_dip_fused.cu).
        descriptors[j] = (pos | (1 << 31)) - (1 << 32)
    return descriptors.cuda()


# ---------------------------------------------------------------------------
# build_dip_descriptors
# ---------------------------------------------------------------------------


def test_build_dip_descriptors_resolves_hits_and_misses():
    I = 32
    slot_of = torch.full((I,), -1, dtype=torch.int32, device="cuda")
    cached_channels = [3, 7, 15]
    for slot, c in enumerate(cached_channels):
        slot_of[c] = slot

    selected = torch.tensor([1, 3, 5, 7, 9, 15], dtype=torch.int32, device="cuda")  # 3 hits, 3 misses
    descriptors, miss_channels, miss_count = soinfer.ops.build_dip_descriptors(selected, slot_of)

    mc = int(miss_count.item())
    assert mc == 3
    assert set(miss_channels[:mc].tolist()) == {1, 5, 9}

    desc = descriptors.tolist()
    assert not _is_staging(desc[1]) and _offset(desc[1]) == 0  # channel 3 -> cache slot 0
    assert not _is_staging(desc[3]) and _offset(desc[3]) == 1  # channel 7 -> cache slot 1
    assert not _is_staging(desc[5]) and _offset(desc[5]) == 2  # channel 15 -> cache slot 2
    for i in (0, 2, 4):
        assert _is_staging(desc[i])
    miss_positions = [_offset(desc[i]) for i in (0, 2, 4)]
    assert sorted(miss_positions) == [0, 1, 2]
    for i, expected_channel in zip((0, 2, 4), (1, 5, 9)):
        assert int(miss_channels[_offset(desc[i])].item()) == expected_channel


def test_build_dip_descriptors_all_hits_no_misses():
    I = 16
    slot_of = torch.arange(I, dtype=torch.int32, device="cuda")  # every channel cached at its own index
    selected = torch.tensor([2, 5, 9], dtype=torch.int32, device="cuda")
    descriptors, _miss_channels, miss_count = soinfer.ops.build_dip_descriptors(selected, slot_of)
    assert int(miss_count.item()) == 0
    for d, expected_slot in zip(descriptors.tolist(), [2, 5, 9]):
        assert not _is_staging(d) and _offset(d) == expected_slot


def test_build_dip_descriptors_all_misses():
    I = 16
    slot_of = torch.full((I,), -1, dtype=torch.int32, device="cuda")
    selected = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device="cuda")
    descriptors, miss_channels, miss_count = soinfer.ops.build_dip_descriptors(selected, slot_of)
    assert int(miss_count.item()) == 4
    assert set(miss_channels[:4].tolist()) == {0, 4, 8, 12}
    assert all(_is_staging(d) for d in descriptors.tolist())


# ---------------------------------------------------------------------------
# gemv_dip_fused_up
# ---------------------------------------------------------------------------


def test_gemv_dip_fused_up_matches_dense_reference_with_mixed_sources():
    torch.manual_seed(0)
    k, H, group_size = 24, 256, 64
    W = torch.randn(k, H)
    Wq, scale, qt = _quantize_pack(W, group_size)
    W_dequant = formats.dequantize(qt).cuda()

    cache_logical, staging_logical = _split_logical_rows(k, seed=1)
    cache_Wq, cache_scale = Wq[cache_logical], scale[cache_logical]
    staging_Wq, staging_scale = Wq[staging_logical], scale[staging_logical]
    descriptors = _build_descriptors(k, cache_logical, staging_logical)

    x = torch.randn(H, device="cuda", dtype=torch.float16)
    expected = torch.mv(W_dequant.float(), x.float()).half()  # logical row order 0..k-1
    actual = soinfer.ops.gemv_dip_fused_up(cache_Wq, cache_scale, staging_Wq, staging_scale, descriptors, x, H,
                                            group_size)
    assert_gemv_matches(actual, expected, "gemv_dip_fused_up_mixed")


def test_gemv_dip_fused_up_all_cache_matches_gemv_w4a16_group_lop3():
    """Degenerate case (empty staging buffer): must reduce to exactly
    M4's plain gemv_w4a16_group_lop3 on the cache buffer alone."""
    torch.manual_seed(2)
    k, H, group_size = 16, 128, 32
    W = torch.randn(k, H)
    Wq, scale, _ = _quantize_pack(W, group_size)
    descriptors = torch.arange(k, dtype=torch.int32, device="cuda")  # all cache, slot == logical index
    empty_Wq = torch.empty(0, Wq.shape[1], dtype=torch.uint8, device="cuda")
    empty_scale = torch.empty(0, scale.shape[1], dtype=torch.float32, device="cuda")

    x = torch.randn(H, device="cuda", dtype=torch.float16)
    expected = soinfer.ops.gemv_w4a16_group_lop3(Wq, scale, x, H, group_size)
    actual = soinfer.ops.gemv_dip_fused_up(Wq, scale, empty_Wq, empty_scale, descriptors, x, H, group_size)
    assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# gemv_dip_fused_down
# ---------------------------------------------------------------------------


def test_gemv_dip_fused_down_matches_dense_reference_with_mixed_sources():
    torch.manual_seed(3)
    k, H, group_size = 24, 256, 64
    W_T = torch.randn(k, H)  # down_proj_T convention (M7 task 3): rows indexed by channel
    Wq, scale, qt = _quantize_pack(W_T, group_size)
    W_dequant = formats.dequantize(qt).cuda()

    cache_logical, staging_logical = _split_logical_rows(k, seed=4)
    cache_Wq, cache_scale = Wq[cache_logical], scale[cache_logical]
    staging_Wq, staging_scale = Wq[staging_logical], scale[staging_logical]
    descriptors = _build_descriptors(k, cache_logical, staging_logical)

    h_selected = torch.randn(k, device="cuda", dtype=torch.float16)  # logical order 0..k-1
    expected = (h_selected.float() @ W_dequant.float()).half()
    actual = soinfer.ops.gemv_dip_fused_down(cache_Wq, cache_scale, staging_Wq, staging_scale, descriptors,
                                              h_selected, H, group_size)
    assert_gemv_matches(actual, expected, "gemv_dip_fused_down_mixed")


def test_gemv_dip_fused_down_all_staging_matches_gemv_w4a16_sparse_accumulate():
    """Degenerate case (empty cache buffer): must reduce to exactly M7
    task 3's plain gemv_w4a16_sparse_accumulate on the staging buffer
    alone."""
    torch.manual_seed(5)
    k, H, group_size = 16, 128, 32
    W_T = torch.randn(k, H)
    Wq, scale, _ = _quantize_pack(W_T, group_size)
    descriptors = (torch.arange(k, dtype=torch.int32, device="cuda")) | (1 << 31)  # all staging
    empty_Wq = torch.empty(0, Wq.shape[1], dtype=torch.uint8, device="cuda")
    empty_scale = torch.empty(0, scale.shape[1], dtype=torch.float32, device="cuda")

    h_selected = torch.randn(k, device="cuda", dtype=torch.float16)
    expected = soinfer.ops.gemv_w4a16_sparse_accumulate(Wq, scale, h_selected, H, group_size)
    actual = soinfer.ops.gemv_dip_fused_down(empty_Wq, empty_scale, Wq, scale, descriptors, h_selected, H,
                                              group_size)
    assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# End-to-end: build_dip_descriptors + both fused GEMVs, tied to a HotCache
# ---------------------------------------------------------------------------


def test_full_pipeline_matches_m7_dense_reference():
    """Ties build_dip_descriptors + both fused GEMVs together against the
    plain M7 (non-cached) path on the SAME logical selected rows, split
    arbitrarily between a 'cache' and a 'staging' buffer -- confirms M8's
    cache-or-stream fusion computes the identical result M7 task 3 would,
    for any hit/miss split, not just the two degenerate all-cache/
    all-staging cases above."""
    torch.manual_seed(6)
    H, group_size, k = 128, 32, 20
    I = 64  # intermediate_size the k selected channels are drawn from

    up_W = torch.randn(I, H)
    downT_W = torch.randn(I, H)
    up_Wq_full, up_scale_full, up_qt = _quantize_pack(up_W, group_size)
    downT_Wq_full, downT_scale_full, downT_qt = _quantize_pack(downT_W, group_size)

    selected = torch.randperm(I)[:k].int().cuda()
    selected_long = selected.long()

    # M7 (non-cached) reference: plain row_select + existing M7 kernels.
    x = torch.randn(H, device="cuda", dtype=torch.float16)
    up_sel_Wq, up_sel_scale = up_Wq_full.index_select(0, selected_long), up_scale_full.index_select(0, selected_long)
    u_ref = soinfer.ops.gemv_w4a16_group_lop3(up_sel_Wq, up_sel_scale, x, H, group_size)
    h_selected = torch.randn(k, device="cuda", dtype=torch.float16)
    downT_sel_Wq = downT_Wq_full.index_select(0, selected_long)
    downT_sel_scale = downT_scale_full.index_select(0, selected_long)
    down_ref = soinfer.ops.gemv_w4a16_sparse_accumulate(downT_sel_Wq, downT_sel_scale, h_selected, H, group_size)

    # M8 (cache-or-stream) path: half the I channels are "cached" (arbitrary, not tied to `selected`).
    cached_channels = torch.randperm(I)[: I // 2].int().cuda()
    slot_of = torch.full((I,), -1, dtype=torch.int32, device="cuda")
    slot_of[cached_channels.long()] = torch.arange(cached_channels.numel(), dtype=torch.int32, device="cuda")
    cache_up_Wq = up_Wq_full.index_select(0, cached_channels.long())
    cache_up_scale = up_scale_full.index_select(0, cached_channels.long())
    cache_down_Wq = downT_Wq_full.index_select(0, cached_channels.long())
    cache_down_scale = downT_scale_full.index_select(0, cached_channels.long())

    descriptors, miss_channels, miss_count = soinfer.ops.build_dip_descriptors(selected, slot_of)
    mc = int(miss_count.item())
    miss_channels_valid = miss_channels[:mc].long()
    staging_up_Wq = up_Wq_full.index_select(0, miss_channels_valid)
    staging_up_scale = up_scale_full.index_select(0, miss_channels_valid)
    staging_down_Wq = downT_Wq_full.index_select(0, miss_channels_valid)
    staging_down_scale = downT_scale_full.index_select(0, miss_channels_valid)

    u_fused = soinfer.ops.gemv_dip_fused_up(cache_up_Wq, cache_up_scale, staging_up_Wq, staging_up_scale,
                                             descriptors, x, H, group_size)
    down_fused = soinfer.ops.gemv_dip_fused_down(cache_down_Wq, cache_down_scale, staging_down_Wq,
                                                  staging_down_scale, descriptors, h_selected, H, group_size)

    assert torch.equal(u_fused, u_ref)
    assert torch.equal(down_fused, down_ref)


# ---------------------------------------------------------------------------
# Error checking
# ---------------------------------------------------------------------------


def test_gemv_dip_fused_up_rejects_mismatched_num_groups():
    torch.manual_seed(7)
    k, H, group_size = 8, 64, 32
    Wq, scale, _ = _quantize_pack(torch.randn(k, H), group_size)
    descriptors = torch.arange(k, dtype=torch.int32, device="cuda")
    x = torch.randn(H, device="cuda", dtype=torch.float16)
    bad_staging_scale = torch.empty(0, scale.shape[1] + 1, dtype=torch.float32, device="cuda")
    bad_staging_Wq = torch.empty(0, Wq.shape[1], dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError):
        soinfer.ops.gemv_dip_fused_up(Wq, scale, bad_staging_Wq, bad_staging_scale, descriptors, x, H, group_size)
