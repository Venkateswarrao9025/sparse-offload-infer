"""M3 acceptance (PROJECT_SPEC.md sec 6, M3): pack(unpack(q)) round-trips
exactly for every shape including ragged tails, and quantize/dequantize
behaves correctly across all five formats. Pure Python/PyTorch -- no CUDA,
runs on any machine (this is the point of doing quantization work before any
kernel exists).
"""
import pytest
import torch

from soinfer.quant import calibrate, formats, pack

# ---------------------------------------------------------------------------
# pack/unpack: bit-exact round-trip, independent of any float quantization.
# ---------------------------------------------------------------------------

INT4_SHAPES = [
    (1,), (7,), (8,), (9,), (16,), (17,),  # 1-D, including exact and ragged K
    (3, 8), (3, 13), (4, 5, 21),  # multi-dim, ragged tails
]


@pytest.mark.parametrize("shape", INT4_SHAPES)
def test_int4_pack_unpack_roundtrip(shape):
    q = torch.randint(-8, 8, shape, dtype=torch.int8)  # full 4-bit signed range
    packed, orig_k = pack.pack_int4(q)
    assert orig_k == shape[-1]
    recovered = pack.unpack_int4(packed, orig_k)
    assert recovered.shape == q.shape
    assert torch.equal(recovered, q)


def test_int4_pack_shape_and_dtype():
    q = torch.randint(-8, 8, (2, 17), dtype=torch.int8)
    packed, orig_k = pack.pack_int4(q)
    assert orig_k == 17
    assert packed.dtype == torch.uint8
    # 17 -> 3 groups of 8 (24 padded) -> 4 bytes/group -> 12 bytes
    assert packed.shape == (2, 12)


def test_int4_awq_order_byte_layout():
    """Directly check the byte layout documented in csrc/include/layout.h:
    byte0=v0|v2<<4, byte1=v4|v6<<4, byte2=v1|v3<<4, byte3=v5|v7<<4."""
    v = torch.tensor([1, -2, 3, -4, 5, -6, 7, -7], dtype=torch.int8)
    packed, orig_k = pack.pack_int4(v)
    assert orig_k == 8

    vals = v.tolist()  # plain Python ints: arbitrary precision, no int8 overflow on <<4
    nib = lambda x: x & 0xF  # noqa: E731
    expected = torch.tensor(
        [
            nib(vals[0]) | (nib(vals[2]) << 4),
            nib(vals[4]) | (nib(vals[6]) << 4),
            nib(vals[1]) | (nib(vals[3]) << 4),
            nib(vals[5]) | (nib(vals[7]) << 4),
        ],
        dtype=torch.uint8,
    )
    assert torch.equal(packed, expected)


@pytest.mark.parametrize("shape", [(1,), (7,), (8,), (9,), (3, 13)])
def test_int8_pack_unpack_roundtrip(shape):
    q = torch.randint(-127, 128, shape, dtype=torch.int8)
    packed, orig_k = pack.pack_int8(q)
    recovered = pack.unpack_int8(packed, orig_k)
    assert torch.equal(recovered, q)


def test_int4_pack_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        pack.pack_int4(torch.randint(-8, 8, (4,), dtype=torch.int32))


def test_int4_unpack_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        pack.unpack_int4(torch.zeros(4, dtype=torch.int8), orig_k=8)


# ---------------------------------------------------------------------------
# quantize/dequantize: format-level correctness.
# ---------------------------------------------------------------------------

CONFIGS_8BIT = [
    formats.QuantConfig(bits=8, granularity="per_tensor"),
    formats.QuantConfig(bits=8, granularity="per_channel"),
    formats.QuantConfig(bits=8, granularity="group", group_size=128),
    formats.QuantConfig(bits=8, granularity="block32"),
    formats.QuantConfig(bits=8, granularity="mx_e8m0"),
]


@pytest.mark.parametrize("config", CONFIGS_8BIT, ids=lambda c: c.granularity)
def test_quantize_dequantize_shape_and_error(config):
    torch.manual_seed(0)
    w = torch.randn(16, 300)  # 300 is ragged against group sizes 32/128
    qt = formats.quantize(w, config)
    assert qt.qweight.shape[-1] >= 300
    assert qt.orig_k == 300

    recon = formats.dequantize(qt)
    assert recon.shape == w.shape
    # 8-bit symmetric quantization of unit-variance data: error should be
    # small relative to the data itself, not exact (that's the whole point).
    rel_err = (recon - w).abs().mean() / w.abs().mean()
    assert rel_err < 0.05, f"{config.granularity}: unexpectedly large 8-bit relative error {rel_err}"


def test_zero_tensor_quantizes_to_zero():
    w = torch.zeros(8, 64)
    for config in CONFIGS_8BIT:
        qt = formats.quantize(w, config)
        recon = formats.dequantize(qt)
        assert torch.equal(recon, w), config.granularity


def test_quantized_values_within_qmax():
    torch.manual_seed(1)
    w = torch.randn(8, 300) * 10.0
    for bits in (4, 8):
        qmax = formats.qmax_for_bits(bits)
        for config in [
            formats.QuantConfig(bits=bits, granularity="per_tensor"),
            formats.QuantConfig(bits=bits, granularity="per_channel"),
            formats.QuantConfig(bits=bits, granularity="group", group_size=128),
        ]:
            qt = formats.quantize(w, config)
            assert qt.qweight.abs().max().item() <= qmax, config.granularity


def test_mx_e8m0_scale_is_power_of_two():
    torch.manual_seed(2)
    w = torch.randn(4, 300)
    config = formats.QuantConfig(bits=4, granularity="mx_e8m0")
    qt = formats.quantize(w, config)
    nonzero = qt.scale[qt.scale > 0]
    log2_scale = torch.log2(nonzero)
    assert torch.allclose(log2_scale, torch.round(log2_scale), atol=1e-5)


def test_block32_matches_mx_e8m0_grouping_but_not_scale_values():
    """block32 and mx_e8m0 share grouping (block size 32); mx_e8m0's scale is
    block32's scale rounded to a power of two -- Study B's whole point is to
    isolate that one difference."""
    torch.manual_seed(3)
    w = torch.randn(4, 300)
    block32 = formats.quantize(w, formats.QuantConfig(bits=4, granularity="block32"))
    mx = formats.quantize(w, formats.QuantConfig(bits=4, granularity="mx_e8m0"))
    assert block32.scale.shape == mx.scale.shape
    assert torch.equal(mx.scale, formats.round_to_pow2(block32.scale))


def test_per_tensor_int4_collapses_more_than_group128():
    """The known result PROJECT_SPEC.md M3 asks to reproduce: per-tensor
    INT4 forces far more weights to exact zero than group-128 INT4, because
    one outlier sets the scale for the entire tensor instead of just its
    own 128-wide group."""
    torch.manual_seed(4)
    w = torch.randn(64, 4096)
    w[0, 0] *= 50.0  # inject a single outlier, as any real LLM weight matrix has

    per_tensor = formats.quantize(w, formats.QuantConfig(bits=4, granularity="per_tensor"))
    group128 = formats.quantize(w, formats.QuantConfig(bits=4, granularity="group", group_size=128))

    frac_pt = formats.fraction_exact_zero(per_tensor)
    frac_g128 = formats.fraction_exact_zero(group128)
    assert frac_pt > frac_g128, f"per_tensor zero-frac {frac_pt} should exceed group128's {frac_g128}"


# ---------------------------------------------------------------------------
# calibrate.py: pluggable scale_fn strategies.
# ---------------------------------------------------------------------------


def test_mse_optimal_never_worse_than_minmax():
    torch.manual_seed(5)
    w = torch.randn(8, 128) * 3.0
    w[:, 0] *= 20.0  # per-group outlier so minmax and mse-optimal can actually differ
    config = formats.QuantConfig(bits=4, granularity="group", group_size=128)

    minmax_qt = formats.quantize(w, config, scale_fn=calibrate.min_max)
    mse_qt = formats.quantize(w, config, scale_fn=calibrate.mse_optimal)

    mse_minmax = (formats.dequantize(minmax_qt) - w).pow(2).mean().item()
    mse_mse_optimal = (formats.dequantize(mse_qt) - w).pow(2).mean().item()
    assert mse_mse_optimal <= mse_minmax + 1e-6


def test_percentile_scale_is_smaller_or_equal_to_minmax():
    torch.manual_seed(6)
    w = torch.randn(8, 128)
    w[:, 0] *= 50.0
    grouped = w.reshape(8, 1, 128)
    amax = grouped.abs().amax(dim=-1)
    mm = calibrate.min_max(grouped, amax, qmax=7)
    pct = calibrate.percentile(grouped, amax, qmax=7, pct=99.0)
    assert torch.all(pct <= mm + 1e-8)


def test_awq_scale_reparameterization_is_exact():
    torch.manual_seed(7)
    w = torch.randn(16, 64)
    x = torch.randn(64)
    act_abs_mean = x.abs()  # stand-in for a real calibration-pass average
    s = calibrate.awq_scale(w, act_abs_mean, alpha=0.5)
    w_scaled = calibrate.apply_awq_scale(w, s)
    x_scaled = x / s
    assert torch.allclose(w_scaled @ x_scaled, w @ x, atol=1e-4)
