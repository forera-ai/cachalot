"""CACHALOT_ATTN_ACT_QUANT_O (v270 job A, docs/V270-MAC-JOBS-RECORD.md): off returns the attention output untouched; on applies the reference's
FP8 act_quant round trip in 32-value blocks along the last axis, bit for bit the repo's own round trip, also inside a compiled trace."""

import pytest

mx = pytest.importorskip("mlx.core")

import cachalot.model.fp8_fused_metal as fm  # noqa: E402
from cachalot.model.fp8_act_mlx import fp8_roundtrip_activation_mlx  # noqa: E402

SHAPES = [(8, 4096), (3, 8, 4096), (1, 8, 4096)]          # [groups, group input], [tokens, groups, group input], DSpark's [block, groups, D]


def _x(shape, seed=0):
    mx.random.seed(seed)
    return (mx.random.normal(shape) * 3.0).astype(mx.bfloat16)


def test_off_returns_the_same_array(monkeypatch):
    monkeypatch.setattr(fm, "ATTN_ACT_QUANT_O", False)
    x = _x((8, 4096))
    assert fm.act_quant_o(x) is x


@pytest.mark.parametrize("shape", SHAPES)
def test_on_is_the_blockwise_fp8_round_trip_along_the_last_axis(monkeypatch, shape):
    monkeypatch.setattr(fm, "ATTN_ACT_QUANT_O", True)
    x = _x(shape)
    y = fm.act_quant_o(x)
    ref = fp8_roundtrip_activation_mlx(x.reshape(-1)).reshape(shape)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert bool(mx.array_equal(y, ref))
    assert not bool(mx.array_equal(y, x))                    # it does change the values


def test_on_inside_a_compiled_trace(monkeypatch):
    monkeypatch.setattr(fm, "ATTN_ACT_QUANT_O", True)
    x = _x((8, 4096), seed=1)
    f = mx.compile(lambda a: fm.act_quant_o(a) * 2)
    assert bool(mx.array_equal(f(x), fm.act_quant_o(x) * 2))


def test_the_knob_is_read_from_the_environment(monkeypatch):
    import importlib

    monkeypatch.setenv("CACHALOT_ATTN_ACT_QUANT_O", "1")
    mod = importlib.reload(fm)
    try:
        assert mod.ATTN_ACT_QUANT_O is True
    finally:
        monkeypatch.delenv("CACHALOT_ATTN_ACT_QUANT_O")
        importlib.reload(fm)
    assert fm.ATTN_ACT_QUANT_O is False
