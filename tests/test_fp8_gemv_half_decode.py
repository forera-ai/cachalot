"""The arithmetic E4M3 decode of the decoded-activation GEMV (CACHALOT_FP8_GEMV_DECODE) must give the same bits as the table
decode for every weight byte, every lane split, and inside `mx.compile` (no `mx.eval` in a trace).

docs/L4F-ATTN-FP8-GEMV-RECORD.md: the half-precision bit trick replaces the 256-entry constant table; loop and reduction order
are unchanged, so the outputs are equal as bit patterns. The NaN code (0x7F/0xFF) is 0.0 in the table and stays 0.0 here.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from cachalot.model import fp8_fused_metal as ffm
from cachalot.model.fp8_fused_metal import fp8_gemv_decoded, quantize_activation_fp8_fused

BLOCK = 32


def _bits(a):
    return a.view(mx.uint32)


def _all_codes_weight(rows: int, cols: int):
    """Every row cycles through all 256 byte values, offset per row so each column position sees every code."""
    idx = (mx.arange(rows)[:, None] + mx.arange(cols)[None, :]) % 256
    w = idx.astype(mx.uint8)
    s = mx.random.randint(120, 131, shape=((rows + BLOCK - 1) // BLOCK, cols // BLOCK)).astype(mx.uint8)
    mx.eval(w, s)
    return w, s


def _act(k: int):
    x = mx.random.normal(shape=(k,)).astype(mx.bfloat16)
    q, qs, qd = quantize_activation_fp8_fused(x)
    mx.eval(q, qs, qd)
    return qd, qs


# K=1280 -> 40 blocks -> 8 lanes per row; K=5120 -> 160 blocks -> 32; K=8192 -> 256 -> 32; K=96 -> 3 blocks -> 32 (fallback);
# K=64 -> 2 blocks -> 32; K=512 -> 16 blocks -> 16 lanes; K=2304 -> 72 blocks -> 8 lanes (shared w2, threadgroup 128)
@pytest.mark.parametrize("rows,k", [(512, 1280), (257, 5120), (320, 8192), (96, 512), (70, 96), (257, 2304)])
def test_half_decode_bit_identical_over_all_codes(rows, k):
    w, s = _all_codes_weight(rows, k)
    qd, qs = _act(k)
    ref = fp8_gemv_decoded(qd, qs, w, s, half_decode=False)
    got = fp8_gemv_decoded(qd, qs, w, s, half_decode=True)
    mx.eval(ref, got)
    assert bool(mx.all(_bits(ref) == _bits(got)).item())


def test_each_code_decodes_the_same():
    """Row i holds code i in every column; a unit activation on one column exposes the decoded weight itself."""
    w = mx.broadcast_to(mx.arange(256).astype(mx.uint8)[:, None], (256, 32))
    w = mx.contiguous(w)
    s = mx.full((8, 1), 127, dtype=mx.uint8)
    act_scales = mx.ones((1,), dtype=mx.float32)
    mx.eval(w, s)
    for col in (0, 7, 31):
        act = (mx.arange(32) == col).astype(mx.float32)
        a = fp8_gemv_decoded(act, act_scales, w, s, half_decode=False)
        b = fp8_gemv_decoded(act, act_scales, w, s, half_decode=True)
        mx.eval(a, b)
        assert bool(mx.all(_bits(a) == _bits(b)).item())
        assert float(a[0x7F].item()) == 0.0 and float(b[0xFF].item()) == 0.0
        assert float(b[0x38].item()) == 1.0  # E4M3 0x38 = 1.0


def test_half_decode_inside_mx_compile():
    rows, k = 128, 1280
    w, s = _all_codes_weight(rows, k)
    qd, qs = _act(k)
    ref = fp8_gemv_decoded(qd, qs, w, s, half_decode=False)
    mx.eval(ref)

    @mx.compile
    def traced(a, a_s, ww, ws):
        return fp8_gemv_decoded(a, a_s, ww, ws, half_decode=True) * 1.0

    got = traced(qd, qs, w, s)
    mx.eval(got)
    assert bool(mx.all(_bits(ref) == _bits(got)).item())


def test_knob_default_on_and_follows_env(monkeypatch):
    assert ffm.GEMV_HALF_DECODE is (ffm.os.environ.get("CACHALOT_FP8_GEMV_DECODE", "1") != "0")
    w, s = _all_codes_weight(64, 1280)
    qd, qs = _act(1280)
    ref = fp8_gemv_decoded(qd, qs, w, s, half_decode=False)
    monkeypatch.setattr(ffm, "GEMV_HALF_DECODE", True)
    got = fp8_gemv_decoded(qd, qs, w, s)
    mx.eval(ref, got)
    assert bool(mx.all(_bits(ref) == _bits(got)).item())
