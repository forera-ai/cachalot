"""DI-2 bounded replay (CACHALOT_PREFILL_DECODER_KEEP, docs/v1017-di2-curve/RECORD.md): the knob's range, the window floor in the batched prefill
attention (keys before the replay start are masked, not attended to as the zeroed ring), and the snapshot tag that keeps its prefixes apart."""

import mlx.core as mx
import numpy as np
import pytest

from cachalot.model import attention_prefill_batched as apb
from cachalot.model import snapshot_store
from cachalot.model.prefill_tail_plan import TAIL_ROWS, WINDOW, decoder_keep, tail_start


@pytest.mark.parametrize("raw, expect", [(None, None), ("", None), ("140", 140), (str(WINDOW), WINDOW), (str(TAIL_ROWS), TAIL_ROWS)])
def test_decoder_keep_reads_the_knob(monkeypatch, raw, expect):
    if raw is None:
        monkeypatch.delenv("CACHALOT_PREFILL_DECODER_KEEP", raising=False)
    else:
        monkeypatch.setenv("CACHALOT_PREFILL_DECODER_KEEP", raw)
    assert decoder_keep() == expect


@pytest.mark.parametrize("raw", ["127", str(TAIL_ROWS + 1), "0"])
def test_decoder_keep_refuses_a_replay_outside_the_window_and_the_exact_tail(monkeypatch, raw):
    monkeypatch.setenv("CACHALOT_PREFILL_DECODER_KEEP", raw)
    with pytest.raises(ValueError):
        decoder_keep()


def test_a_shorter_keep_moves_the_replay_start():
    assert tail_start(0, 8192, 140) == (8192 - 140, True)
    assert tail_start(0, 8192) == (8192 - TAIL_ROWS, True)
    assert tail_start(8000, 8100, 140) == (8000, False)       # fewer rows than the replay: every row runs with the real windows


def _attend(monkeypatch, window: mx.array, floor: int, start_pos: int = 20) -> np.ndarray:
    """Small attention_prefill_batched call with plain bf16 projections, so only the window keys and the floor decide the output."""
    monkeypatch.setattr(apb, "fp8_linear_rows", lambda x, w, s: x @ w.T)
    monkeypatch.setattr(apb, "fp8_roundtrip_rows", lambda x: x)
    monkeypatch.setattr(apb, "_rope_rows", lambda x, cos, sin, rope_dim, inverse=False: x)
    monkeypatch.setattr(apb, "act_quant_o", lambda x: x)
    rng = np.random.default_rng(7)
    hidden, heads, dim, w, t, rank = 16, 2, 8, 4, 3, 4
    b = lambda *s: mx.array(rng.normal(scale=0.5, size=s).astype(np.float32)).astype(mx.bfloat16)
    token = apb.WINDOW_FLOOR.set(floor)
    try:
        out, _ = apb.attention_prefill_batched(
            b(t, hidden), start_pos=start_pos, window_cache=window, rope_cos=mx.zeros((64, 2)), rope_sin=mx.zeros((64, 2)),
            attn_sink=mx.zeros((heads,)), q_norm_weight=mx.ones((hidden,)), kv_norm_weight=mx.ones((dim,)),
            wq_a=b(hidden, hidden), wq_a_scales=None, wq_b=b(heads * dim, hidden), wq_b_scales=None, wkv=b(dim, hidden), wkv_scales=None,
            wo_a_bf16=b(rank, heads * dim), wo_b=b(hidden, rank), wo_b_scales=None,
            window_size=w, n_heads=heads, head_dim=dim, rope_head_dim=4, n_groups=1, o_lora_rank=rank)
    finally:
        apb.WINDOW_FLOOR.reset(token)
    return np.array(out.astype(mx.float32))


def test_the_floor_masks_the_ring_instead_of_attending_to_zero_keys(monkeypatch):
    rng = np.random.default_rng(3)
    zeros = mx.zeros((4, 8), dtype=mx.bfloat16)
    real = mx.array(rng.normal(size=(4, 8)).astype(np.float32)).astype(mx.bfloat16)
    # Without a floor the ring's keys take part: a zeroed ring and a real one give different rows (DI-1's exact tail relies only on the end being exact).
    assert not np.allclose(_attend(monkeypatch, zeros, 0), _attend(monkeypatch, real, 0))
    # With the floor at the replay start no ring key is seen, whatever the ring holds, and the zeroed ring no longer dilutes the softmax.
    masked = _attend(monkeypatch, zeros, 20)
    assert np.array_equal(masked, _attend(monkeypatch, real, 20))
    assert not np.allclose(masked, _attend(monkeypatch, zeros, 0))


def test_the_floor_defaults_to_every_real_key():
    assert apb.WINDOW_FLOOR.get() == 0


def test_snapshots_of_a_bounded_replay_get_their_own_key(monkeypatch):
    monkeypatch.delenv("CACHALOT_PREFILL_DECODER_KEEP", raising=False)
    exact = snapshot_store.numerics_version(True)
    monkeypatch.setenv("CACHALOT_PREFILL_DECODER_KEEP", "512")
    assert snapshot_store.numerics_version(True) == exact + "-keep512"
    assert snapshot_store.numerics_version(False) == snapshot_store.NUMERICS_VERSION   # the knob acts only inside an encoder-only prefill
