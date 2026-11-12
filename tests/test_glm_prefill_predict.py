"""G6b, CACHALOT_GLM_PREFILL_PREDICT_TOPK (docs/glm-speed/G6B-RECORD.md): a GLM prefill chunk below the whole-layer
floor reads ahead the union of its rows' predicted next-layer experts, most-predicted first; off (0) and at or above
the floor nothing changes, a caller's own speculate list (MiniMax) still wins, and the outputs are the same."""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from cachalot.glm import experts
from cachalot.glm.experts import StreamingSwitchGLU

DIM, K = 4, 2


def make_module():
    index = {(layer, e): SimpleNamespace(layer=layer, expert=e) for layer in (3, 4) for e in range(16)}
    calls = []

    class Store:
        def get_many_prefill(self, entries, *, speculate=None):
            calls.append(None if speculate is None else [(s.layer, s.expert) for s in speculate])
            return [SimpleNamespace(slot=None) for _ in entries]

        def release_prefill_layer(self, layer):
            pass

    mod = StreamingSwitchGLU(3, Store(), index, None, None)
    mod._qmm = lambda xe, slot, name: xe * 2.0
    mod._activation = lambda up, gate: up + gate
    asked = []

    def predict(x, k):
        asked.append((x.shape[0], k))
        # rows predict {7, 9}, {9, 1}, {9, 7}: 9 three times, 7 twice, 1 once
        return mx.array([[7, 9], [9, 1], [9, 7]][: x.shape[0]]), mx.ones((x.shape[0], 2))

    mod.predict = predict
    return mod, calls, asked


def run(mod, rows=3, speculate=None):
    x = mx.arange(rows * DIM, dtype=mx.float32).reshape(1, rows, DIM)
    inds = mx.array([[[1, 2], [2, 3], [1, 3]][:rows]])
    return mod(x, inds, speculate=speculate)


@pytest.fixture
def knobs(monkeypatch):
    def set_(topk, floor=128):
        monkeypatch.setattr(experts, "PREFILL_PREDICT_TOPK", topk)
        monkeypatch.setattr(experts, "SPECULATE_MIN_TOKENS", floor)
        monkeypatch.setattr(experts, "PREFILL_SCAN", True)
    return set_


def test_off_reads_nothing_ahead_below_the_floor(knobs):
    knobs(0)
    mod, calls, asked = make_module()
    run(mod)
    assert calls == [None] and asked == []


def test_on_reads_the_predicted_union_most_rows_first(knobs):
    knobs(2)
    mod, calls, asked = make_module()
    run(mod)
    assert asked == [(3, 2)]
    assert calls == [[(4, 9), (4, 7), (4, 1)]]


def test_at_or_above_the_floor_the_whole_next_layer_is_read(knobs):
    knobs(2, floor=3)
    mod, calls, asked = make_module()
    run(mod)
    assert asked == []
    assert calls == [[(4, e) for e in range(16)]]


def test_a_callers_speculate_list_wins(knobs):
    knobs(2)
    mod, calls, asked = make_module()
    given = [SimpleNamespace(layer=4, expert=5)]
    run(mod, speculate=given)
    assert asked == [] and calls == [[(4, 5)]]


def test_no_predictor_no_change(knobs):
    knobs(2)
    mod, calls, asked = make_module()
    mod.predict = None
    run(mod)
    assert calls == [None]


def test_outputs_identical_on_and_off(knobs):
    knobs(0)
    off = np.array(run(make_module()[0]))
    knobs(5)
    on = np.array(run(make_module()[0]))
    assert np.array_equal(off, on)


def test_code_default_is_off():
    src = (experts.__file__ and open(experts.__file__).read())
    assert 'os.environ.get("CACHALOT_GLM_PREFILL_PREDICT_TOPK", "0")' in src
