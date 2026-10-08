import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from cache_sim import simulate  # noqa: E402


def _trace(n_decode):
    layers, experts, phase, pos = [], [], [], []
    for la in range(40):  # one prefill token, then n_decode decode tokens, no segment marks
        layers.append(la), experts.append(list(range(6))), phase.append(0), pos.append(0)
    for t in range(n_decode):
        for la in range(40):
            layers.append(la), experts.append(list(range(6))), phase.append(1), pos.append(1 + t)
    return {
        "phase": np.array(phase, dtype=np.int8),
        "layer": np.array(layers, dtype=np.int16),
        "position": np.array(pos, dtype=np.int32),
        "experts": np.array(experts, dtype=np.int16),
    }


def test_unmarked_trace_is_split_by_phase():
    out = simulate(_trace(5), [], slots=400, policy="lru", tau=0.0)
    assert out["tokens"] == 5
    assert out["hit"] == 1.0


def test_trace_without_decode_tokens_is_refused():
    arrays = _trace(0)
    with pytest.raises(ValueError):
        simulate(arrays, [], slots=400, policy="lru", tau=0.0)


def test_context_floor_follows_ledger_ds_floor_ctx():
    from cache_sim import context_floor

    assert context_floor(79.0, 0) == 79.0
    assert context_floor(79.0, 1800) == pytest.approx(83.0)
    # DS-FLOOR-CTX measured 84.8 ms at 25.0k tokens over a 79.1 ms short probe
    assert context_floor(79.1, 25_000) == pytest.approx(84.8, abs=0.5)


def test_layers_beyond_forty_are_replayed():
    """A MiniMax trace routes layers 3-59: the prefill replay must reach layers >= 40 and the quota follow the layer count."""
    from cache_sim import trace_layers

    layers = list(range(3, 60))
    arrays = {
        "phase": np.array([0] * 57 + [1] * 57, dtype=np.int8),
        "layer": np.array(layers + layers, dtype=np.int16),
        "position": np.array([0] * 57 + [1] * 57, dtype=np.int32),
        "experts": np.array([[1, 2, 3, 4]] * 114, dtype=np.int16),
    }
    assert trace_layers(arrays) == 57
    out = simulate(arrays, [], slots=57 * 4, policy="lru", tau=0.0)
    assert out["tokens"] == 1 and out["hit"] == 1.0  # layers 40-59 would miss if prefill had skipped them
