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
