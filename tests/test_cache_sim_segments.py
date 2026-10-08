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


def _two_requests_one_decode_segment():
    """Layers 3-4, 4 experts a row. A prefill, then two decode tokens of request A and two of request B with no prefill
    between them (B's prompt was fully reused), so both share one decode segment and their positions repeat."""
    layers, experts, phase, pos = [], [], [], []
    for la in (3, 4):
        layers.append(la), experts.append([0, 1, 2, 3]), phase.append(0), pos.append(0)
    for request in range(2):
        for t in range(2):
            for la in (3, 4):
                layers.append(la), experts.append([100 * (request + 1) + 10 * t + i for i in range(4)]), phase.append(1)
                pos.append(5 + t)
    return {
        "phase": np.array(phase, dtype=np.int8),
        "layer": np.array(layers, dtype=np.int16),
        "position": np.array(pos, dtype=np.int32),
        "experts": np.array(experts, dtype=np.int16),
    }


def test_requests_sharing_a_decode_segment_are_not_merged_by_position():
    out = simulate(_two_requests_one_decode_segment(), [], slots=8, policy="lru", tau=0.0)
    assert out["tokens"] == 4  # grouping rows by position counted 2 and reordered the rows of the two requests


def test_prefetch_counts_loads_used_and_wasted_from_a_layer3_trace():
    arrays = _two_requests_one_decode_segment()
    n_tok = 4
    # each token: layer 3 predicts layer 4's set, one expert the layer then routes and one it does not
    pred_source = np.array([3] * n_tok, dtype=np.int16)
    pred_target = np.array([4] * n_tok, dtype=np.int16)
    pred_experts = np.array([[arrays["experts"][2 + 2 * t + 1][0], 999] for t in range(n_tok)], dtype=np.int16)
    pred_weights = np.array([[0.9, 0.1]] * n_tok, dtype=np.float32)
    arrays.update(pred_source=pred_source, pred_target=pred_target, pred_position=np.zeros(n_tok, np.int32),
                  pred_experts=pred_experts, pred_weights=pred_weights)
    tok_offset = 2  # the prefill rows come first
    for t in range(n_tok):  # the predicted routed expert is the first one of the target layer's row
        assert arrays["layer"][tok_offset + 2 * t + 1] == 4
    off = simulate(arrays, [], slots=8, policy="lru", tau=0.0)
    every = simulate(arrays, [], slots=8, policy="lru", tau=0.0, prefetch_gate=0.0)
    gated = simulate(arrays, [], slots=8, policy="lru", tau=0.0, prefetch_gate=0.5)
    assert every["misses_per_token"] == off["misses_per_token"]  # prefetch changes who waits, not what is missing
    assert every["pred_loads_per_token"] == 2.0 and every["pred_used_per_token"] == 1.0
    assert every["reads_per_token"] == off["misses_per_token"] + 1.0  # one wasted load a token
    assert gated["pred_loads_per_token"] == 1.0 and gated["pred_used_per_token"] == 1.0
    assert gated["demand_per_token"] == off["demand_per_token"] - 1.0


def test_prefetch_needs_predicted_sets():
    with pytest.raises(ValueError):
        simulate(_trace(2), [], slots=400, policy="lru", tau=0.0, prefetch_gate=0.0)
