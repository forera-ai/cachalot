import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from pred_gate_price import replay, token_ms  # noqa: E402


def _trace(pred_weight):
    """One prefill token, then two decode tokens. Layer 1 routes experts 10-15; the prediction from layer 0
    names experts 10-14 (used) and 99 (unused), all with `pred_weight`."""
    phase, layer, experts = [], [], []
    for la in range(40):
        phase.append(0), layer.append(la), experts.append(list(range(6)))
    for _ in range(2):
        for la in range(40):
            phase.append(1), layer.append(la), experts.append(list(range(10, 16)) if la == 1 else list(range(6)))
    pred_experts = np.array([[10, 11, 12, 13, 14, 99]] * 2, dtype=np.int16)
    return {
        "phase": np.array(phase, dtype=np.int8),
        "layer": np.array(layer, dtype=np.int16),
        "position": np.zeros(len(layer), dtype=np.int32),
        "experts": np.array(experts, dtype=np.int16),
        "pred_source": np.zeros(2, dtype=np.int16),
        "pred_target": np.ones(2, dtype=np.int16),
        "pred_position": np.zeros(2, dtype=np.int32),
        "pred_experts": pred_experts,
        "pred_weights": np.full((2, 6), pred_weight, dtype=np.float32),
    }


def test_shipped_gate_reads_every_predicted_load_and_counts_the_used_ones():
    r = replay(_trace(0.3), slots=400, th=0.0, mode="exact")
    # token 1: six predicted loads (five used), one demand miss (expert 15); token 2: only expert 99 is read again
    assert r["tokens"] == 2
    assert r["pred_loads"] == 3.5 and r["pred_used"] == 2.5
    assert r["demand"] == 0.5 and r["reads"] == 4.0


def test_gate_above_every_weight_is_prediction_off_and_budget0_drops_instead():
    off = replay(_trace(0.3), slots=400, th=None, mode="exact")
    gated = replay(_trace(0.3), slots=400, th=0.5, mode="exact")
    assert gated == off and off["demand"] == 3.0 and off["reads"] == 3.0
    b0 = replay(_trace(0.3), slots=400, th=None, mode="budget0")
    assert b0["reads"] == 0 and b0["drops"] == 6.0


def test_token_ms_is_the_larger_of_the_miss_and_byte_terms():
    r = {"misses": 10.0, "reads": 50.0}
    assert token_ms(r, 1.0, 9_950_000) > 490  # bytes: 50 x 9.95 ms
    assert token_ms({"misses": 10.0, "reads": 0.0}, 6.8, 9_950_000) == 78 + 10 * 1.73
