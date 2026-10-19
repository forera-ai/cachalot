"""N3 design input: the offline prefetch coverage model on tiny hand-made traces (Mac only)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "cuda"))

import prefetch_coverage as pc  # noqa: E402
import replay_arena as ra  # noqa: E402


def trace(rows, preds=()):
    """rows: (layer, position, [6 experts]) decode rows in order; preds: (target, position, [6 experts])."""
    n = len(rows)
    a = {"phase": np.full(n, ra.PHASE_DECODE, np.int8), "layer": np.array([r[0] for r in rows], np.int16),
         "position": np.array([r[1] for r in rows], np.int32), "experts": np.array([r[2] for r in rows], np.int16),
         "pred_target": np.array([p[0] for p in preds], np.int16), "pred_position": np.array([p[1] for p in preds], np.int32),
         "pred_experts": np.array([p[2] for p in preds], np.int16).reshape(len(preds), 6)}
    a["pred_source"] = a["pred_target"] - 1
    return a


def test_without_prefetch_every_first_use_is_a_demand_miss_and_repeats_hit():
    a = trace([(0, 1, [0, 1, 2, 3, 4, 5]), (0, 2, [0, 1, 2, 3, 4, 5])])
    r = pc.coverage(a, 100, "none")
    assert r["tokens"] == 2 and r["demand_misses_per_token"] == 3.0 and r["prefetch_loads_per_token"] == 0.0


def test_a_correct_prediction_covers_the_miss_and_a_wrong_one_is_wasted():
    rows = [(0, 1, [0, 1, 2, 3, 4, 5]), (1, 1, [10, 11, 12, 13, 14, 15])]
    preds = [(1, 1, [10, 11, 12, 20, 21, 22])]                      # three right, three wrong
    r = pc.coverage(trace(rows, preds), 100, "pred")
    assert r["covered_per_token"] == 3.0 and r["prefetch_loads_per_token"] == 6.0 and r["wasted_loads_per_token"] == 3.0
    assert r["demand_misses_per_token"] == 6 + 3                    # layer 0 has no prediction, layer 1 keeps three real misses


def test_the_oracle_covers_everything_and_wastes_nothing():
    rows = [(0, 1, [0, 1, 2, 3, 4, 5]), (1, 1, [10, 11, 12, 13, 14, 15])]
    r = pc.coverage(trace(rows), 100, "pred", oracle=True)
    assert r["demand_misses_per_token"] == 0.0 and r["wasted_loads_per_token"] == 0.0 and r["covered_per_token"] == 12.0


def test_a_weight_gate_loads_only_predicted_experts_at_or_above_the_threshold():
    rows = [(0, 1, [0, 1, 2, 3, 4, 5]), (1, 1, [10, 11, 12, 13, 14, 15])]
    preds = [(1, 1, [10, 11, 12, 20, 21, 22])]                      # three right, three wrong
    a = trace(rows, preds)
    a["pred_weights"] = np.array([[0.4, 0.3, 0.1, 0.1, 0.05, 0.05]], np.float32)
    ungated = pc.coverage(a, 100, "pred")
    assert pc.coverage(a, 100, "pred", min_weight=0.0) == {**ungated, "min_weight": 0.0}
    g = pc.coverage(a, 100, "pred", min_weight=0.3)                  # keeps experts 10 and 11, both right
    assert g["prefetch_loads_per_token"] == 2.0 and g["covered_per_token"] == 2.0 and g["wasted_loads_per_token"] == 0.0
    assert g["demand_misses_per_token"] == 6 + 4
