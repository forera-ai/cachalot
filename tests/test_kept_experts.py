import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import kept_experts as ke  # noqa: E402


def _rows(extra_reduced, n_tok=40):
    rows = []
    rng = np.random.default_rng(0)
    for t in range(n_tok):
        for layer in range(4):
            kept = 6 if rng.random() < 0.6 else 4
            rows.append({"turn": 0, "position": t, "layer": layer, "kept": kept, "requested": 6, "sync_ms": 1.0 + 0.1 * layer + rng.normal(0, 0.01)})
    # the window after a reduced layer is longer by extra_reduced
    by = {(r["position"], r["layer"]): r for r in rows}
    for r in rows:
        prev = by.get((r["position"], r["layer"] - 1))
        if prev is not None and prev["kept"] < 6:
            r["sync_ms"] += extra_reduced
    return rows


def test_fixed_effect_difference_recovers_the_reduced_layer_penalty():
    rows = ke.with_previous(_rows(0.2))
    assert abs(ke.fixed_effect_difference(rows) - 0.2) < 0.02


def test_split_excess_attributes_the_excess_to_reduced_layers():
    exact = {l: 1.0 + 0.1 * l for l in range(4)}
    rows = ke.with_previous(_rows(0.2))
    s = ke.split_excess(rows, exact)
    assert s["tally_ms_per_token"]["reduced"] > 0.2 and abs(s["tally_ms_per_token"]["full"]) < 0.05


def test_with_previous_marks_the_first_layer():
    rows = ke.with_previous(_rows(0.0, n_tok=2))
    assert all((r["prev_kept"] is None) == (r["layer"] == 0) for r in rows)
