import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import sync_excess as se  # noqa: E402


def _rows(read_scale, n_tokens=30, slope=0.2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tokens):
        for layer in range(3):
            rd = float(rng.uniform(0, read_scale))
            rows.append({"turn": 0, "position": t, "layer": layer, "sync_read_ms": rd, "sync_ms": 1.0 + 0.1 * layer + slope * rd + rng.normal(0, 0.01)})
    return rows


def test_fit_recovers_slope_and_predicts_token_sync():
    data = {"a": _rows(0.5), "b": _rows(2.0, seed=1)}
    f = se.fit(data)
    assert abs(f["slope"] - 0.2) < 0.02
    assert abs(se.predict_token(data["b"], f) - se.per_token(data["b"])["sync"]) < 0.1


def test_position_slope_is_zero_without_position_dependence():
    assert abs(se.position_slope(_rows(1.0, n_tokens=80))) < 0.01
