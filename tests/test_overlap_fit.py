import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import overlap_fit as of  # noqa: E402


def _rows(n=400, seed=0, h0=10.0, h1=0.3, k=1.5, w0=90.0):
    rng = np.random.default_rng(seed)
    reads = rng.uniform(5, 100, n)
    x = k * reads + rng.normal(0, 3, n)
    hid = np.clip(h0 + h1 * x + rng.normal(0, 2, n), 0, x)
    blocked = x - hid
    wall = w0 + blocked + rng.normal(0, 3, n)
    return {"reads": reads, "read_busy_ms": x, "hidden_read_ms": hid, "blocked_ms": blocked, "wall_ms": wall,
            "sync_ms": np.full(n, 60.0), "turn": np.repeat(np.arange(n // 10), 10).astype(float)}


def test_affine_overlap_recovers_its_parameters_and_wins_on_blocked_time():
    p = of.fit_overlap(_rows())
    assert abs(p["C"]["h1"] - 0.3) < 0.05 and abs(p["C"]["h0"] - 10.0) < 3.0 and abs(p["k"]["k"] - 1.5) < 0.1
    s = of.score(p, _rows(seed=1))
    assert s["C"]["blocked_rmse"] < s["B"]["blocked_rmse"] < s["A"]["blocked_rmse"]


def test_hidden_never_exceeds_busy_time():
    x = np.array([0.0, 5.0, 50.0, 200.0])
    for form, p in (("A", {"H": 38.0}), ("B", {"f": 0.4}), ("C", {"h0": 12.0, "h1": 0.3})):
        h = of.hidden_of(form, p, x)
        assert np.all(h <= x + 1e-12) and np.all(h >= 0.0)


def test_bootstrap_gain_resamples_whole_turns():
    d = _rows()
    p = of.fit_overlap(d)
    point, lo, hi = of.bootstrap_wall_gain(p, d, n=200)
    assert lo <= point <= hi
