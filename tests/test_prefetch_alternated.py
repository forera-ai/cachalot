import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import prefetch_alternated as pa  # noqa: E402


def _row(cond, pair, base, extra, n=60):
    pos = list(range(20, 20 + n))
    st = [0]
    for k in pos[:-1]:
        st.append(st[-1] + int(1e6 * (base + extra)))
    return {"cond": cond, "role": "pair", "pair": pair, "stamps": st, "positions": pos, "tokens": n, "text": "x",
            "ms_tok": base + extra, "misses_tok": 30.0, "dropped": 0, "summary": []}


def test_pair_gap_is_off_minus_on():
    on, off = _row("on", 0, 140.0, 0.0), _row("off", 0, 140.0, 5.0)
    assert abs(pa.pair_gap(pa._walls(on), pa._walls(off)) - 5.0) < 1e-6


def test_pairs_of_records_order_and_skips_short_pairs():
    rows = [_row("off", 0, 140.0, 5.0), _row("on", 0, 140.0, 0.0), _row("on", 1, 140.0, 0.0, n=10), _row("off", 1, 140.0, 5.0, n=10)]
    ps = pa.pairs_of(rows)
    assert len(ps) == 1 and ps[0]["first"] == "off" and abs(ps[0]["gap"] - 5.0) < 1e-6


def test_bootstrap_interval_brackets_the_mean():
    x = np.array([3.0, 5.0, 4.0, 6.0, 4.5, 5.5])
    lo, hi = pa._boot(x)
    assert lo < x.mean() < hi


def test_cluster_bootstrap_resamples_whole_prompts():
    x = np.array([5.0, 5.2, 3.0, 3.2, 4.0, 4.2])
    cl = np.array([0, 0, 1, 1, 2, 2])
    lo, hi = pa._boot(x, clusters=cl, reps=300)
    assert lo < x.mean() < hi and hi - lo > 0.5
