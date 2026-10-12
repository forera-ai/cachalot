import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import token_trace_price_sampled as tp  # noqa: E402


def test_matched_T_recovers_cost_and_cancels_drift():
    rng = np.random.default_rng(0)
    ks = range(30, 130)
    base = {k: 150.0 + rng.normal(0, 1) for k in ks}
    # run a carries +3 ms of drift on every token and +0.6 ms extra on sampled tokens (k % 4 == 0)
    a = {k: base[k] + 3.0 + (0.6 if k % 4 == 0 else 0.0) + rng.normal(0, 0.05) for k in ks}
    t = tp.matched_T(a, base)
    assert abs(t - 0.6) < 0.1


def test_matched_T_needs_enough_positions():
    assert tp.matched_T({k: 1.0 for k in range(5)}, {k: 1.0 for k in range(5)}) is None


def test_estimates_pair_rows_by_role_and_condition():
    def row(cond, role, pair, extra):
        pos = list(range(20, 80))
        st = [0]
        for k in pos[:-1]:
            st.append(st[-1] + int(1e6 * (100.0 + (extra if k % 4 == 0 else 0.0))))
        return {"cond": cond, "role": role, "pair": pair, "stamps": st, "positions": pos}
    rows = [row("e4", "pair", 0, 0.5), row("off", "pair", 0, 0.0)]
    t = tp.estimates(rows, "pair", "e4")
    assert len(t) == 1 and abs(t[0] - 0.5) < 1e-6
