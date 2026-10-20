"""The host-aware row-level window model (Mac only): arm counts equal the verified coverage models; the formula; the fit recovers known parameters."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))
sys.path.insert(0, str(ROOT / "tests"))

import prefetch_coverage as pcov  # noqa: E402
import window_fit as wf  # noqa: E402
from test_cuda_predictor_price import _multi_lead  # noqa: E402
from test_cuda_overlap_arena import synth  # noqa: E402


@pytest.fixture(scope="module")
def arrays():
    a = _multi_lead(synth(tokens=10))
    a["pred_weights"] = np.random.default_rng(5).dirichlet(np.ones(6), size=len(a["pred_target"])).astype(np.float32)
    return a


def test_arm_counts_equal_the_verified_coverage_models(arrays):
    n = 10
    for arm, ref in (("P0L", pcov.coverage_timed(arrays, 17, 1, n)), ("L2", pcov.coverage_timed(arrays, 17, 2, n)),
                     ("G20", pcov.coverage(arrays, 17, "pred", max_decode_tokens=n, min_weight=0.2)),
                     ("D", pcov.coverage(arrays, 17, "none", max_decode_tokens=n)), ("O", pcov.coverage(arrays, 17, "pred", oracle=True, max_decode_tokens=n))):
        rows = wf.arm_rows(arrays, 17, n, arm)
        assert rows[:, 1].sum() / n == pytest.approx(ref["demand_misses_per_token"]), arm
        assert rows[:, 0].sum() / n == pytest.approx(ref.get("prefetch_loads_per_token", 0.0)), arm


def test_the_exposure_formula():
    rows = np.array([[0.0, 0.0], [3.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
    e = wf.exposure(rows, 1, c=0.2, l_dem=0.1, l_pf=0.05, window=0.4)
    pf = max(0.0, 0.05 + 3 * 0.2 - 0.4) + max(0.0, 0.05 + 2 * 0.2 - 0.4)
    dem = (0.1 + 1 * 0.2) + (0.1 + 2 * 0.2)
    assert e == pytest.approx(pf + dem)
    assert wf.exposure(rows, 2, 0.2, 0.1, 0.05, 0.4) == pytest.approx(e / 2)


def test_the_fit_recovers_known_parameters_from_synthetic_measurements(arrays):
    n, true = 10, (0.20, 0.10, 0.05)
    counts = {("old", 17.0, arm): wf.arm_rows(arrays, 17, n, arm) for arm in ("D", "P0L")}
    groups = [{"trace": "old", "percent": 17.0, "C": cc, "level": "L4", "stage": "t", "file": "x",
               "E": {"D": wf.exposure(counts[("old", 17.0, "D")], n, true[0], true[1], 0.0, 0.0),
                     "P0L": wf.exposure(counts[("old", 17.0, "P0L")], n, true[0], true[1], true[2], cc)}} for cc in (0.1, 0.2, 0.4, 0.8)]
    c, l_dem, l_pf = wf.fit(groups, counts, n)
    assert wf.exposure(counts[("old", 17.0, "D")], n, c, l_dem, 0.0, 0.0) == pytest.approx(groups[0]["E"]["D"], abs=0.02)
    # L_pf is only identifiable where a row's prefetch tail is cut by the window (flat otherwise): the fit must reproduce the data
    held = wf.evaluate(groups, counts, n, (c, l_dem, l_pf), ("D", "P0L"))
    assert max(r["error"] for r in held) < 0.02


def test_the_queue_state_terms_and_their_fit():
    rows = np.array([[0.0, 0.0], [3.0, 0.0], [2.0, 1.0], [0.0, 2.0], [1.0, 3.0]])
    base = wf.exposure(rows, 1, 0.2, 0.0, 0.0, 0.4)
    assert wf.exposure(rows, 1, 0.2, 0.0, 0.0, 0.4, 0.1, "T1") == pytest.approx(base + 0.1 * (1 + 3))        # misses in rows that also have prefetch
    assert wf.exposure(rows, 1, 0.2, 0.0, 0.0, 0.4, 0.1, "T2") == pytest.approx(base + 0.1 * 2)             # rows with both kinds
    assert wf.exposure(rows, 1, 0.2, 0.0, 0.0, 0.4, 0.1, "T3") == pytest.approx(base + 0.1 * (1 + 3))       # pairs: 2 in one row, 3 in another
    assert wf.exposure(rows, 1, 0.2, 0.0, 0.0, 0.4, 5.0, None) == pytest.approx(base)


def test_the_queue_state_fit_recovers_a_planted_beta(arrays):
    n, true = 10, (0.20, 0.0, 0.0, 0.03, "T1")
    counts = {("old", 17.0, arm): wf.arm_rows(arrays, 17, n, arm) for arm in ("D", "P0L")}
    groups = [{"trace": "old", "percent": 17.0, "C": cc, "level": "L4", "stage": "t", "file": "x",
               "E": {"D": wf.exposure(counts[("old", 17.0, "D")], n, true[0], 0.0, 0.0, 0.0),
                     "P0L": wf.exposure(counts[("old", 17.0, "P0L")], n, true[0], 0.0, 0.0, cc, true[3], "T1")}} for cc in (0.1, 0.2, 0.4, 0.8)]
    c, l_dem, l_pf = wf.fit(groups, counts, n)
    tp = wf.fit_term(groups, counts, n, "T1", (c, l_dem, l_pf))
    fitted = wf.evaluate(groups, counts, n, tp, ("P0L",))
    assert max(r["error"] for r in fitted) < 0.03 and tp[3] > 0.005             # beta trades off against L_pf: the data must be reproduced, the sign recovered
