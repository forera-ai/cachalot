"""N3 stage D: the lead-2 arm `L2` on the Mac: the timed bookkeeping, gate (g), the real run path on the CPU, and that lead 1 did not move."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))
sys.path.insert(0, str(ROOT / "tests"))

import overlap_arena as oa  # noqa: E402
import prefetch_coverage as pcov  # noqa: E402
import replay_arena as ra  # noqa: E402
import suite  # noqa: E402
from test_cuda_overlap_arena import _gates, synth, validate  # noqa: E402
from test_cuda_predictor_price import _multi_lead  # noqa: E402

SLOTS_REH = 40


@pytest.fixture(scope="module")
def arrays():
    return _multi_lead(synth(tokens=10))


def test_mode_lead_and_the_lookup_selects_the_arms_lead(arrays):
    assert oa.mode_lead("pred") == 1 and oa.mode_lead("pred~2") == 2 and oa.mode_parts("pred~2") == ("pred", None)
    l1, l2 = oa.predicted_lookup(arrays, "pred"), oa.predicted_lookup(arrays, "pred~2")
    assert l1 and l2 and all(t >= 2 for _p, t in l2) and set(l2) < set(l1) and any(l1[k] != l2[k] for k in l2)


def test_issue_ahead_needs_the_same_token_all_the_way():
    plan = [("D", 0, 0, [], 0), ("D", 0, 1, [], 0), ("D", 0, 2, [], 0), ("P", 0, []), ("D", 1, 0, [], 1), ("D", 1, 1, [], 1)]
    assert oa.issue_ahead_ok(plan, 0, 2) and not oa.issue_ahead_ok(plan, 1, 2) and not oa.issue_ahead_ok(plan, 4, 2)
    assert not oa.issue_ahead_ok(plan, 5, 2)


def test_the_arena_equals_the_independent_reference_and_the_timed_coverage_model(arrays):
    mode = oa.ARM_MODE["L2"]
    mine = oa.bookkeeping_pass_pf(arrays, SLOTS_REH, 4, mode, 10)
    ref = oa.reference_pass_pf(arrays, SLOTS_REH, 4, mode, 10)
    assert mine["hash"] == ref["hash"] and mine["chain2"] == ref["chain2"] and mine["demand_misses"] == ref["demand_misses"]
    slots = ra.slots_for(17)
    mine17 = oa.bookkeeping_pass_pf(arrays, slots, 4, mode, 10)
    cov = pcov.coverage_timed(arrays, 17, 2, 10)                                           # gate (g)
    assert mine17["demand_per_token"] == pytest.approx(cov["demand_misses_per_token"])
    assert mine17["loads_per_token"] == pytest.approx(cov["prefetch_loads_per_token"])
    assert mine17["covered_per_token"] == pytest.approx(cov["covered_per_token"])


def test_lead_1_is_unchanged_by_the_stage_d_ordering(arrays):
    for pct in (17, 50):
        slots = ra.slots_for(pct)
        mine = oa.bookkeeping_pass_pf(arrays, slots, 4, "pred", 10)
        cov = pcov.coverage(arrays, pct, "pred", max_decode_tokens=10)
        timed = pcov.coverage_timed(arrays, pct, 1, 10)
        assert mine["demand_per_token"] == pytest.approx(cov["demand_misses_per_token"]) == pytest.approx(timed["demand_misses_per_token"])
        assert mine["loads_per_token"] == pytest.approx(cov["prefetch_loads_per_token"]) == pytest.approx(timed["prefetch_loads_per_token"])
        assert oa.reference_pass_pf(arrays, slots, 4, "pred", 10)["hash"] == mine["hash"]


def test_the_d_profiles_register_the_record_cells():
    full = oa.cell_defs("d-full")
    cells = [c for g in full for c in oa.cells_of(g)]
    assert len(cells) == 23 and {g["percent"] for g in full} == {17, 8}
    assert {(c["level"], c["c"]) for c in cells if c["percent"] == 17} == {(5, 1.0), (4, 0.6), (4, 0.3)}
    assert sum(c["arm"] == "L2" for c in cells) == 4 and all(any(x["arm"] == "D" and x["c"] == c["c"] and x["group"] == c["group"] for x in cells) for c in cells)


def _run(arrays, expected=None):
    gates, failures, raw = _gates(), [], {}
    g = oa.cell_defs("d-rehearsal")[0]
    rows, status = oa.run_group(arrays, g, expected, "r", suite.Deadline(300), gates, failures, raw, rehearsal=True, weight_start=1 << 16,
                                calib_tolerance=50.0, nbytes=2048, readback_every=20, drift_limit=10.0)
    return rows, status, gates, failures


def test_the_real_run_path_for_l2_with_every_gate_asserted(arrays):
    rows, status, gates, failures = _run(arrays)
    assert status == "complete" and failures == [] and all(gates.values()), (gates, failures)
    validate(rows)
    arms = {r["arm"] for r in rows}
    assert {"n3.p17.L0.C0.05.L2", "n3.p17.L4.C0.05.L2"} <= arms, arms
    fields = {r["field"] for r in rows if r["arm"] == "n3.p17.L0.C0.05"}
    assert {"E_L2", "H_L2", "H_P0L"} <= fields, fields
    l2 = {r["field"]: r["value"] for r in rows if r["arm"] == "n3.p17.L0.C0.05.L2"}
    ref = oa.bookkeeping_pass_pf(arrays, SLOTS_REH, 4, oa.ARM_MODE["L2"], 6)
    assert l2["demand_misses_per_token"] == pytest.approx(ref["demand_per_token"])
    assert l2["prefetch_loads_per_token"] == pytest.approx(ref["loads_per_token"])


def _expected(arrays, tamper=False):
    cells = {}
    for arm in ("D", "P0L", "L2", "O"):
        mode = oa.ARM_MODE[arm]
        ref = oa.reference_pass_pf(arrays, SLOTS_REH, 4, mode, 6)
        cell = {"hash": ref["hash"]}
        if oa.mode_lead(mode) > 1:
            cell.update(coverage_demand_per_token=ref["demand_per_token"] + (1.0 if tamper else 0.0), coverage_loads_per_token=ref["loads_per_token"])
        cells[f"17.6.{mode}"] = cell
    return {"cells": cells}


def test_gate_g_trips_for_the_lead_2_arm(arrays):
    _rows, _status, gates, failures = _run(arrays, _expected(arrays))
    assert gates["event_hash"] and gates.get("coverage", True) and failures == []
    _rows, _status, gates, _failures = _run(arrays, _expected(arrays, tamper=True))
    assert gates["coverage"] is False
