"""N3 stage C: the router-weight-gated prefetch arms (G00, G10, G20) on the Mac: bookkeeping, gate (g), and the real run path on the CPU."""
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

SLOTS_REH = 40


@pytest.fixture(scope="module")
def arrays():
    a = synth(tokens=10)
    rng = np.random.default_rng(3)
    a["pred_weights"] = rng.dirichlet(np.ones(6), size=len(a["pred_target"])).astype(np.float32)   # varied weights, so gates differ
    return a


def test_mode_parts_and_the_gated_lookup_filters_by_recorded_weight(arrays):
    assert oa.mode_parts("pred") == ("pred", None) and oa.mode_parts("pred@0.20") == ("pred", 0.2) and oa.mode_parts("oracle") == ("oracle", None)
    full, gated = oa.predicted_lookup(arrays), oa.predicted_lookup(arrays, "pred@0.20")
    assert full.keys() == gated.keys() and all(len(gated[k]) <= len(full[k]) for k in full) and any(len(gated[k]) < len(full[k]) for k in full)
    w = {(int(p), int(t)): ws for t, p, ws in zip(arrays["pred_target"], arrays["pred_position"], arrays["pred_weights"], strict=True)}
    for k, exps in gated.items():
        assert exps == [e for e, x in zip(full[k], w[k], strict=True) if x >= 0.2]


def test_g00_is_p0l_by_construction(arrays):
    a = oa.bookkeeping_pass_pf(arrays, SLOTS_REH, 4, "pred", 10)
    b = oa.bookkeeping_pass_pf(arrays, SLOTS_REH, 4, oa.ARM_MODE["G00"], 10)
    assert a["hash"] == b["hash"] and a["chain2"] == b["chain2"] and a["prefetch_loads"] == b["prefetch_loads"]


def test_a_gate_loads_less_and_the_arena_equals_the_independent_reference(arrays):
    loads = []
    for arm in ("G00", "G10", "G20"):
        mode = oa.ARM_MODE[arm]
        mine = oa.bookkeeping_pass_pf(arrays, SLOTS_REH, 4, mode, 10)
        ref = oa.reference_pass_pf(arrays, SLOTS_REH, 4, mode, 10)
        assert mine["hash"] == ref["hash"] and mine["chain2"] == ref["chain2"] and mine["demand_misses"] == ref["demand_misses"]
        loads.append(mine["prefetch_loads"])
    assert loads[0] >= loads[1] >= loads[2] and loads[0] > loads[2]


def test_gate_g_the_gated_counts_equal_the_offline_coverage_model(arrays):
    for arm in ("G10", "G20"):
        mode = oa.ARM_MODE[arm]
        slots = ra.slots_for(17)
        mine = oa.bookkeeping_pass_pf(arrays, slots, 4, mode, 10)
        cov = pcov.coverage(arrays, 17, "pred", max_decode_tokens=10, min_weight=oa.mode_parts(mode)[1])
        assert mine["demand_per_token"] == pytest.approx(cov["demand_misses_per_token"])
        assert mine["loads_per_token"] == pytest.approx(cov["prefetch_loads_per_token"])
        assert mine["covered_per_token"] == pytest.approx(cov["covered_per_token"])


def test_the_c_profiles_register_the_record_cells():
    full = oa.cell_defs("c-full")
    cells = [c for g in full for c in oa.cells_of(g)]
    assert len(cells) == 19 and {g["percent"] for g in full} == {17, 8}
    assert {(c["level"], c["c"]) for c in cells if c["percent"] == 17} == {(5, 1.0), (4, 0.3)}
    assert sum(c["arm"] in oa.GATE_ARMS for c in cells) == 5 and not any(c["arm"] == "G00" for c in cells)


def test_exposed_and_hidden_scores_the_gated_arms():
    eh = oa.exposed_and_hidden({"N0": 40.0, "N1": 40.0, "D": 50.0, "P0L": 46.0, "G20": 45.0, "O": 42.0})
    assert eh["E_G20"] == pytest.approx(5.0) and eh["H_G20"] == pytest.approx(0.5) and eh["H_P0L"] == pytest.approx(0.4)


def _run(arrays, expected=None, group=None):
    gates, failures, raw = _gates(), [], {}
    g = group or oa.cell_defs("c-rehearsal")[0]
    rows, status = oa.run_group(arrays, g, expected, "r", suite.Deadline(300), gates, failures, raw, rehearsal=True, weight_start=1 << 16,
                                calib_tolerance=50.0, nbytes=2048, readback_every=20, drift_limit=10.0)
    return rows, status, gates, failures


def test_the_real_run_path_for_the_gated_arms_with_every_gate_asserted(arrays):
    rows, status, gates, failures = _run(arrays)
    assert status == "complete" and failures == [] and all(gates.values()), (gates, failures)
    validate(rows)
    arms = {r["arm"] for r in rows}
    assert {"n3.p17.L0.C0.05.G00", "n3.p17.L0.C0.05.G20", "n3.p17.L4.C0.05.G10"} <= arms, arms
    fields = {r["field"] for r in rows if r["arm"] == "n3.p17.L0.C0.05"}
    assert {"E_G00", "E_G10", "E_G20", "H_G00", "H_G10", "H_G20", "H_P0L"} <= fields, fields
    g00 = {r["field"]: r["value"] for r in rows if r["arm"] == "n3.p17.L0.C0.05.G00"}
    p0l = {r["field"]: r["value"] for r in rows if r["arm"] == "n3.p17.L0.C0.05.P0L"}
    for f in ("demand_misses_per_token", "wasted_per_token"):
        assert g00[f] == pytest.approx(p0l[f])               # G00 is P0L by construction


def _expected(arrays, tamper=False):
    cells = {}
    for arm in ("D", "P0L", "G00", "G10", "G20", "O"):
        mode = oa.ARM_MODE[arm]
        ref = oa.reference_pass_pf(arrays, SLOTS_REH, 4, mode, 6)
        cell = {"hash": ref["hash"]}
        if oa.mode_parts(mode)[1] is not None:
            cov = pcov.coverage(arrays, 17, "pred", max_decode_tokens=6, min_weight=oa.mode_parts(mode)[1])
            cell.update(coverage_demand_per_token=ref["demand_per_token"] + (1.0 if tamper else 0.0),
                        coverage_loads_per_token=ref["loads_per_token"])
        cells[f"17.6.{mode}"] = cell
    return {"cells": cells}


def test_gate_g_trips_when_a_gated_count_differs_from_the_coverage_model(arrays):
    rows, status, gates, failures = _run(arrays, _expected(arrays))
    assert gates["event_hash"] and gates.get("coverage", True) and failures == []
    rows, status, gates, failures = _run(arrays, _expected(arrays, tamper=True))
    assert gates["coverage"] is False
