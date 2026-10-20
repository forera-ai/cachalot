"""The bridge session (Mac only): the e-profiles, the row-level window model, and the scoring script on fabricated runs."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))
sys.path.insert(0, str(ROOT / "tests"))

import bridge_score as bs  # noqa: E402
import overlap_arena as oa  # noqa: E402
import prefetch_coverage as pcov  # noqa: E402
import replay_arena as ra  # noqa: E402
import suite  # noqa: E402
import window_model as wm  # noqa: E402
from test_cuda_overlap_arena import _gates, synth, validate  # noqa: E402


def test_the_e_profiles_register_the_bridge_cells():
    full = oa.cell_defs("e-full")
    cells = [c for g in full for c in oa.cells_of(g)]
    assert len(cells) == 12 and {(c["level"], c["c"]) for c in cells} == {(5, 1.0), (4, 0.6), (4, 0.3)}
    assert all(c["arm"] in ("N", "D", "P0L") for c in cells) and len(oa.cells_of(oa.cell_defs("e-smoke")[0])) == 8


def test_the_e_rehearsal_runs_the_real_path_with_every_gate():
    a = synth(tokens=8)
    gates, failures, raw = _gates(), [], {}
    rows, status = oa.run_group(a, oa.cell_defs("e-rehearsal")[0], None, "r", suite.Deadline(300), gates, failures, raw, rehearsal=True,
                                weight_start=1 << 16, calib_tolerance=50.0, nbytes=2048, readback_every=20, drift_limit=10.0)
    assert status == "complete" and failures == [] and all(gates.values()), (gates, failures)
    validate(rows)
    assert any(r["field"] == "H_P0L" for r in rows)


def test_the_window_model_counts_equal_the_coverage_model_and_the_formula():
    a = synth(tokens=10)
    pf = wm.row_counts(a, 17, 10, True)
    none = wm.row_counts(a, 17, 10, False)
    n = len({r[0] for r in pf})
    cov = pcov.coverage_timed(a, 17, 1, 10)
    assert sum(r[3] for r in pf) / n == pytest.approx(cov["demand_misses_per_token"])
    assert sum(r[2] for r in pf) / n == pytest.approx(cov["prefetch_loads_per_token"])
    assert sum(r[3] for r in none) / n == pytest.approx(pcov.coverage(a, 17, "none", max_decode_tokens=10)["demand_misses_per_token"])
    m = wm.model(pf, none, 0.2, 0.3)
    assert m["E_P0L_model"] == pytest.approx(sum(max(0.0, r[2] * 0.2 - 0.3) + r[3] * 0.2 for r in pf) / n)
    huge = wm.model(pf, none, 0.2, 100.0)                       # a huge window hides every prefetch: only demand copies remain
    assert huge["E_P0L_model"] == pytest.approx(sum(r[3] for r in pf) * 0.2 / n) and huge["overloaded_row_share"] == 0.0
    assert wm.model(pf, none, 0.2, 0.0)["H_P0L_model"] <= wm.model(pf, none, 0.2, 1.0)["H_P0L_model"]


def _fake_run(tmp, name, kind, created, h, ed=9.0):
    sha = json.loads((ROOT / "benchmarks" / "cuda" / ("n3c-expected.json" if kind == "old" else "n3d-expected.json")).read_text())["trace_sha256"]
    (tmp / f"{name}-run-manifest.json").write_text(json.dumps({"created_utc": created, "status": "complete", "workload": {"id": "N3-stageE-e-full", "trace_sha256": sha},
                                                              "gates": {"event_hash": True, "failed_cells": []}}))
    rows = [{"arm": arm, "field": f, "value": v} for arm in bs.GROUPS.values() for f, v in (("H_P0L", h), ("E_D", ed))]
    (tmp / f"{name}-rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_the_scoring_script_orders_runs_and_computes_differences(tmp_path):
    for name, kind, created, h in (("a", "old", "2026-10-21T10:00:00Z", 0.10), ("b", "new", "2026-10-21T10:10:00Z", 0.23),
                                    ("c", "new", "2026-10-21T10:20:00Z", 0.25), ("d", "old", "2026-10-21T10:30:00Z", 0.12)):
        _fake_run(tmp_path, name, kind, created, h)
    runs = bs.load_runs(tmp_path)
    t = bs.table(runs)
    assert t["order"] == ["old", "new", "new", "old"]
    c = t["cells"]["old.C0.3"]
    assert c["H_mean"] == pytest.approx(0.11) and c["H_spread"] == pytest.approx(0.02)
    assert t["cells"]["diff.C0.3"] == pytest.approx(0.24 - 0.11)
    assert ra.RECORD_BYTES > 0
