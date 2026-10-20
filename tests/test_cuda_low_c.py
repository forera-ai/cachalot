"""Low layer time (Mac only; record docs/CUDA-LOW-C-RECORD.md): the profiles, the shapes and the scoring script on planted runs."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import low_c_score as lc  # noqa: E402
import overlap_arena as oa  # noqa: E402
import row_timing as rt  # noqa: E402
import test_cuda_predicted_overlap_fit as planted  # noqa: E402

C = 0.15


def _run(seed: int) -> dict:
    run = planted._run(C, seed)
    for role in ("N0", "N1", "D", "P0L"):
        run[rt.cell_id(4, C, role)] = planted._cell("N" if role[0] == "N" else role, C, np.random.default_rng(7))
    run.update(planted._run(C, seed + 10, level=6))
    return run


def test_the_h_profiles_register_one_layer_time_and_twelve_cells():
    full = [c for g in oa.cell_defs("h-full") for c in oa.cells_of(g)]
    assert len(full) == 12 and {(c["level"], c["c"]) for c in full} == {(3, 0.15), (6, 0.15), (4, 0.15)}
    assert all(c["arm"] in ("N", "D", "P0L") for c in full)
    assert {c["level"] for c in full if c["rows"]} == {3} and not any(c["rows"] for c in full if c["level"] != 3)
    assert len([c for g in oa.cell_defs("h-smoke") for c in oa.cells_of(g)]) == 8


def test_the_expected_file_has_the_cells_the_h_profiles_need():
    exp = json.loads((ROOT / "benchmarks" / "cuda" / "n3d-expected.json").read_text())
    for prof in ("h-smoke", "h-full"):
        for g in oa.cell_defs(prof):
            for arm in {a for e in g["plan"] for a in e["arms"] if a in oa.ARM_MODE}:
                assert f"{g['percent']}.{g['tokens']}.{oa.ARM_MODE[arm]}" in exp["cells"]


def test_plan_lists_the_h_profile(capsys):
    oa.cmd_plan(None)
    out = capsys.readouterr().out
    assert "h-full" in out and "C=[0.15]" in out


def test_the_stage_tag_and_instrument_version_know_the_h_profile():
    text = (ROOT / "benchmarks" / "cuda" / "overlap_arena.py").read_text()
    assert '"h": "n3h"' in text and '"n3h": "N3H-1"' in text and text.count('"h-smoke", "h-full"') >= 2


def test_the_shapes_match_the_records_section_two():
    s = lc.shapes(0.15)
    assert s["affine_in_inverse_c"] == pytest.approx(0.1649, abs=2e-4)
    assert s["power_law"] == pytest.approx(0.1402, abs=3e-4)
    assert s["linear_in_c"] == pytest.approx(0.1100, abs=3e-4)
    assert lc.shapes(0.15, lc.KNOWN_V1)["affine_in_inverse_c"] == pytest.approx(0.1833, abs=2e-4)
    assert lc.extrapolated_v1_slope(0.15) == pytest.approx(0.1833, abs=2e-4)
    for c, v in lc.KNOWN_MEASURED.items():                                    # the affine shape reproduces the known points to 0.001
        assert lc.shapes(c)["affine_in_inverse_c"] == pytest.approx(v, abs=1e-3)


def test_the_four_point_refit_reports_residuals():
    r = lc.refit_with(0.165)
    assert set(r["residuals"]) == {1.0, 0.6, 0.3, 0.15} and r["max_abs_residual"] < 0.002
    assert lc.refit_with(0.10)["max_abs_residual"] > 0.015                    # a slope far off the curve shows up


def test_the_scoring_script_scores_a_planted_pair_of_runs():
    res = lc.score(_run(1), _run(2))
    assert res["level3"]["available"] and res["hidden"]["available"]
    assert abs(res["slope_measured_run_a"] - planted.S_TRUE) < 0.03 and abs(res["slope_measured_run_b"] - planted.S_TRUE) < 0.03
    assert set(res["verdict"]) >= {"L-1", "L-2", "L-3", "L-4", "L-5", "L-6", "L-7", "L-8", "L-10"}
    assert res["verdict"]["L-3"] == "in interval"
    assert res["demand_exposure_ms"]["run_a"] is not None


def test_verdicts_use_only_the_registered_numbers():
    base = {"slope_measured_run_a": 0.16, "slope_measured_run_b": 0.165, "hidden": {"available": True, "H_measured_L4": -0.1, "error_predicted": 0.04},
            "level3": {"overlap_row_r_run_b": 0.95, "overlap_ratio_run_b": 0.98}, "extrapolated": {"error": 0.1}, "demand_exposure_ms": {"run_a": 10.0},
            "gates_run_a": {"calibration": True, "floor": True, "no_failed_cells": True}, "gates_run_b": {}}
    v = lc.verdicts(base)
    assert all(x == "in interval" for x in v.values()) and len(v) == 10
    assert lc.verdicts(dict(base, slope_measured_run_a=0.138))["L-1"] == "outside interval"
    assert lc.verdicts(dict(base, slope_measured_run_a=0.14))["L-2"] == "outside interval"
    flat = lc.verdicts(dict(base, slope_measured_run_a=0.105))
    assert flat["L-1"] == "past falsifier" and flat["L-2"] == "past falsifier"
    assert lc.verdicts(dict(base, slope_measured_run_b=0.25))["L-3"] == "past falsifier"
    assert lc.verdicts(dict(base, hidden={"available": True, "H_measured_L4": 0.2, "error_predicted": 0.2}))["L-4"] == "past falsifier"
    assert lc.verdicts(dict(base, extrapolated={"error": 0.9}))["L-8"] == "past falsifier"
    assert lc.verdicts(dict(base, demand_exposure_ms={"run_a": 13.0}))["L-10"] == "past falsifier"
    bad = lc.verdicts(dict(base, gates_run_a={"calibration": True, "floor": False, "no_failed_cells": True}))
    assert bad["L-9"] == "past falsifier"


def test_gate_summary_counts_failed_cells():
    g = lc.gate_summary({"gates": {"calibration": True, "floor": True, "failed_cells": [{"cell": "x"}]}})
    assert g["calibration"] and g["floor"] and g["no_failed_cells"] is False and "failed_cells" not in g


def test_the_scoring_cli_reads_two_run_directories(tmp_path):
    for name, seed in (("a", 1), ("b", 2)):
        d = tmp_path / name
        d.mkdir()
        (d / "raw-samples.json").write_text(json.dumps({"run": name, "samples": _run(seed)}))
        (d / "run-manifest.json").write_text(json.dumps({"gates": {"calibration": True, "floor": True, "failed_cells": []}}))
    assert lc.main([str(tmp_path / "a"), str(tmp_path / "b"), "--json", str(tmp_path / "o.json")]) == 0
    out = json.loads((tmp_path / "o.json").read_text())
    assert out["verdict"]["L-1"] and out["verdict"]["L-9"] == "in interval"


def test_session_script_has_the_h_steps():
    text = (ROOT / "benchmarks" / "cuda" / "session.sh").read_text()
    assert "n3hsmoke)" in text and "n3hfull)" in text and "h-full" in text and "h-smoke" in text
