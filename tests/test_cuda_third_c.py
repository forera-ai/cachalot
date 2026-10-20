"""Third layer time (Mac only; record docs/CUDA-THIRD-C-RECORD.md): the profiles, the shapes and the scoring script on planted runs."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import overlap_arena as oa  # noqa: E402
import row_timing as rt  # noqa: E402
import test_cuda_predicted_overlap_fit as planted  # noqa: E402
import third_c_score as tc  # noqa: E402

C = 0.6


def _run(seed: int) -> dict:
    run = planted._run(C, seed)                                                         # level 3: D, P0L (+ N0, N1)
    for role in ("N0", "N1", "D", "P0L"):
        run[rt.cell_id(4, C, role)] = planted._cell("N" if role[0] == "N" else role, C, np.random.default_rng(7))
    run.update(planted._run(C, seed + 10, level=6))
    return run


def test_the_g_profiles_register_one_layer_time_and_twelve_cells():
    full = [c for g in oa.cell_defs("g-full") for c in oa.cells_of(g)]
    assert len(full) == 12 and {(c["level"], c["c"]) for c in full} == {(3, 0.6), (6, 0.6), (4, 0.6)}
    assert all(c["arm"] in ("N", "D", "P0L") for c in full)
    assert {c["level"] for c in full if c["rows"]} == {3} and not any(c["rows"] for c in full if c["level"] != 3)
    assert len([c for g in oa.cell_defs("g-smoke") for c in oa.cells_of(g)]) == 8


def test_the_expected_file_has_the_cells_the_g_profiles_need():
    exp = json.loads((ROOT / "benchmarks" / "cuda" / "n3d-expected.json").read_text())
    for prof in ("g-smoke", "g-full"):
        for g in oa.cell_defs(prof):
            for arm in {a for e in g["plan"] for a in e["arms"] if a in oa.ARM_MODE}:
                assert f"{g['percent']}.{g['tokens']}.{oa.ARM_MODE[arm]}" in exp["cells"]


def test_plan_lists_the_g_profile(capsys):
    oa.cmd_plan(None)
    out = capsys.readouterr().out
    assert "g-full" in out and "C=[0.6]" in out


def test_the_shapes_pass_through_the_known_points_and_match_the_record():
    s = tc.shapes(0.6, tc.KNOWN_V1)                                                      # the record's section 2 values (V1 basis)
    assert s["linear_in_c"] == pytest.approx(0.0909, abs=1e-4)
    assert s["affine_in_inverse_c"] == pytest.approx(0.0767, abs=1e-4)
    assert s["power_law"] == pytest.approx(0.0801, abs=2e-4)
    m = tc.shapes(0.6)                                                                   # the erratum's measured-overlap basis
    assert (m["linear_in_c"], m["affine_in_inverse_c"], m["power_law"]) == pytest.approx((0.0873, 0.0753, 0.0786), abs=2e-4)
    for c, v in tc.KNOWN_MEASURED.items():
        assert tc.shapes(c)["linear_in_c"] == pytest.approx(v) and tc.shapes(c)["affine_in_inverse_c"] == pytest.approx(v)
        assert tc.shapes(c)["power_law"] == pytest.approx(v)
    assert tc.interpolated_v1_slope(0.6) == pytest.approx(0.0767, abs=1e-4)
    assert tc.interpolated_v1_slope(1.0) == pytest.approx(0.0624) and tc.interpolated_v1_slope(0.3) == pytest.approx(0.1122)


def test_the_scoring_script_scores_a_planted_pair_of_runs():
    res = tc.score(_run(1), _run(2))
    assert res["level3"]["available"] and res["hidden"]["available"] and res["level6"]["available"]
    assert abs(res["slope_measured_run_a"] - planted.S_TRUE) < 0.03 and abs(res["slope_measured_run_b"] - planted.S_TRUE) < 0.03
    assert res["level3"]["overlap_row_r_run_b"] > 0.9
    assert set(res["verdict"]) >= {"G-1", "G-2", "G-3", "G-4", "G-5", "G-6", "G-7", "G-8", "G-9"}
    assert res["verdict"]["G-3"] == "in interval"                 # a replicate of the same planted stream


def test_verdicts_use_only_the_registered_numbers():
    base = {"slope_measured_run_a": 0.080, "slope_measured_run_b": 0.082,
            "hidden": {"available": True, "H_measured_L4": 0.41, "error_predicted": 0.03},
            "level3": {"overlap_row_r_run_b": 0.95, "overlap_ratio_run_b": 0.98}, "interpolated": {"error": 0.05},
            "level6": {"available": True, "error": 0.1}}
    v = tc.verdicts(base)
    assert all(x == "in interval" for x in v.values()) and len(v) == 9
    hi = dict(base, slope_measured_run_a=0.095 + 0.001)
    assert tc.verdicts(hi)["G-1"] == "outside interval" and tc.verdicts(hi)["G-2"] == "past falsifier"
    lin = dict(base, slope_measured_run_a=0.112)
    assert tc.verdicts(lin)["G-1"] == "past falsifier" and tc.verdicts(lin)["G-2"] == "past falsifier"
    apart = dict(base, slope_measured_run_b=0.110)
    assert tc.verdicts(apart)["G-3"] == "past falsifier"
    worse = dict(base, hidden={"available": True, "H_measured_L4": 0.41, "error_predicted": 0.09}, interpolated={"error": 0.5})
    assert tc.verdicts(worse)["G-7"] == "past falsifier" and tc.verdicts(worse)["G-8"] == "past falsifier"


def test_the_scoring_cli_reads_two_run_directories(tmp_path):
    for name, seed in (("a", 1), ("b", 2)):
        d = tmp_path / name
        d.mkdir()
        (d / "raw-samples.json").write_text(json.dumps({"run": name, "samples": _run(seed)}))
    assert tc.main([str(tmp_path / "a"), str(tmp_path / "b"), "--json", str(tmp_path / "o.json")]) == 0
    assert json.loads((tmp_path / "o.json").read_text())["verdict"]["G-1"]


def test_session_script_has_the_g_steps():
    text = (ROOT / "benchmarks" / "cuda" / "session.sh").read_text()
    assert "n3gsmoke)" in text and "n3gfull)" in text and "g-full" in text and "g-smoke" in text
