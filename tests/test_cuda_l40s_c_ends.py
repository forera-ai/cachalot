"""L40S layer-time ends (Mac only; record docs/CUDA-L40S-C-ENDS-RECORD.md): the j-profiles, their rehearsal through the real path and the scoring script."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))
sys.path.insert(0, str(ROOT / "tests"))

import l40s_c_ends_score as lc  # noqa: E402
import overlap_arena as oa  # noqa: E402
import row_timing as rt  # noqa: E402
import suite  # noqa: E402
import test_cuda_predicted_overlap_fit as planted  # noqa: E402
from test_cuda_overlap_arena import _gates, synth, validate  # noqa: E402

pytest.importorskip("torch")


def test_the_j_profiles_register_four_layer_times_at_the_full_copy():
    groups = oa.cell_defs("j-full")
    assert [g["id"] for g in groups] == ["p17j"] and "frac" not in groups[0]
    cells = oa.cells_of(groups[0])
    assert len(cells) == 32 and len({c["id"] for c in cells}) == 32
    assert {(c["level"], c["c"]) for c in cells} == {(lv, c) for lv in (3, 4) for c in (1.0, 0.6, 0.3, 0.15)}
    assert all(c["arm"] in ("N", "D", "P0L") for c in cells) and {c["level"] for c in cells if c["rows"]} == {3}
    smoke = [c for g in oa.cell_defs("j-smoke") for c in oa.cells_of(g)]
    assert len(smoke) == 12 and {(c["level"], c["c"]) for c in smoke} == {(3, 1.0), (3, 0.15), (4, 0.15)} and all(c["tokens"] == 40 for c in smoke)


def test_the_expected_file_has_the_cells_the_j_profiles_need():
    exp = json.loads((ROOT / "benchmarks" / "cuda" / "n3d-expected.json").read_text())
    for prof in ("j-smoke", "j-full"):
        for g in oa.cell_defs(prof):
            for arm in {a for e in g["plan"] for a in e["arms"] if a in oa.ARM_MODE}:
                assert f"{g['percent']}.{g['tokens']}.{oa.ARM_MODE[arm]}" in exp["cells"]


def test_plan_lists_the_j_profile_and_the_tags_know_it(capsys):
    oa.cmd_plan(None)
    out = capsys.readouterr().out
    assert "j-full" in out and "p17j" in out and "cells=32" in out
    text = (ROOT / "benchmarks" / "cuda" / "overlap_arena.py").read_text()
    assert '"j": "n3j"' in text and '"n3j": "N3J-1"' in text and text.count('"j-smoke", "j-full"') >= 2


def test_the_j_rehearsal_runs_the_real_path_with_every_gate():
    a = synth(tokens=8)
    gates, failures, raw = _gates(), [], {}
    rows, status = oa.run_group(a, oa.cell_defs("j-rehearsal")[0], None, "r", suite.Deadline(300), gates, failures, raw, rehearsal=True,
                                weight_start=1 << 16, calib_tolerance=50.0, nbytes=8192, readback_every=20, drift_limit=10.0)
    assert status == "complete" and failures == [] and all(gates.values()), (gates, failures)
    validate(rows)
    assert "p17j.L3.C0.05.P0L" in raw and raw["copy_bytes"] == {"p17j": 8192}
    assert any(r["field"] == "source_copy_ms_median" and r["arm"].startswith("n3.p17j.") for r in rows)


# ------------------------------------------------------------------------------------------ the scoring script

def _planted_run(tmp, name, seed, copy_ms):
    rng = np.random.default_rng(seed)
    raw = {}
    for c in lc.LAYER_TIMES:
        for level in (3, 4):
            for role in ("N0", "N1", "D", "P0L"):
                raw[lc.cid(level, c, role)] = planted._cell("N" if role[0] == "N" else role, c, rng)
    d = tmp / name
    d.mkdir()
    (d / "raw-samples.json").write_text(json.dumps({"run": name, "samples": raw}))
    (d / "run-manifest.json").write_text(json.dumps({"gates": {"calibration": True, "floor": True, "failed_cells": []}}))
    (d / "rows.jsonl").write_text(json.dumps({"run": name, "arm": "n3.p17j.L3.C1.0.N0", "field": "source_copy_ms_median", "value": copy_ms}) + "\n")
    return d, raw


def test_the_scoring_script_scores_planted_runs(tmp_path):
    da, ra_ = _planted_run(tmp_path, "a", 1, 0.38)
    db, rb = _planted_run(tmp_path, "b", 2, 0.38)
    res = lc.score(ra_, rb, da, db, json.loads((da / "run-manifest.json").read_text()), json.loads((db / "run-manifest.json").read_text()))
    assert len(res["points_run_a"]) == 4 and all(p["slope"] is not None and p["copy_ms"] == 0.38 for p in res["points_run_a"])
    assert res["ratio_run_a"] is not None and "affine_in_inverse_c_run_b" in res and set(res["g_error_run_b"]) == {"1.0", "0.6", "0.3", "0.15"}
    assert set(res["verdict"]) >= {"J-1", "J-2", "J-3", "J-4", "J-5", "J-6", "J-7", "J-9", "J-10"}
    assert res["verdict"]["J-3"] == "in interval" and res["verdict"]["J-5"] == "in interval"
    assert lc.main([str(da), str(db), "--json", str(tmp_path / "o.json")]) == 0 and json.loads((tmp_path / "o.json").read_text())["verdict"]


def _pt(c, s, h=None, ex=17.0, cm=0.38):
    return {"c_layer": c, "copy_ms": cm, "r": cm / c, "slope": s, "hidden_measured": h, "exposure_ms": ex}


def _clean():
    slopes = {1.0: 0.093, 0.6: 0.093, 0.3: 0.095, 0.15: 0.097}
    hid = {1.0: 0.40, 0.6: 0.30, 0.3: 0.10, 0.15: 0.02}
    pts = [_pt(c, slopes[c], hid[c]) for c in lc.LAYER_TIMES]
    return {"points_run_a": pts, "points_run_b": [dict(p) for p in pts], "gates_run_a": {"calibration": True, "floor": True, "no_failed_cells": True},
            "gates_run_b": {"calibration": True, "floor": True, "no_failed_cells": True}, "ratio_run_a": 0.097 / 0.093, "ratio_run_b": 0.097 / 0.093}


def test_verdicts_use_only_the_registered_numbers():
    v = lc.verdicts(_clean())
    assert all(x == "in interval" for x in v.values()) and set(v) == {f"J-{i}" for i in range(1, 11)}
    res = _clean()
    res["points_run_b"][3]["slope"] = 0.25                                  # the 5090-like rise at 0.15 ms
    assert lc.verdicts(res)["J-2"] == "past falsifier"
    res = _clean()
    res["ratio_run_a"], res["ratio_run_b"] = 1.8, 1.7
    assert lc.verdicts(res)["J-6"] == "past falsifier"
    res["ratio_run_b"] = 1.4
    assert lc.verdicts(res)["J-6"] == "outside interval"
    res = _clean()
    res["points_run_b"][0]["slope"] = 0.14
    assert lc.verdicts(res)["J-1"] == "past falsifier"
    res = _clean()
    res["points_run_b"][2]["slope"] = 0.13                                  # 40 % above the first instance's 0.093
    assert lc.verdicts(res)["J-4"] == "past falsifier"
    res = _clean()
    res["points_run_a"][0]["copy_ms"] = 0.31
    assert lc.verdicts(res)["J-5"] == "outside interval"
    res["points_run_a"][0]["copy_ms"] = 0.6
    assert lc.verdicts(res)["J-5"] == "past falsifier"
    res = _clean()
    for x in (res["points_run_a"], res["points_run_b"]):
        x[3]["slope"] = 0.06
    assert lc.verdicts(res)["J-7"] == "past falsifier"
    res = _clean()
    res["points_run_b"][3]["hidden_measured"] = 0.30
    assert lc.verdicts(res)["J-8"] == "past falsifier"
    res = _clean()
    res["points_run_a"][3]["exposure_ms"] = 25.0
    assert lc.verdicts(res)["J-9"] == "past falsifier"
    res = _clean()
    res["gates_run_b"] = {"calibration": True, "floor": False, "no_failed_cells": True}
    assert lc.verdicts(res)["J-3"] == "past falsifier"
    res = _clean()
    for p in res["points_run_b"][:4]:
        p["slope"] *= 1.5
    assert lc.verdicts(res)["J-10"] == "past falsifier"


def test_session_script_has_the_j_steps():
    text = (ROOT / "benchmarks" / "cuda" / "session.sh").read_text()
    assert "n3jsmoke)" in text and "n3jfull)" in text and "j-full" in text and "j-smoke" in text and "N3J" in text
