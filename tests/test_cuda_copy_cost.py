"""Copy-cost sweep (Mac only; record docs/CUDA-COPY-COST-RECORD.md): the i-profiles, the copy fraction, its gates and the scoring script."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))
sys.path.insert(0, str(ROOT / "tests"))

import copy_cost_score as cc  # noqa: E402
import overlap_arena as oa  # noqa: E402
import row_timing as rt  # noqa: E402
import suite  # noqa: E402
import test_cuda_predicted_overlap_fit as planted  # noqa: E402
from test_cuda_overlap_arena import LAYERS, SLOTS, _gates, synth, validate  # noqa: E402

pytest.importorskip("torch")


def test_the_i_profiles_register_three_fractions_and_two_layer_times():
    groups = oa.cell_defs("i-full")
    assert [g["id"] for g in groups] == ["p17f100", "p17f50", "p17f25"] and [g["frac"] for g in groups] == [1.0, 0.5, 0.25]
    cells = [c for g in groups for c in oa.cells_of(g)]
    assert len(cells) == 48 and all(len(oa.cells_of(g)) == 16 for g in groups)
    assert {(c["level"], c["c"]) for c in cells} == {(3, 0.3), (4, 0.3), (3, 0.6), (4, 0.6)}
    assert all(c["arm"] in ("N", "D", "P0L") for c in cells) and {c["level"] for c in cells if c["rows"]} == {3}
    assert len({c["id"] for c in cells}) == 48
    assert [c for g in oa.cell_defs("i-smoke") for c in oa.cells_of(g)] and len([c for g in oa.cell_defs("i-smoke") for c in oa.cells_of(g)]) == 12


def test_the_expected_file_has_the_cells_the_i_profiles_need():
    exp = json.loads((ROOT / "benchmarks" / "cuda" / "n3d-expected.json").read_text())
    for prof in ("i-smoke", "i-full"):
        for g in oa.cell_defs(prof):
            for arm in {a for e in g["plan"] for a in e["arms"] if a in oa.ARM_MODE}:
                assert f"{g['percent']}.{g['tokens']}.{oa.ARM_MODE[arm]}" in exp["cells"]


def test_plan_lists_the_i_profile_and_the_tags_know_it(capsys):
    oa.cmd_plan(None)
    out = capsys.readouterr().out
    assert "i-full" in out and "p17f25" in out and "cells=16" in out
    text = (ROOT / "benchmarks" / "cuda" / "overlap_arena.py").read_text()
    assert '"i": "n3i"' in text and '"n3i": "N3I-1"' in text and text.count('"i-smoke", "i-full"') >= 2


def _arena(nbytes=2048):
    arena = oa.OverlapArena(SLOTS, nbytes, rehearsal_cpu=True, n_layers=LAYERS)
    arena.standin.set_bytes(1 << 12)
    return arena


def test_the_copy_fraction_rounds_to_4_kib_and_rejects_nonsense():
    arena = _arena(nbytes=64 * 1024)
    assert arena.set_copy_fraction(1.0) == 64 * 1024 and arena.copy_src is arena.sources
    assert arena.set_copy_fraction(0.5) == 32 * 1024 and arena.set_copy_fraction(0.25) == 16 * 1024
    assert all(s.numel() == 16 * 1024 for s in arena.copy_src)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            arena.set_copy_fraction(bad)


def test_a_timed_copy_moves_only_the_prefix_and_the_verify_checks_the_prefix():
    arena = _arena(nbytes=16 * 1024)
    cb = arena.set_copy_fraction(0.25)
    arena.arena.zero_()
    arena.prepare_views(True)
    key = (1, 1)
    arena.enqueue([(3, key)])
    src = arena.copy_src[oa.ra.source_index(key)]
    slot = arena._slot(3)
    assert bool((slot[:cb] == src).all()) and not bool(slot[cb:].any())
    assert arena.verify(3, key)
    assert len(arena.qualify_sources(iters=1)) == oa.ra.SOURCE_RECORDS


def test_a_fraction_below_one_needs_views():
    arena = _arena()
    arena.set_copy_fraction(0.5)
    with pytest.raises(RuntimeError, match="pre-built views"):
        arena.enqueue([(0, (1, 1))])


def test_the_counts_do_not_depend_on_the_copy_fraction():
    a = synth(tokens=8)

    def run(frac):
        arena = _arena()
        arena.set_copy_fraction(frac)
        return oa.run_cell(a, SLOTS, LAYERS, arena, "P0L", level=3, max_decode_tokens=6, readback_every=40)

    r1, r5 = run(1.0), run(0.5)
    assert r5["readback_failures"] == 0 and r1["readback_failures"] == 0
    assert r1["book"]["chain2"] == r5["book"]["chain2"] and r1["book"]["demand_misses"] == r5["book"]["demand_misses"]
    assert r1["issued_copies"] == r5["issued_copies"]


def test_the_i_rehearsal_runs_the_real_path_with_every_gate():
    a = synth(tokens=8)
    gates, failures, raw = _gates(), [], {}
    rows, status = oa.run_group(a, oa.cell_defs("i-rehearsal")[0], None, "r", suite.Deadline(300), gates, failures, raw, rehearsal=True,
                                weight_start=1 << 16, calib_tolerance=50.0, nbytes=8192, readback_every=20, drift_limit=10.0)
    assert status == "complete" and failures == [] and all(gates.values()), (gates, failures)
    validate(rows)
    assert raw["copy_bytes"] == {"p17f50": 4096} and "p17f50.L3.C0.05.P0L" in raw
    assert any(r["field"] == "copy_bytes" and r["value"] == 4096 for r in rows)
    assert any(r["field"] == "source_copy_ms_median" for r in rows)


def test_the_sanity_gate_uses_the_bytes_actually_copied():
    fast = {"arm": "n3.p17f25.L3.C0.3.N0", "field": "source_copy_ms_min", "value": 0.05}      # 2.49 MB in 0.05 ms = 50 GB/s
    assert oa.sanity_ok([fast], {"copy_bytes": {"p17f25": 2_487_500}}) is True
    assert oa.sanity_ok([fast], {}) is False                                                    # the same time for 9.95 MB would be 199 GB/s
    assert oa.sanity_ok([dict(fast, arm="n3.p17.L3.C0.3.N0")], {}) is False


# ------------------------------------------------------------------------------------------ the scoring script

def _planted_run(tmp, name, seed, copy_ms):
    rng = np.random.default_rng(seed)
    raw = {}
    for f in cc.FRACTIONS:
        for c in cc.LAYER_TIMES:
            for level in (3, 4):
                for role in ("N0", "N1", "D", "P0L"):
                    raw[cc.cid(f, level, c, role)] = planted._cell("N" if role[0] == "N" else role, c, rng)
    d = tmp / name
    d.mkdir()
    (d / "raw-samples.json").write_text(json.dumps({"run": name, "samples": raw}))
    (d / "run-manifest.json").write_text(json.dumps({"gates": {"calibration": True, "floor": True, "failed_cells": []}}))
    lines = [json.dumps({"run": name, "arm": f"n3.{cc.gid(f)}.L3.C0.3.N0", "field": "source_copy_ms_median", "value": copy_ms[f]}) for f in cc.FRACTIONS]
    (d / "rows.jsonl").write_text("\n".join(lines) + "\n")
    return d, raw


def test_the_scoring_script_scores_planted_runs(tmp_path):
    cm = {1.0: 0.40, 0.5: 0.22, 0.25: 0.12}
    da, ra_ = _planted_run(tmp_path, "a", 1, cm)
    db, rb = _planted_run(tmp_path, "b", 2, cm)
    res = cc.score(ra_, rb, da, db, json.loads((da / "run-manifest.json").read_text()), json.loads((db / "run-manifest.json").read_text()))
    assert len(res["points_run_a"]) == 6 and all(p["slope"] is not None and p["copy_ms"] for p in res["points_run_a"])
    assert res["line_run_b"]["r2"] == res["line_run_b"]["r2"]                    # a number, not NaN
    assert set(res["verdict"]) >= {"SW-1", "SW-2", "SW-3", "SW-4", "SW-5", "SW-6", "SW-7", "SW-8", "SW-9", "SW-10"}
    assert res["verdict"]["SW-1"] == "in interval" and res["verdict"]["SW-2"] == "in interval" and res["verdict"]["SW-8"] == "in interval"
    assert cc.main([str(da), str(db), "--json", str(tmp_path / "o.json")]) == 0 and json.loads((tmp_path / "o.json").read_text())["verdict"]


def _pt(f, c, copy_ms, s, h=0.3, hm=0.3, ex=None):
    return {"f": f, "c_layer": c, "copy_ms": copy_ms, "r": copy_ms / c, "slope": s, "hidden_measured": h, "hidden_model": hm,
            "exposure_ms": ex if ex is not None else 52.15 * copy_ms, "exposure_predicted_ms": 52.15 * copy_ms}


def _clean():
    pts = [_pt(f, c, cm, cc.G_A + cc.G_B * cm / c) for f, cm in ((1.0, 0.40), (0.5, 0.22), (0.25, 0.12)) for c in cc.LAYER_TIMES]
    return {"points_run_a": pts, "points_run_b": [dict(p) for p in pts], "gates_run_a": {"calibration": True, "floor": True, "no_failed_cells": True},
            "gates_run_b": {}, "line_run_b": cc.line_fit([p["r"] for p in pts], [p["slope"] for p in pts]),
            "g_error_run_b": {f"f{p['f']}.C{p['c_layer']}": 0.0 for p in pts}}


def test_verdicts_use_only_the_registered_numbers():
    v = cc.verdicts(_clean())
    assert all(x == "in interval" for x in v.values()) and len(v) == 10
    res = _clean()
    res["points_run_b"][4]["slope"] = res["points_run_b"][0]["slope"]                     # f = 0.25 at C = 0.3? index 4 = (0.25, 0.3): copy cost no longer matters
    assert cc.verdicts(res)["SW-3"] == "past falsifier"
    res = _clean()
    res["line_run_b"] = {"a": 0.1, "b": 0.0, "r2": 0.3}
    assert cc.verdicts(res)["SW-4"] == "past falsifier"
    res["line_run_b"]["r2"] = 0.7
    assert cc.verdicts(res)["SW-4"] == "outside interval"
    res = _clean()
    res["points_run_a"][0]["copy_ms"] = 0.30
    assert cc.verdicts(res)["SW-1"] == "outside interval"
    res["points_run_a"][0]["copy_ms"] = 0.8
    assert cc.verdicts(res)["SW-1"] == "past falsifier"
    res = _clean()
    res["points_run_a"][1]["exposure_ms"] = 3 * res["points_run_a"][1]["exposure_predicted_ms"]
    assert cc.verdicts(res)["SW-9"] == "past falsifier"
    res = _clean()
    res["gates_run_a"] = {"calibration": True, "floor": False, "no_failed_cells": True}
    assert cc.verdicts(res)["SW-8"] == "past falsifier"
    res = _clean()
    for p in res["points_run_b"][:4]:
        p["slope"] *= 1.5
    assert cc.verdicts(res)["SW-10"] == "past falsifier"                            # four of six moved 50 %: only two within 30 %


def test_session_script_has_the_i_steps():
    text = (ROOT / "benchmarks" / "cuda" / "session.sh").read_text()
    assert "n3ismoke)" in text and "n3ifull)" in text and "i-full" in text and "i-smoke" in text
