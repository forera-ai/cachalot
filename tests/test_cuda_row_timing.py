"""Row and per-copy timing (Mac only): the f-profiles, level 6, the per-row records and their two gates, and the scoring script on
fabricated runs (record docs/CUDA-ROW-TIMING-RECORD.md)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))
sys.path.insert(0, str(ROOT / "tests"))

import overlap_arena as oa  # noqa: E402
import row_timing as rt  # noqa: E402
import suite  # noqa: E402
from test_cuda_overlap_arena import LAYERS, SLOTS, _gates, synth, validate  # noqa: E402

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def arrays():
    return synth(tokens=10)


def _cell(arrays, arm, level, tokens=8, rows=False):
    arena = oa.OverlapArena(SLOTS, 2048, rehearsal_cpu=True, n_layers=LAYERS)
    arena.standin.set_bytes(1 << 12)
    return oa.run_cell(arrays, SLOTS, LAYERS, arena, arm, level=level, max_decode_tokens=tokens, readback_every=40, row_detail=rows)


def test_the_f_profiles_register_the_row_timing_cells():
    full = [c for g in oa.cell_defs("f-full") for c in oa.cells_of(g)]
    assert len(full) == 24 and {(c["level"], c["c"]) for c in full} == {(3, 1.0), (3, 0.3), (6, 1.0), (6, 0.3), (4, 1.0), (4, 0.3)}
    assert all(c["arm"] in ("N", "D", "P0L") for c in full)
    assert {c["level"] for c in full if c["rows"]} == {3} and not any(c["rows"] for c in full if c["level"] != 3)
    assert len([c for g in oa.cell_defs("f-smoke") for c in oa.cells_of(g)]) == 8


def test_level_6_makes_the_same_decisions_as_level_3_and_records_every_copy(arrays):
    ref = oa.reference_pass_pf(arrays, SLOTS, LAYERS, oa.ARM_MODE["P0L"], 8)
    r3, r6 = _cell(arrays, "P0L", 3), _cell(arrays, "P0L", 6)
    assert r3["book"]["chain2"] == r6["book"]["chain2"] == ref["chain2"]
    assert "rows" not in r3["tokens"][0] and "rows" in r6["tokens"][0]
    for t in r6["tokens"]:
        issued = sum(r[0] + r[1] for r in t["rows"])
        assert len(t["copies"]) == issued
        assert all(len(r) == len(oa.ROW_FIELDS) for r in t["rows"]) and all(len(c) == len(oa.COPY_FIELDS) for c in t["copies"])
    chk = oa.detail_check(r6["tokens"])
    assert chk["rowsum_ok"] and chk["copyevents_ok"]


def test_level_3_with_rows_records_rows_and_no_copies(arrays):
    r = _cell(arrays, "P0L", 3, rows=True)
    assert all(t["rows"] and t["copies"] == [] for t in r["tokens"])
    assert oa.detail_check(r["tokens"])["rowsum_ok"]
    assert "rows" not in _cell(arrays, "N", 3, rows=True)["tokens"][0]               # the control has no book: no rows


def test_the_row_counts_are_the_bookkeepings(arrays):
    r = _cell(arrays, "P0L", 6)
    n_dem = sum(row[0] for t in r["tokens"] for row in t["rows"])
    n_pf = sum(row[1] for t in r["tokens"] for row in t["rows"])
    assert 0 < n_dem <= r["book"]["demand_misses"]                                       # prefill admissions are not decode rows
    assert n_pf <= r["book"]["prefetch_loads"]
    assert any(row[2] > 0 for t in r["tokens"] for row in t["rows"])                     # n_pf_ahead is carried from the previous row


def test_a_tampered_row_trips_the_rowsum_gate_and_a_missing_copy_trips_the_copy_gate(arrays):
    toks = _cell(arrays, "P0L", 6)["tokens"]
    assert oa.detail_check(toks)["copyevents_ok"]
    bad = json.loads(json.dumps(toks))
    bad[2]["rows"][3][oa.ROW_FIELDS.index("start")] += 5.0
    assert not oa.detail_check(bad)["rowsum_ok"]
    bad = json.loads(json.dumps(toks))
    bad[2]["copies"].pop()
    assert not oa.detail_check(bad)["copyevents_ok"]


def test_an_exhausted_copy_event_pool_fails_loudly(arrays, monkeypatch):
    monkeypatch.setattr(oa, "COPY_EVENT_POOL", 3)
    with pytest.raises(RuntimeError, match="pool is exhausted"):
        _cell(arrays, "P0L", 6)


def test_the_f_rehearsal_runs_the_real_path_with_every_gate_including_the_new_two(arrays):
    a = synth(tokens=8)
    gates, failures, raw = _gates(), [], {}
    rows, status = oa.run_group(a, oa.cell_defs("f-rehearsal")[0], None, "r", suite.Deadline(300), gates, failures, raw, rehearsal=True,
                                weight_start=1 << 16, calib_tolerance=50.0, nbytes=2048, readback_every=20, drift_limit=10.0)
    assert status == "complete" and failures == [] and all(gates.values()), (gates, failures)
    assert gates["rowsum"] is True and gates["copyevents"] is True
    validate(rows)
    cell = raw["p17.L6.C0.05.P0L"]
    assert cell["row_fields"] == list(oa.ROW_FIELDS) and cell["rows"][0] and cell["copies"][0] is not None
    assert "rows" in raw["p17.L3.C0.05.D"] and raw["p17.L4.C0.05.D"].get("rows") is None


def test_plan_and_dry_run_list_the_f_profiles(capsys):
    oa.cmd_plan(None)
    out = capsys.readouterr().out
    assert "f-full" in out and "cells=24" in out


def test_the_expected_file_has_the_cells_the_f_profiles_need():
    exp = json.loads((ROOT / "benchmarks" / "cuda" / "n3d-expected.json").read_text())
    for prof in ("f-smoke", "f-full"):
        for g in oa.cell_defs(prof):
            for arm in {a for e in g["plan"] for a in e["arms"] if a in oa.ARM_MODE}:
                assert f"{g['percent']}.{g['tokens']}.{oa.ARM_MODE[arm]}" in exp["cells"]


# ------------------------------------------------------------------------------------------ the scoring script on fabricated runs

def _fab_cell(level, c, arm, tokens=30, layers=10, miss=2, ahead_cost=0.2, per=0.19, with_copies=False, wall=10.0, seed=0):
    rng = np.random.default_rng(seed)
    rows_all, copies_all = [], []
    for _ in range(tokens + 1):
        rows, copies, t, prev_pf = [], [], 0.0, 0
        for i in range(layers):
            n_dem = int(rng.integers(0, miss + 1))
            n_pf = int(rng.integers(0, 4)) if arm == "P0L" else 0
            q = max(0.0, prev_pf * ahead_cost - c) if arm == "P0L" else 0.0
            pre = t
            dem_a = pre + q
            dem_b = dem_a + n_dem * per
            start = max(pre, dem_b)
            mid, end = start + 0.4 * c, start + c
            pf_a, pf_b = (end, end + n_pf * per) if n_pf else (None, None)
            rows.append([n_dem, n_pf, prev_pf, pre, start, mid, end, dem_a, dem_b, pf_a, pf_b])
            if with_copies:
                for k in range(n_dem):
                    copies.append([i, 0, k, round(dem_a + (k + 1) * per, 4), per])
                for k in range(n_pf):
                    copies.append([i, 1, k, round(end + (k + 1) * per, 4), per])
            prev_pf, t = n_pf, end
        rows_all.append(rows)
        copies_all.append(copies)
    cell = {"wall_ms": [wall] * (tokens + 1), "rows": rows_all, "closure": {"share": 0.95}}
    if with_copies:
        cell["copies"] = copies_all
    return cell


def _fab_run(seed=0):
    raw = {}
    for level in (3, 6):
        for c in (1.0, 0.3):
            for arm in ("D", "P0L"):
                raw[rt.cell_id(level, c, arm)] = _fab_cell(level, c, arm, with_copies=(level == 6), seed=seed, wall=10.0 + (0.2 if level == 6 else 0.0))
    for c in (1.0, 0.3):
        for role, w in (("N0", 5.0), ("N1", 5.0), ("D", 12.0), ("P0L", 9.0)):
            raw[rt.cell_id(4, c, role)] = {"wall_ms": [w] * 5}
    return raw


def test_the_scoring_script_recovers_the_fabricated_costs():
    res = rt.score(_fab_run(0), _fab_run(1))
    r12 = res["rt1_rt2"]
    assert r12["median_demand_copy_ms"] == pytest.approx(0.19) and r12["first_minus_later_ms"] == pytest.approx(0.0)
    e = res["rt3_rt4"][0.3]["L3"]
    assert e["available"] and e["p0l"]["q_per_miss"] > e["d"]["q_per_miss"] == pytest.approx(0.0)
    assert 0.0 <= e["q_share_of_excess"] <= 1.05
    assert res["rt6"][1.0]["D"]["excess_ms"] == pytest.approx(0.2)
    x = res["rt7"][0.3]
    assert x["r_run_b"] > 0.95 and x["c_fit_run_a"] == pytest.approx(0.2, abs=0.02)
    assert res["verdict"]["RT-1"] == "in 0.17-0.22" and res["verdict"]["RT-6@0.3.D"] == "held"


def test_the_scoring_cli_reads_two_run_directories(tmp_path, capsys):
    for name, seed in (("a", 0), ("b", 1)):
        d = tmp_path / name
        d.mkdir()
        (d / "raw-samples.json").write_text(json.dumps({"run": name, "samples": _fab_run(seed)}))
    assert rt.main([str(tmp_path / "a"), str(tmp_path / "b"), "--json", str(tmp_path / "o.json")]) == 0
    assert json.loads((tmp_path / "o.json").read_text())["verdict"]["RT-1"]


# ------------------------------------------------------------------------------------------ 0.66.46: retune on a revisited C, floor gate

def test_a_plan_that_revisits_a_c_retunes_the_stand_in_each_time(monkeypatch):
    a = synth(tokens=6)
    tuned = []
    orig = oa.StandIn.calibrate

    def spy(self, target_ms, start, tolerance=0.05):
        tuned.append(target_ms)
        return orig(self, target_ms, start, tolerance=tolerance)

    monkeypatch.setattr(oa.StandIn, "calibrate", spy)
    group = {"id": "p17", "percent": 17, "slots": 40, "tokens": 4, "plan": [
        {"level": 3, "c": 0.05, "arms": ("N", "D", "N")}, {"level": 3, "c": 0.04, "arms": ("N", "D", "N")}, {"level": 6, "c": 0.05, "arms": ("N", "D", "N")}]}
    gates, failures, raw = _gates(), [], {}
    oa.run_group(a, group, None, "r", suite.Deadline(300), gates, failures, raw, rehearsal=True, weight_start=1 << 16, calib_tolerance=50.0,
                 nbytes=2048, readback_every=20, drift_limit=10.0)
    assert tuned == [0.05, 0.04, 0.05]                      # 0.05 again after 0.04: stage A to E never did this


def test_the_floor_gate_trips_when_the_stand_in_is_not_tuned_to_c(monkeypatch):
    a = synth(tokens=6)
    group = {"id": "p17", "percent": 17, "slots": 40, "tokens": 4, "plan": [{"level": 3, "c": 1.0, "arms": ("N", "D", "N")}]}
    gates, failures, raw = _gates(), [], {}
    oa.run_group(a, group, None, "r", suite.Deadline(300), gates, failures, raw, rehearsal=True, weight_start=1 << 16, calib_tolerance=50.0,
                 nbytes=2048, readback_every=20, drift_limit=10.0, floor_check=True)             # the CPU stand-in takes microseconds, not 1 ms
    assert gates["floor"] is False and any("not tuned to C" in f["error"] for f in failures)


def test_the_fix_profile_and_the_merge_replace_only_the_voided_cells():
    cells = [c for g in oa.cell_defs("f-fix") for c in oa.cells_of(g)]
    assert len(cells) == 8 and {(c["level"], c["c"]) for c in cells} == {(6, 1.0), (4, 1.0)}
    raw = {"p17.L6.C1.0.D": {"v": "old"}, "p17.L6.C0.3.D": {"v": "keep"}, "calibration": {"x": 1}}
    fix = {"p17.L6.C1.0.D": {"v": "new"}, "calibration": {"y": 2}}
    m = rt.merge_fix(raw, fix)
    assert m["p17.L6.C1.0.D"]["v"] == "new" and m["p17.L6.C0.3.D"]["v"] == "keep" and m["calibration"] == {"x": 1}
