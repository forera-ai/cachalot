"""N3 stage B: the levels (digest off, lean bookkeeping, pre-built views, bare events, a CUDA graph) and the P0L arm, on the CPU
rehearsal: identical decisions at every level, every gate asserted through the real run path, the stream order of the new
arm and level on a recording arena. Never a CUDA number."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))
sys.path.insert(0, str(ROOT / "tests"))

import manifest as mf  # noqa: E402
import overlap_arena as oa  # noqa: E402
import replay_arena as ra  # noqa: E402
import suite  # noqa: E402

from test_cuda_overlap_arena import LAYERS, SLOTS, RecordingArena, _gates, synth  # noqa: E402

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def arrays():
    return synth(tokens=14)


def validate(rows):
    assert rows and all(mf.validate_row(r) == [] for r in rows), [mf.validate_row(r) for r in rows if mf.validate_row(r)][:3]


def _cell(arrays, arm, level, tokens=10, arena=None):
    arena = arena or oa.OverlapArena(SLOTS, 2048, rehearsal_cpu=True, n_layers=LAYERS)
    arena.standin.set_bytes(1 << 12)
    return oa.run_cell(arrays, SLOTS, LAYERS, arena, arm, level=level, max_decode_tokens=tokens, readback_every=40)


# ------------------------------------------------------------------------------------ identical decisions at every level

@pytest.mark.parametrize("arm", ["D", "P0", "P0L", "P40", "O"])
def test_every_level_makes_the_same_decisions_and_assigns_the_same_slots(arrays, arm):
    mode = oa.ARM_MODE[arm]
    ref = oa.reference_pass_pf(arrays, SLOTS, LAYERS, mode, 10)
    mine = oa.bookkeeping_pass_pf(arrays, SLOTS, LAYERS, mode, 10)
    assert ref["chain2"] == mine["chain2"] and ref["demand_misses"] == mine["demand_misses"]
    for level in range(0, 5):
        b = _cell(arrays, arm, level)["book"]
        assert b["chain2"] == ref["chain2"], level                      # the slot of every miss and every load is the reference's
        assert (b["demand_misses"], b["prefetch_loads"], b["covered"], b["wasted"]) == (
            ref["demand_misses"], ref["prefetch_loads"], ref["covered"], ref["wasted"]), level


def test_the_digest_is_skipped_at_level_1_and_kept_at_level_0(arrays):
    assert _cell(arrays, "P0", 0)["book"]["hash"] is not None
    b1 = _cell(arrays, "P0", 1)["book"]
    assert b1["hash"] == oa.hashlib.sha256().hexdigest() and b1["chain2"] is not None      # nothing was fed to the hash
    assert _cell(arrays, "P0", 2)["book"]["hash"] is None


def test_the_lean_book_rejects_a_bad_mode_and_counts_tokens(arrays):
    with pytest.raises(ValueError):
        oa.LeanBook(8, 4, "bogus")
    b = _cell(arrays, "D", 2, tokens=7)["book"]
    assert b["tokens"] == 7 and b["prefetch_loads"] == 0 and b["demand_per_token"] > 0


# ------------------------------------------------------------------------------------ the levels' timing records

def test_bare_levels_record_only_the_wall_and_the_host_time(arrays):
    res = _cell(arrays, "O", 4)
    assert len(res["tokens"]) == 10 and all(t.get("bare") and set(t) == {"wall_ms", "host_ms", "tok", "bare"} for t in res["tokens"])
    assert oa.closure(res["tokens"])["bare"] is True and oa.closure(res["tokens"])["ok"]
    rows = oa.cell_rows("r", {"id": "p17.L4.C0.05.O", "role": "O", "level": 4}, res, True, {"error": 0.0, "bytes": 1, "capped": False}, None)
    validate(rows)
    fields = {r["field"] for r in rows}
    assert {"wall_ms_token", "host_issue_ms_token", "bookkeeping_chain2"} <= fields
    assert not ({"compute_busy_ms_token", "compute_stall_ms_token", "host_gap_ms_token"} & fields)
    assert any(r["field"] == "closure_share" and r["tag"] == "unavailable" for r in rows)


def test_instrumented_levels_still_close(arrays):
    for level in (0, 1, 2, 3):
        res = _cell(arrays, "P0L", level)
        assert oa.closure(res["tokens"])["ok"] and "busy_ms" in res["tokens"][0]


# ------------------------------------------------------------------------------------ the stream order of the new arm and level

def _logged(arrays, arm, level):
    arena = RecordingArena(SLOTS, 2048, rehearsal_cpu=True, n_layers=LAYERS)
    arena.standin.set_bytes(1 << 12)
    oa.run_cell(arrays, SLOTS, LAYERS, arena, arm, level=level, max_decode_tokens=4, readback_every=10**9)
    return arena.log


def test_the_late_prefetch_without_a_wait_follows_the_launch_and_never_waits(arrays):
    log = _logged(arrays, "P0L", 0)
    assert not any(e[0] == "copy_wait" for e in log)
    for pos in range(LAYERS - 1):
        end = [i for i, e in enumerate(log) if e[0] == "rec_compute" and e[1:3] == ("end", pos)]
        pf_a = [i for i, e in enumerate(log) if e[0] == "rec_copy" and e[1:3] == ("pf_a", pos)]
        assert end and pf_a
        for en, pf in zip(end, pf_a, strict=False):
            assert en < pf                                              # issued after the layer's compute is launched


def test_the_bare_level_records_only_the_events_correctness_needs(arrays):
    log = _logged(arrays, "P0L", 4)
    kinds = {e[1] for e in log if e[0] in ("rec_compute", "rec_copy")}
    assert kinds == {"dem_b", "end"}
    for pos in range(LAYERS):
        rec = [i for i, e in enumerate(log) if e[0] == "rec_copy" and e[1:3] == ("dem_b", pos)]
        wait = [i for i, e in enumerate(log) if e[0] == "compute_wait" and e[1:3] == ("dem_b", pos)]
        end = [i for i, e in enumerate(log) if e[0] == "rec_compute" and e[1:3] == ("end", pos)]
        for r, w, en in zip(rec, wait, end, strict=False):
            assert r < w < en
    assert [e for e in log if e[0] == "layer"] and not [e for e in log if e[0] == "kernel"]      # one call a layer (no graph on the CPU)


def test_the_early_arms_keep_their_order_at_every_level(arrays):
    for level in (0, 3):
        log = _logged(arrays, "O", level)
        assert not any(e[0] == "copy_wait" for e in log)


def test_prebuilt_views_replace_the_per_copy_view(arrays, monkeypatch):
    arena = oa.OverlapArena(SLOTS, 2048, rehearsal_cpu=True, n_layers=LAYERS)
    arena.standin.set_bytes(1 << 12)
    calls = {"n": 0}
    real = arena._slot

    def counting(slot):
        calls["n"] += 1
        return real(slot)

    monkeypatch.setattr(arena, "_slot", counting)
    r2 = oa.run_cell(arrays, SLOTS, LAYERS, arena, "D", level=2, max_decode_tokens=4, readback_every=10**9)
    without = calls["n"]
    assert arena.views is None and without >= r2["issued_copies"] > 0     # a view per copy (and the untimed prefill populate)
    calls["n"] = 0
    arena.issued_copies = 0
    r3 = oa.run_cell(arrays, SLOTS, LAYERS, arena, "D", level=3, max_decode_tokens=4, readback_every=10**9)
    assert arena.views is not None and len(arena.views) == SLOTS and r3["issued_copies"] == r2["issued_copies"]
    # the timed copies no longer build a view: what remains is one view per slot (made once) and the untimed prefill populate
    assert calls["n"] == SLOTS + without - r3["issued_copies"]
    arena.prepare_views(False)
    assert arena.views is None


def test_the_graph_is_unavailable_on_the_cpu_and_said(arrays):
    arena = oa.OverlapArena(SLOTS, 2048, rehearsal_cpu=True, n_layers=LAYERS)
    arena.standin.set_bytes(1 << 12)
    assert arena.standin.build_graph() is False and "CPU" in arena.standin.graph_error
    res = oa.run_cell(arrays, SLOTS, LAYERS, arena, "O", level=5, max_decode_tokens=3, readback_every=10**9)
    assert len(res["tokens"]) == 3                                      # level 5 without a graph falls back to the eager pair


# ------------------------------------------------------------------------------------ the real run path with the levels

def _rehearse_b(arrays, plan=None, **kw):
    group = oa.cell_defs("b-rehearsal")[0]
    if plan is not None:
        group = {**group, "plan": plan}
    gates, failures, raw = _gates(), [], {}
    gates.setdefault("chain2", True)
    rows, status = oa.run_group(arrays, group, None, "r", suite.Deadline(300), gates, failures, raw, rehearsal=True, weight_start=1 << 16,
                                calib_tolerance=50.0, nbytes=2048, readback_every=30, drift_limit=10.0, **kw)
    return rows, status, gates, failures, raw


def test_the_real_run_path_over_every_level_with_every_gate(arrays):
    rows, status, gates, failures, raw = _rehearse_b(arrays)
    assert status == "complete" and failures == [] and all(gates.values()), (gates, failures)
    validate(rows)
    arms = {r["arm"] for r in rows}
    for level in range(5):
        assert f"n3.p17.L{level}.C0.05.N0" in arms and f"n3.p17.L{level}.C0.05.O" in arms
    assert "n3.p17.L5.C0.05.N0" not in arms                              # no graph on the CPU: the level's cells are skipped and said
    assert "n3.p17.L0.C0.05.P0L" in arms and "n3.p17.L0.C0.05.P40" in arms and "n3.p17.L4.C0.05.P40" not in arms
    fields = {r["field"] for r in rows}
    assert {"E_D", "E_O", "H_P0L", "H_O", "host_issue_ms_token", "wall_ms_token", "bookkeeping_chain2", "graph_unavailable"} <= fields
    assert any(r["field"] == "graph_unavailable" and r["tag"] == "unavailable" for r in rows)       # level 5 on the CPU: said, not failed


def test_a_level_without_a_demand_arm_reports_hidden_share_against_level_0(arrays):
    plan = [{"level": 0, "c": 0.05, "arms": ("N", "D", "O", "N")}, {"level": 2, "c": 0.05, "arms": ("N", "O", "P0L", "N")}]
    rows, status, gates, failures, _raw = _rehearse_b(arrays, plan)
    assert status == "complete" and failures == []
    fields = {(r["arm"], r["field"]) for r in rows}
    assert ("n3.p17.L2.C0.05", "H_O_vs_L0_E_D") in fields and ("n3.p17.L2.C0.05", "H_P0L_vs_L0_E_D") in fields
    assert ("n3.p17.L2.C0.05", "H_O") not in fields


def test_a_chain2_mismatch_trips_the_new_gate(arrays, monkeypatch):
    real = oa.LeanBook.summary

    def wrong(self):
        out = real(self)
        out["chain2"] ^= 1
        return out

    monkeypatch.setattr(oa.LeanBook, "summary", wrong)
    _rows, _s, gates, _f, _raw = _rehearse_b(arrays, [{"level": 2, "c": 0.05, "arms": ("N", "O", "N")}])
    assert gates["chain2"] is False


def test_a_bare_level_does_not_trip_the_closure_gate_and_a_bad_wall_still_trips_it_at_level_0(arrays, monkeypatch):
    _rows, _s, gates, _f, _raw = _rehearse_b(arrays, [{"level": 4, "c": 0.05, "arms": ("N", "D", "N")}])
    assert gates["closure"] is True
    real = oa.run_cell

    def scaled(*a, **k):
        res = real(*a, **k)
        if res["arm"] == "D":
            for t in res["tokens"]:
                t["wall_ms"] *= 3.0
        return res

    monkeypatch.setattr(oa, "run_cell", scaled)
    _rows, _s, gates, _f, _raw = _rehearse_b(arrays, [{"level": 0, "c": 0.05, "arms": ("N", "D", "N")}])
    assert gates["closure"] is False


def test_the_registered_cell_counts():
    smoke, full = oa.cell_defs("b-smoke")[0], oa.cell_defs("b-full")[0]
    assert len(oa.cells_of(smoke)) == 14 and len(oa.cells_of(full)) == 43
    assert {c["level"] for c in oa.cells_of(full)} == set(range(6))
    ids = [c["id"] for c in oa.cells_of(full)]
    assert ids[0] == "p17.L0.C1.0.N0" and "p17.L5.C1.0.O" in ids and "p17.L4.C0.3.P0L" in ids and len(set(ids)) == 43
    assert sum(1 for c in oa.cells_of(full) if c["c"] == 0.3) == 13


def test_plan_and_dry_run_list_the_b_profiles(capsys):
    assert oa.main(["plan"]) == 0 and "b-full" in capsys.readouterr().out
    assert oa.main(["run", "--trace", "x", "--expected", "y", "--cells", "b-smoke"]) == 0


REAL_TRACE = ROOT / "benchmarks" / "results" / "routing-trace-deepseek-long" / "trace.npz"


@pytest.mark.skipif(not REAL_TRACE.exists(), reason="the 11 MB recorded trace is not in the repository (git-ignored results)")
def test_every_level_on_the_recorded_trace_equals_the_reference_chain():
    with np.load(REAL_TRACE) as d:
        full = {k: d[k] for k in d.files if k != "segments"}
    slots = ra.slots_for(17)
    for arm, mode in (("P0L", "pred"), ("O", "oracle"), ("D", "none")):
        ref = oa.reference_pass_pf(full, slots, 40, mode, 24)
        for level in (0, 2, 4):
            arena = oa.OverlapArena(slots, 4096, rehearsal_cpu=True, n_layers=40)
            arena.standin.set_bytes(1 << 14)
            res = oa.run_cell(full, slots, 40, arena, arm, level=level, max_decode_tokens=24, readback_every=10**9)
            assert res["book"]["chain2"] == ref["chain2"] and res["book"]["demand_misses"] == ref["demand_misses"], (arm, level)
