"""N3 stage A harness: bookkeeping against two independent references, every gate tripped on purpose, the stream-ordering of
every arm on a recording arena, and the real run path (`run_group`) on the CPU rehearsal. Never a CUDA number."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))

import manifest as mf  # noqa: E402
import overlap_arena as oa  # noqa: E402
import prefetch_coverage as pc  # noqa: E402
import replay_arena as ra  # noqa: E402
import suite  # noqa: E402

torch = pytest.importorskip("torch")

LAYERS, EXPERTS, SLOTS = 4, 24, 30


def synth(tokens=12, prefill_tokens=3, seed=1, pf_right=3):
    """A small trace: a prefill phase (phase 0) then decode rows, with predicted sets (`pf_right` of 6 right, the rest random)."""
    rng = np.random.default_rng(seed)
    rows = []                                              # (phase, layer, position, experts)
    pred = []                                              # (target, position, experts)
    pos = 0
    for _ in range(prefill_tokens):
        rows += [(0, la, pos, rng.choice(EXPERTS, 6, replace=False)) for la in range(LAYERS)]
        pos += 1
    for _ in range(tokens):
        for la in range(LAYERS):
            real = rng.choice(EXPERTS // 2, 6, replace=False) if la % 2 else rng.choice(EXPERTS, 6, replace=False)
            rows.append((1, la, pos, real))
            if la >= 1:
                pred.append((la, pos, np.concatenate([real[:pf_right], rng.choice(EXPERTS, 6 - pf_right, replace=False)])))
        pos += 1
    tgt = np.array([p[0] for p in pred], np.int16)
    return {"phase": np.array([r[0] for r in rows], np.int8), "layer": np.array([r[1] for r in rows], np.int16),
            "position": np.array([r[2] for r in rows], np.int32), "experts": np.array([r[3] for r in rows], np.int16),
            "pred_target": tgt, "pred_source": tgt - 1, "pred_position": np.array([p[1] for p in pred], np.int32),
            "pred_experts": np.array([p[2] for p in pred], np.int16)}


@pytest.fixture(scope="module")
def arrays():
    return synth()


def validate(rows):
    assert rows and all(mf.validate_row(r) == [] for r in rows), [mf.validate_row(r) for r in rows if mf.validate_row(r)][:3]


# ------------------------------------------------------------------------------------ bookkeeping

@pytest.mark.parametrize("mode", ["none", "pred", "oracle"])
def test_arena_bookkeeping_equals_the_independent_reference(arrays, mode):
    mine = oa.bookkeeping_pass_pf(arrays, SLOTS, LAYERS, mode, None)
    ref = oa.reference_pass_pf(arrays, SLOTS, LAYERS, mode, None)
    assert mine["hash"] == ref["hash"]
    assert (mine["demand_misses"], mine["prefetch_loads"], mine["covered"], mine["wasted"]) == (
        ref["demand_misses"], ref["prefetch_loads"], ref["covered"], ref["wasted"])


@pytest.mark.parametrize("mode", ["none", "pred", "oracle"])
def test_counts_equal_the_offline_coverage_model(arrays, mode, monkeypatch):
    monkeypatch.setattr(ra, "slots_for", lambda percent: SLOTS)
    off = pc.coverage(arrays, 17, "none" if mode == "none" else "pred", oracle=(mode == "oracle"))
    mine = oa.bookkeeping_pass_pf(arrays, SLOTS, LAYERS, mode, None)
    assert mine["demand_per_token"] == pytest.approx(off["demand_misses_per_token"])
    assert mine["loads_per_token"] == pytest.approx(off["prefetch_loads_per_token"])
    assert mine["covered_per_token"] == pytest.approx(off["covered_per_token"])
    assert mine["wasted_per_token"] == pytest.approx(off["wasted_loads_per_token"])


def test_a_prediction_changes_the_counts_and_the_oracle_covers_everything(arrays):
    none = oa.bookkeeping_pass_pf(arrays, SLOTS, LAYERS, "none", None)
    pred = oa.bookkeeping_pass_pf(arrays, SLOTS, LAYERS, "pred", None)
    orc = oa.bookkeeping_pass_pf(arrays, SLOTS, LAYERS, "oracle", None)
    assert pred["demand_misses"] < none["demand_misses"] and pred["prefetch_loads"] > 0 and pred["wasted"] > 0
    assert orc["demand_misses"] == 0 and orc["wasted"] == 0 and orc["covered"] == none["demand_misses"] or orc["demand_misses"] <= 1
    assert none["prefetch_loads"] == 0


def test_arena_pf_prefetch_semantics():
    a = oa.ArenaPF(2, 4)
    assert a.prefetch((0, 1))[0] and a.prefetch((0, 2))[0]
    loaded, slot, ops = a.prefetch((0, 1))
    assert not loaded and ops == ()                          # resident: left alone, no recency refresh
    assert list(a.lru) == [(0, 1), (0, 2)]
    loaded, slot, ops = a.prefetch((0, 3))
    assert loaded and ops[0] == (0, 0, 1) and ops[-1] == (1, 0, 3)     # evicts the LRU end like a demand miss
    a.check_invariants()


def test_pending_keys_are_wasted_when_the_row_ends_or_covered_when_used():
    b = oa.PfBook(8, 4, "pred")
    b.prefetch(0, 1, [3, 4])
    b.demand(0, 1, [3, 9])
    b.end_row(1)
    s = b.summary()
    assert (s["prefetch_loads"], s["covered"], s["wasted"], s["demand_misses"]) == (2, 1, 1, 1)
    with pytest.raises(ValueError):
        oa.PfBook(8, 4, "bogus")


# ------------------------------------------------------------------------------------ stand-in and gates (c), (d), (f)

def test_calibration_gate_and_the_cpu_calibration():
    ok = oa.calibration_result(1.0, 1.04, 100, 0.05)
    bad = oa.calibration_result(1.0, 1.2, 100, 0.05)
    assert ok["ok"] and not bad["ok"] and bad["error"] == pytest.approx(0.2)
    arena = oa.OverlapArena(4, 4096, rehearsal_cpu=True, n_layers=LAYERS)
    r = arena.standin.calibrate(0.05, 1 << 16, cap_bytes=1 << 24, tolerance=5.0)
    assert r["ok"] and arena.standin.nbytes() > 0
    capped = arena.standin.calibrate(1e6, 1 << 12, cap_bytes=1 << 14, tolerance=0.05)
    assert not capped["ok"] and capped["capped"]


def test_exposed_and_hidden_and_drift():
    eh = oa.exposed_and_hidden({"N0": 10.0, "N1": 10.2, "D": 20.0, "P0": 15.0, "O": 11.0})
    assert eh["floor_ms"] == pytest.approx(10.1) and eh["drift"] == pytest.approx(0.02)
    assert eh["E_D"] == pytest.approx(9.9) and eh["H_P0"] == pytest.approx(1 - 4.9 / 9.9) and eh["H_O"] == pytest.approx(1 - 0.9 / 9.9)
    assert oa.exposed_and_hidden({"D": 1.0}) == {}


def test_closure_counts_the_tokens_within_tolerance():
    toks = [{"wall_ms": 10.0, "span_ms": 10.0}] + [{"wall_ms": 10.0, "span_ms": 9.8}] * 9 + [{"wall_ms": 10.0, "span_ms": 7.0}]
    cl = oa.closure(toks)
    assert cl["n"] == 10 and cl["share"] == pytest.approx(0.9) and cl["ok"] and cl["meets_prediction"]
    bad = oa.closure([{"wall_ms": 10.0, "span_ms": 10.0}] + [{"wall_ms": 10.0, "span_ms": 5.0}] * 5)
    assert not bad["ok"]


# ------------------------------------------------------------------------------------ the stream order of every arm

class RecordingArena(oa.OverlapArena):
    """A rehearsal arena that logs the stream-level calls in order, to test what the CPU cannot time: that the compute stream waits
    for the batch event after it is recorded, that the late prefetch waits for the mid event after it is recorded, and so on."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log = []
        self.names = {}
        for par, pool in enumerate(self.pool):
            for kind, evs in pool.items():
                for i, e in enumerate(evs):
                    self.names[id(e)] = (kind, i, par)

    def _n(self, ev):
        return self.names.get(id(ev), ("other", 0, 0))

    def rec_compute(self, ev):
        self.log.append(("rec_compute", *self._n(ev)))
        super().rec_compute(ev)

    def rec_copy(self, ev):
        self.log.append(("rec_copy", *self._n(ev)))
        super().rec_copy(ev)

    def compute_wait(self, ev):
        self.log.append(("compute_wait", *self._n(ev)))

    def copy_wait(self, ev):
        self.log.append(("copy_wait", *self._n(ev)))

    def enqueue(self, loads):
        self.log.append(("enqueue", len(loads)))
        super().enqueue(loads)

    def compute_part(self, which):
        self.log.append(("kernel", which))
        super().compute_part(which)

    def compute_layer(self):
        self.log.append(("layer",))
        super().compute_layer()


def _run_logged(arrays, arm):
    arena = RecordingArena(SLOTS, 2048, rehearsal_cpu=True, n_layers=LAYERS)
    arena.standin.set_bytes(1 << 12)
    res = oa.run_cell(arrays, SLOTS, LAYERS, arena, arm, max_decode_tokens=4, readback_every=10**9)
    return arena, res


@pytest.mark.parametrize("arm", ["D", "P0", "P40", "O"])
def test_the_compute_stream_waits_for_each_batch_event_after_it_is_recorded(arrays, arm):
    arena, _ = _run_logged(arrays, arm)
    log = arena.log
    for pos in range(LAYERS):
        rec = [i for i, e in enumerate(log) if e[0] == "rec_copy" and e[1:3] == ("dem_b", pos)]
        wait = [i for i, e in enumerate(log) if e[0] == "compute_wait" and e[1:3] == ("dem_b", pos)]
        start = [i for i, e in enumerate(log) if e[0] == "rec_compute" and e[1:3] == ("start", pos)]
        assert rec and wait and start
        for r, w, st in zip(rec, wait, start, strict=False):
            assert r < w < st                                   # recorded, then waited on, then the layer starts


def test_the_late_prefetch_waits_for_the_mid_event_after_it_is_recorded(arrays):
    arena, res = _run_logged(arrays, "P40")
    log = arena.log
    waits = [i for i, e in enumerate(log) if e[0] == "copy_wait"]
    assert waits
    for w in waits:
        _, kind, pos, par = log[w]
        assert kind == "mid"
        rec = max(i for i, e in enumerate(log[:w]) if e[0] == "rec_compute" and e[1:4] == ("mid", pos, par))
        nxt_copy = next(i for i, e in enumerate(log[w:], w) if e[0] == "enqueue")
        assert rec < w < nxt_copy                               # recorded, then the copy stream waits, then the prefetch copies


@pytest.mark.parametrize("arm", ["P0", "O"])
def test_the_early_prefetch_is_queued_before_the_layers_compute_and_behind_its_demand_copies(arrays, arm):
    arena, _ = _run_logged(arrays, arm)
    log = arena.log
    assert not any(e[0] == "copy_wait" for e in log)
    for pos in range(LAYERS - 1):
        dem_b = [i for i, e in enumerate(log) if e[0] == "rec_copy" and e[1:3] == ("dem_b", pos)]
        pf_a = [i for i, e in enumerate(log) if e[0] == "rec_copy" and e[1:3] == ("pf_a", pos)]
        pre = [i for i, e in enumerate(log) if e[0] == "rec_compute" and e[1:3] == ("pre", pos)]
        for d, p, q in zip(dem_b, pf_a, pre, strict=False):
            assert d < p < q


def test_the_floor_arm_issues_no_copies_and_no_waits(arrays):
    arena, res = _run_logged(arrays, "N")
    assert not any(e[0] in ("enqueue", "compute_wait", "copy_wait", "rec_copy") for e in arena.log)
    assert res["book"] is None and res["issued_copies"] == 0


# ------------------------------------------------------------------------------------ the timed replay and the real run path

@pytest.mark.parametrize("arm", ["N", "D", "P0", "P40", "O"])
def test_run_cell_counts_equal_the_bookkeeping_and_the_timings_close(arrays, arm):
    arena = oa.OverlapArena(SLOTS, 2048, rehearsal_cpu=True, n_layers=LAYERS)
    arena.standin.set_bytes(1 << 12)
    res = oa.run_cell(arrays, SLOTS, LAYERS, arena, arm, max_decode_tokens=8, readback_every=50)
    assert len(res["tokens"]) == 8 and not res["truncated"] and res["readback_failures"] == 0
    cl = oa.closure(res["tokens"])
    assert cl["ok"], cl
    if arm != "N":
        mode = oa.ARM_MODE[arm]
        ref = oa.bookkeeping_pass_pf(arrays, SLOTS, LAYERS, mode, 8)
        assert res["book"]["hash"] == ref["hash"] and res["book"]["demand_misses"] == ref["demand_misses"]
        assert res["readbacks"] > 0


def _group(**kw):
    g = {"id": "p17", "percent": 17, "slots": SLOTS, "tokens": 8, "c_values": (0.05,), "arms": ("N", "D", "P0", "P40", "O", "N")}
    g.update(kw)
    return g


def _gates():
    return {"event_hash": True, "determinism": True, "counts": True, "readback": True, "calibration": True, "drift": True,
            "closure": True, "sanity": True}


def _rehearse(arrays, drift_limit=10.0, **kw):
    """The CPU timings of microsecond kernels are noisy, so the drift limit is wide here; the trip test passes the real one."""
    gates, failures, raw = _gates(), [], {}
    rows, status = oa.run_group(arrays, _group(), None, "r", suite.Deadline(120), gates, failures, raw, rehearsal=True,
                                weight_start=1 << 16, calib_tolerance=50.0, nbytes=2048, readback_every=20,
                                drift_limit=drift_limit, **kw)
    return rows, status, gates, failures, raw


def test_the_real_run_path_with_every_gate_asserted(arrays):
    rows, status, gates, failures, raw = _rehearse(arrays)
    assert status == "complete" and failures == [] and all(gates.values()), (gates, failures)
    validate(rows)
    fields = {r["field"] for r in rows}
    assert {"wall_ms_token", "compute_busy_ms_token", "compute_stall_ms_token", "host_gap_ms_token", "copy_busy_demand_ms_token",
            "copy_busy_prefetch_ms_token", "closure_share", "demand_misses_per_token", "wasted_per_token", "E_D", "H_P0", "H_P40", "H_O",
            "drift", "per_copy_ms_c", "source_copy_ms_median", "calibration_error"} <= fields
    arms = {r["arm"] for r in rows}
    assert {"n3.p17.C0.05.N0", "n3.p17.C0.05.N1", "n3.p17.C0.05.D", "n3.p17.C0.05.P40"} <= arms


def test_cells_of_names_the_controls():
    ids = [c["id"] for c in oa.cells_of(_group())]
    assert ids == [f"p17.C0.05.{r}" for r in ("N0", "D", "P0", "P40", "O", "N1")]
    full = oa.cell_defs("full")
    assert sum(len(oa.cells_of(g)) for g in full) == 22 and {g["percent"] for g in full} == {17, 8}


def test_a_wrong_expected_hash_trips_the_event_hash_gate(arrays):
    gates, failures, raw = _gates(), [], {}
    exp = {"cells": {f"17.8.{m}": {"hash": "0" * 64} for m in ("none", "pred", "oracle")}}
    oa.run_group(arrays, _group(), exp, "r", suite.Deadline(120), gates, failures, raw, rehearsal=True, weight_start=1 << 16,
                 calib_tolerance=50.0, nbytes=2048)
    assert gates["event_hash"] is False and any("event hash" in f["error"] for f in failures)


def test_a_failing_readback_trips_the_readback_gate(arrays, monkeypatch):
    monkeypatch.setattr(oa.OverlapArena, "verify", lambda self, slot, key: False)
    _rows, _s, gates, _f, _raw = _rehearse(arrays)
    assert gates["readback"] is False


def test_a_bad_calibration_trips_the_calibration_gate_and_skips_the_cells(arrays, monkeypatch):
    monkeypatch.setattr(oa.StandIn, "measure", lambda self, iters=30, warm=5: 99.0)
    rows, _s, gates, failures, _raw = _rehearse(arrays)
    assert gates["calibration"] is False and rows == [] and any("calibration" in f["error"] for f in failures)


def _scaled_run_cell(monkeypatch, which, factor, fields=("wall_ms", "span_ms", "busy_ms")):
    real = oa.run_cell
    seen = {"N": 0}

    def wrapped(arrays, slots, nl, arena, arm, **kw):
        res = real(arrays, slots, nl, arena, arm, **kw)
        if arm == "N":
            seen["N"] += 1
        if which(arm, seen["N"]):
            for t in res["tokens"]:
                for f in fields:
                    t[f] *= factor
        return res

    monkeypatch.setattr(oa, "run_cell", wrapped)


def test_a_drifting_control_trips_the_drift_gate(arrays, monkeypatch):
    _scaled_run_cell(monkeypatch, lambda arm, n: arm == "N" and n == 2, 3.0)
    _rows, _s, gates, failures, _raw = _rehearse(arrays, drift_limit=oa.DRIFT_LIMIT)
    assert gates["drift"] is False and any("drift" in f["error"] for f in failures)


def test_a_wall_time_the_events_do_not_explain_trips_the_closure_gate(arrays, monkeypatch):
    _scaled_run_cell(monkeypatch, lambda arm, n: arm == "D", 3.0, fields=("wall_ms",))
    _rows, _s, gates, _f, _raw = _rehearse(arrays)
    assert gates["closure"] is False


def test_nondeterministic_bookkeeping_and_wrong_counts_trip_their_gates(arrays, monkeypatch):
    real = oa.bookkeeping_pass_pf
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        out = dict(real(*a, **k))
        if calls["n"] % 2 == 0:
            out["hash"] = "f" * 64                                  # the second pass of a cell disagrees with its first
        return out

    monkeypatch.setattr(oa, "bookkeeping_pass_pf", flaky)
    # reference_pass_pf is the comparison when no expected file is given, and it is not patched: first call must still match
    _rows, _s, gates, _f, _raw = _rehearse(arrays)
    assert gates["determinism"] is False or gates["event_hash"] is False
    monkeypatch.undo()

    def wrong_counts(*a, **k):
        out = dict(real(*a, **k))
        out["demand_misses"] += 1
        return out

    monkeypatch.setattr(oa, "bookkeeping_pass_pf", wrong_counts)
    _rows, _s, gates, _f, _raw = _rehearse(arrays)
    assert gates["counts"] is False or gates["event_hash"] is False


def test_a_deadline_truncates_and_says_so(arrays):
    gates, failures, raw = _gates(), [], {}
    _rows, status = oa.run_group(arrays, _group(), None, "r", suite.Deadline(-1), gates, failures, raw, rehearsal=True,
                                 weight_start=1 << 16, calib_tolerance=50.0, nbytes=2048)
    assert status == "truncated"


def test_plan_expect_and_dry_run(tmp_path, arrays, capsys):
    assert oa.main(["plan"]) == 0 and "p17" in capsys.readouterr().out
    assert oa.main(["run", "--trace", "x", "--expected", "y", "--cells", "smoke"]) == 0
    trace = tmp_path / "t.npz"
    np.savez(trace, segments=np.array("x"), **arrays)
    out = tmp_path / "e.json"
    assert oa.main(["expect", "--trace", str(trace), "--out", str(out)]) == 0
    import json

    exp = json.loads(out.read_text())
    assert exp["trace_sha256"] == ra.file_sha256(trace) and "17.40.pred" in exp["cells"] and "8.400.none" in exp["cells"]
    assert all(c["arena_equals_reference"] for c in exp["cells"].values())


REAL_TRACE = ROOT / "benchmarks" / "results" / "routing-trace-deepseek-long" / "trace.npz"


@pytest.mark.skipif(not REAL_TRACE.exists(), reason="the 11 MB recorded trace is not in the repository (git-ignored results)")
def test_the_real_run_path_on_the_recorded_trace_with_every_gate(tmp_path):
    """The recorded trace interleaves eight prefill segments among the decode rows; 30 tokens, tiny records, CPU."""
    with np.load(REAL_TRACE) as d:
        full = {k: d[k] for k in d.files if k != "segments"}
    g = {"id": "p17", "percent": 17, "tokens": 30, "c_values": (0.05,), "arms": ("N", "D", "P0", "P40", "O", "N")}
    gates, failures, raw = _gates(), [], {}
    rows, status = oa.run_group(full, g, None, "r", suite.Deadline(300), gates, failures, raw, rehearsal=True, weight_start=1 << 16,
                                calib_tolerance=50.0, nbytes=4096, readback_every=500, drift_limit=10.0)
    assert status == "complete" and failures == [] and all(gates.values()), (gates, failures)
    validate(rows)
    ref = oa.bookkeeping_pass_pf(full, ra.slots_for(17), 40, "pred", 30)
    got = {r["field"]: r["value"] for r in rows if r["arm"] == "n3.p17.C0.05.P0"}
    assert got["demand_misses_per_token"] == pytest.approx(ref["demand_per_token"])
    assert got["prefetch_loads_per_token"] == pytest.approx(ref["loads_per_token"])
