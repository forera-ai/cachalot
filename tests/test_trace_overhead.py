import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import trace_overhead as to  # noqa: E402


def test_column_trace_stores_events_in_order_and_stops_at_capacity():
    t = to.Trace(cap=3)
    for i in range(5):
        t.emit(7, i, i * 2)
    assert t.n == 3 and list(t.a[:3]) == [0, 1, 2] and list(t.b[:3]) == [0, 2, 4] and set(t.code[:3]) == {7}
    assert t.ts[0] <= t.ts[1] <= t.ts[2]
    t.reset()
    assert t.n == 0


def test_tuple_trace_matches_the_column_trace_interface():
    t = to.TupleTrace()
    t.emit(3, 1, 2)
    assert len(t.ev) == 1 and t.ev[0][1:] == (3, 1, 2)
    t.reset()
    assert t.ev == []


def test_simulated_token_runs_with_either_tracer_and_leaves_no_residue():
    sim = to.Sim(2000)
    for kind in ("column", "tuple"):
        sim.set_kind(kind)
        assert sim.token(False) > 0 and sim.token(True) > 0
        assert (sim.tracer_main.n if kind == "column" else len(sim.tracer_main.ev)) == 0  # reset after a traced token


def test_per_event_costs_are_sane():
    c = to.per_event_costs(20000)
    assert 0 < c["perf_counter_ns alone"] < 5000 and c["tracer off (the is-not-None test)"] < c["column store (shipped design)"]
