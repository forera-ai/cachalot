"""N2 stage 1 harness: the arena bookkeeping equals the simulator event for event, and the copy loop runs on the CPU rehearsal."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import replay_arena as ra  # noqa: E402
import routing_gen as rg  # noqa: E402

torch = pytest.importorskip("torch")


def small_arrays(layers=4, experts=24, topk=3, tokens=80, prefill=12, skew=1.0, locality=0.3, seed=5):
    return rg.to_arrays(rg.RoutingSpec(layers=layers, experts=experts, topk=topk, tokens=tokens, skew=skew,
                                       locality=locality, seed=seed), prefill_tokens=prefill)


@pytest.mark.parametrize("slots", [6, 11, 24, 40, 90])
@pytest.mark.parametrize("seed", [1, 2])
def test_arena_events_equal_the_simulator(slots, seed):
    arrays = small_arrays(seed=seed)
    ref = ra.reference_pass(arrays, slots, 4)
    mine = ra.bookkeeping_pass(arrays, slots, 4)
    assert mine["hash"] == ref["hash"]
    assert mine["misses"] == ref["misses"] and mine["misses_per_token"] == ref["misses_per_token"]
    assert mine["requests"] == ref["requests"] and mine["tokens"] == ref["tokens"]


def test_the_gate_has_teeth_a_wrong_policy_changes_the_hash(monkeypatch):
    arrays = small_arrays()
    ref = ra.reference_pass(arrays, 11, 4)

    def mru_evict(self, avoid_layer=None):                       # evicts the most recently used key instead of the oldest
        victim = next(reversed(self.lru))
        slot = self.lru.pop(victim)
        self.free.append(slot)
        return victim, slot

    monkeypatch.setattr(ra.ArenaLRU, "_evict", mru_evict)
    assert ra.bookkeeping_pass(arrays, 11, 4)["hash"] != ref["hash"]


def test_a_token_limit_matches_the_simulator_on_the_same_prefix():
    arrays = small_arrays()
    for limit in (1, 7, 30):
        assert ra.bookkeeping_pass(arrays, 11, 4, limit)["hash"] == ra.reference_pass(arrays, 11, 4, limit)["hash"]
        assert ra.bookkeeping_pass(arrays, 11, 4, limit)["tokens"] == limit


def test_determinism_and_slot_invariants():
    arrays = small_arrays(tokens=120)
    a, b = ra.bookkeeping_pass(arrays, 20, 4), ra.bookkeeping_pass(arrays, 20, 4)
    assert a == b


def test_walk_matches_cache_sim_ordering_and_tokens():
    arrays = small_arrays()
    items = list(ra.walk(arrays))
    assert [i for i in items if i[0] == "P"] and all(i[0] in ("P", "D") for i in items)
    toks = [i[1] for i in items if i[0] == "D"]
    assert toks == sorted(toks) and toks[0] == 0
    from cachalot.metrics.routing_trace import _token_ordinals
    d = arrays["phase"] == ra.PHASE_DECODE
    assert (ra.token_ordinals(arrays["layer"][d]) == _token_ordinals(arrays["layer"][d])).all()


def test_fit_line_recovers_a_line_and_reports_r2():
    a, b, r2 = ra.fit_line([1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 9.0])
    assert (round(a, 9), round(b, 9), round(r2, 9)) == (1.0, 2.0, 1.0)
    assert ra.fit_line([2.0, 2.0], [1.0, 3.0])[1:] == (0.0, 0.0)
    with pytest.raises(ValueError):
        ra.fit_line([1.0], [1.0])


@pytest.mark.parametrize("slots", [11, 40])
def test_cpu_rehearsal_runs_the_copy_loop_and_the_gates_hold(slots):
    arrays = small_arrays(tokens=60)
    arena = ra.TorchArena(slots, 4096, rehearsal_cpu=True)
    res = ra.run_cell(arrays, slots, 4, arena, readback_every=10)
    ref = ra.bookkeeping_pass(arrays, slots, 4)
    assert res["readback_failures"] == 0 and res["readbacks"] > 0
    assert res["misses"] == ref["misses"] and res["chain"] == ref["chain"] and res["tokens"] == ref["tokens"]
    assert arena.copies == res["prefill_copies"] + res["misses"]
    assert len(res["token_host_ms"]) == len(res["token_misses"]) == res["tokens"]
    assert all(x >= 0 for x in res["token_host_ms"]) and res["memory"]["arena_bytes"] == slots * 4096


def test_readback_catches_a_corrupted_slot():
    arrays = small_arrays(tokens=40)
    arena = ra.TorchArena(24, 2048, rehearsal_cpu=True)
    orig_fetch = arena.fetch

    def corrupting_fetch(misses):
        out = orig_fetch(misses)
        slot, _key = misses[-1]
        arena._slot(slot).fill_(0)                              # the copy "landed" wrongly
        return out

    arena.fetch = corrupting_fetch
    res = ra.run_cell(arrays, 24, 4, arena, readback_every=1)
    assert res["readback_failures"] > 0


def test_rows_validate_and_per_copy_split_is_by_event_size():
    arrays = small_arrays(tokens=60)
    arena = ra.TorchArena(11, 4096, rehearsal_cpu=True)
    res = ra.run_cell(arrays, 11, 4, arena, readback_every=50)
    cell = {"id": "test.cell"}
    rows = ra.cell_rows("r", cell, res, True, [1.0, 1.1, 0.9], [1.0, 1.0, 1.0], 0.0)
    import manifest as mf
    assert rows and all(mf.validate_row(r) == [] for r in rows)
    assert {r["field"] for r in rows} >= {"exposed_copy_ms_token", "misses_per_token", "bytes_h2d_per_token", "event_hash_equals_expected"}
    split = ra.per_copy_by_event_size(res["layer_rows"])
    assert split["k1_events"] > 0


def test_a_deadline_truncates_the_replay():
    import suite
    arrays = small_arrays(tokens=60)
    arena = ra.TorchArena(11, 2048, rehearsal_cpu=True)
    res = ra.run_cell(arrays, 11, 4, arena, deadline=suite.Deadline(-1))
    assert res["truncated"] and res["tokens"] == 0


def test_plan_lists_the_registered_cells():
    assert [c["id"] for c in ra.cell_defs("full")] == ["trace.p8", "trace.p12", "trace.p17", "gen.p17"]
    assert [c["id"] for c in ra.cell_defs("smoke")] == ["trace.p17.t40", "gen.p17.t40"]
    assert [ra.slots_for(p) for p in (8, 12, 17, 34)] == [1229, 1843, 2611, 5222]


TRACE = ROOT / "benchmarks" / "results" / "routing-trace-deepseek-long" / "trace.npz"


@pytest.mark.skipif(not TRACE.exists(), reason="local routing trace not present (gitignored)")
def test_recorded_trace_event_hash_equals_the_simulator_at_17_percent_on_a_prefix():
    arrays = ra.load_arrays(TRACE)
    nl = ra.trace_layer_count(arrays)
    slots = ra.slots_for(17)
    assert ra.bookkeeping_pass(arrays, slots, nl, 60)["hash"] == ra.reference_pass(arrays, slots, nl, 60)["hash"]
