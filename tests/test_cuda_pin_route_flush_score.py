"""Follow-up O scorer (benchmarks/cuda/pin_route_flush_score.py) on synthetic runs: both sides of the hypotheses, the split, a mode the
instance could not run, and the transition point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "cuda"))

import pin_route as pr  # noqa: E402
import pin_route_flush_score as fs  # noqa: E402

SLOW, FAST, MID = 0.359, 0.182, 0.230
MAN = {"status": "complete", "gates": {"roundtrip": True, "sanity": True, "failed_cells": []}, "limits": {"elapsed_s": 1.0}}


def wait_ms(t):
    return SLOW if t <= 7.5 else (MID if t == 10 else FAST)


def runs(flush=None, read=None, short_gap=SLOW, long_gap=FAST, waits=wait_ms, unavailable3=None, unavailable4=None):
    flush = {"none": SLOW, "memcpy": SLOW, "nt": FAST, "clflushopt": FAST, "clwb": FAST, **(flush or {})}
    read = {"none": FAST, "direct": FAST, "buffered": SLOW, "direct_wait": FAST, **(read or {})}

    def run(group, name, res):
        return {"group": group, "manifest": MAN, "raw": {name: res}, "dir": group}

    fw = list(pr.FINE_WAITS_MS)
    trains = []
    for g in pr.GAPS_MS:
        later = short_gap if g in (1, 2) else (long_gap if g == 20 else SLOW)
        ms = [SLOW] + [later] * 39
        trains.append({"ms": ms, "t_ms": [float(i) for i in range(40)], "gap_ms": g})
    m3 = [m for m in pr.FLUSH_MODES * 5 if m not in (unavailable3 or {})]
    m4 = [m for m in pr.READ_MODES * 6 if m not in (unavailable4 or {})]
    return [run("k4", "alloc_control", {"ms": [SLOW] * 24}), run("l1", "batch_first", {"ms": [FAST] * 24}),
            run("o1", "fine_wait", {"ms": [waits(t) for t in fw], "wait_ms": fw}),
            run("o2", "gap_train", {"trains": trains}),
            run("o3", "flush", {"ms": [flush[m] for m in m3], "mode": m3, "unavailable": unavailable3 or {}}),
            run("o4", "direct_read", {"ms": [read[m] for m in m4], "mode": m4, "read_ms": [3.0] * len(m4), "unavailable": unavailable4 or {}}),
            run("k4", "alloc_control", {"ms": [SLOW] * 24})]


def test_the_dirty_side_reads_in_interval_everywhere():
    out = fs.score(runs())
    assert all(x == "in interval" for x in out["verdict"].values()), out["verdict"]
    assert 7.5 < out["o1_t50_ms"] <= 10 and out["o1_mid_between"] == 4


def test_the_page_side_falsifies_o8_o9_and_reads_o14_together():
    v = fs.score(runs(flush={"nt": SLOW, "clflushopt": SLOW, "clwb": SLOW}))["verdict"]
    assert v["O-8"] == v["O-9"] == v["O-10"] == "past falsifier" and v["O-14"] == "in interval"


def test_a_split_falsifies_o14():
    v = fs.score(runs(flush={"nt": SLOW}))["verdict"]
    assert v["O-8"] == "in interval" and v["O-9"] == "past falsifier" and v["O-14"] == "past falsifier"


def test_settling_short_gap_trains_falsify_o5_and_a_slow_dma_fill_o12():
    v = fs.score(runs(short_gap=FAST, read={"direct": SLOW}))["verdict"]
    assert v["O-5"] == "past falsifier" and v["O-12"] == "past falsifier" and v["O-6"] == "in interval"


def test_a_mode_the_instance_cannot_run_is_not_testable_and_o15_is_outside():
    out = fs.score(runs(unavailable3={"clwb": "CPUID does not report clwb"}, unavailable4={"direct": "O_DIRECT open refused", "direct_wait": "same"}))
    v = out["verdict"]
    assert v["O-10"] == v["O-11"] == v["O-12"] == "not testable" and v["O-8"] == "in interval" and v["O-15"] == "outside interval"


def test_a_switch_falsifies_o4_and_an_early_transition_o3():
    v = fs.score(runs(waits=lambda t: SLOW if t == 0 else FAST))["verdict"]
    assert v["O-4"] == "past falsifier" and v["O-3"] == "past falsifier" and v["O-2"] == "past falsifier"


def test_no_transition_inside_the_sweep_is_past_the_o3_falsifier():
    assert fs.settle_t50({0: 1.0, 5: 1.0, 20: 1.0, 50: 1.0}) is None
    assert fs.settle_t50({0: 1.0, 10: 0.0}) == 5.0 and fs.settle_t50({0: 0.25, 5: 0.0}) == 0.0
    v = fs.score(runs(waits=lambda t: SLOW))["verdict"]
    assert v["O-3"] == "past falsifier" and v["O-2"] == "past falsifier"


def test_controls_that_do_not_separate_stop_the_scoring_and_a_failed_attempt_fails_o15():
    r = runs()
    r[1]["raw"]["batch_first"]["ms"] = [0.359] * 24
    out = fs.score(r)
    assert out["verdict"] == {"O-1": "past falsifier"}
    assert fs.score(runs(), failed_attempts=1)["verdict"]["O-15"] == "past falsifier"


def test_the_scorer_reads_the_raw_samples_the_rehearsal_cells_write(tmp_path):
    """Schema check: every O cell run through the real run_group on the CPU rehearsal, its raw dict fed to the scorer as the GPU run would
    write it (json round trip). The rehearsal's times are not CUDA numbers, so the controls are replaced to let every section run."""
    import json

    import pytest
    pytest.importorskip("torch")
    import suite

    bench = pr.PinRouteBench(rehearsal_cpu=True)
    raws = {}
    for g in ("o1", "o2", "o3", "o4"):
        gates, failures, raw = {"roundtrip": True, "sanity": True}, [], {}
        pr.run_group(bench, g, "r", suite.Deadline(120), gates, failures, raw, small=True, work_dir=str(tmp_path))
        assert failures == [] and gates["roundtrip"]
        raws[g] = json.loads(json.dumps(raw))
    r = runs()
    for i, g in ((2, "o1"), (3, "o2"), (4, "o3"), (5, "o4")):
        r[i]["raw"] = raws[g]
    out = fs.score(r)
    assert out["verdict"]["O-1"] == "in interval" and set(out["verdict"]) == {f"O-{i}" for i in range(1, 16)}
    assert out["verdict"]["O-9"] == "not testable" or sys.platform.startswith("linux")
