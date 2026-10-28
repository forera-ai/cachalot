"""Follow-up N scorer (benchmarks/cuda/pin_route_settle_score.py) on synthetic runs: both sides of the hypotheses and the split."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "cuda"))

import pin_route as pr  # noqa: E402
import pin_route_settle_score as ss  # noqa: E402

SLOW, FAST = 0.359, 0.182
MAN = {"status": "complete", "gates": {"roundtrip": True, "sanity": True, "failed_cells": []}, "limits": {"elapsed_s": 1.0}}


def runs(wait500=SLOW, evict256=FAST, write=SLOW, train=SLOW):
    def run(group, name, res):
        return {"group": group, "manifest": MAN, "raw": {name: res}, "dir": group}
    waits = list(pr.WAITS_MS)
    evict = [pr.EVICT_BYTES[i % 3] for i in range(24)]
    modes = [pr.RETOUCH[i % 3] for i in range(24)]
    return [run("k4", "alloc_control", {"ms": [SLOW] * 24}), run("l1", "batch_first", {"ms": [FAST] * 24}),
            run("n1", "settle_train", {"trains": [{"ms": [train] * 200, "t_ms": [float(i) for i in range(200)]}] * 6}),
            run("n2", "wait_sweep", {"ms": [SLOW if w == 0 else wait500 for w in waits], "wait_ms": waits}),
            run("n3", "evict", {"ms": [SLOW if e == 0 else evict256 for e in evict], "evict_bytes": evict, "fill_ms": [9.0 if e else 0.0 for e in evict]}),
            run("n4", "retouch", {"ms": [write if m == "write" else FAST for m in modes], "mode": modes}),
            run("k4", "alloc_control", {"ms": [SLOW] * 24})]


def test_the_cache_side_reads_together_and_clean():
    v = ss.score(runs())["verdict"]
    assert v["N-1"] == v["N-2"] == v["N-3"] == v["N-4"] == v["N-5"] == v["N-6"] == v["N-7"] == v["N-8"] == "in interval"


def test_the_time_side_falsifies_n2_n3_n4_but_not_n7():
    v = ss.score(runs(wait500=FAST, evict256=SLOW, write=FAST, train=FAST))["verdict"]
    assert v["N-2"] == v["N-3"] == v["N-4"] == "past falsifier" and v["N-7"] == "in interval"


def test_the_observed_split_falsifies_n2_and_n7_and_a_failed_attempt_n8():
    out = ss.score(runs(wait500=FAST), failed_attempts=1)
    v = out["verdict"]
    assert v["N-2"] == "past falsifier" and v["N-3"] == v["N-4"] == "in interval" and v["N-7"] == "past falsifier" and v["N-8"] == "past falsifier"
    assert out["n1_stay_slow"] == 6 and 1.9 < out["slow_over_fast"]["k4"] < 2.0


def test_controls_that_do_not_separate_stop_the_scoring():
    r = runs()
    r[1]["raw"]["batch_first"]["ms"] = [SLOW] * 24
    v = ss.score(r)["verdict"]
    assert v == {"N-1": "past falsifier"}
