"""Follow-up Q scorer (benchmarks/cuda/pin_route_idle_score.py) on synthetic runs: the idle side, the frequency side, the pipeline readings,
and the rehearsal's raw samples read through the scorer."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "cuda"))

import pin_route as pr  # noqa: E402
import pin_route_idle_score as qs  # noqa: E402

SLOW, FAST = 0.359, 0.182
MAN = {"status": "complete", "gates": {"roundtrip": True, "sanity": True, "failed_cells": []}, "limits": {"elapsed_s": 1.0}}
WALLS = {"nt_s2_t1": 0.42, "nt_s3_t1": 0.41, "nt_s4_t1": 0.42, "nt_s2_t2": 0.30, "memcpy_s2_t1": 0.60, "memcpy_s3_t1": 0.60,
         "memcpy_s2_t2": 0.45, "aged_s2_t1": 0.18}


def train(sleep, held, n=100):
    ms = [SLOW] * n if held else [SLOW] * 6 + [0.225] * (n - 6)
    return {"ms": ms, "t_ms": [3.0 * i for i in range(n)], "back_ms": [3.0 * i + 0.4 for i in range(n)], "slept_ms": [sleep + 0.06] * n,
            "sleep_ms": sleep, "period_ms": 3.0}


def runs(held=lambda L: L < 2.0, busy=None, walls=None, unavailable=None):
    busy = {"sleep30": FAST, "spin30": SLOW, "spin60": SLOW, **(busy or {})}
    walls = {**WALLS, **(walls or {})}

    def run(group, name, res):
        return {"group": group, "manifest": MAN, "raw": {name: res}, "dir": group}

    modes = list(pr.BUSY_MODES) * 6
    passes = []
    for a in pr.Q_ARM_ORDER:
        mode, stages, threads = pr.Q_ARMS[a]
        if mode in (unavailable or {}):
            continue
        key = pr.arm_key(mode, stages, threads)
        passes.append({"mode": mode, "kind": "pipelined", "records": 64, "wall_ms": walls[key] * 64, "copy_ms": [0.26] * 64,
                       "fill_ms": [] if mode == "aged" else [0.4] * 64, "flush_ms": [], "content_ok": True, "stages": stages, "threads": threads})
    return [run("k4", "alloc_control", {"ms": [SLOW] * 24}), run("l1", "batch_first", {"ms": [FAST] * 24}),
            run("q1", "idle_stretch", {"trains": [train(L, held(L)) for L in pr.SLEEPS_MS]}),
            run("q2", "busy_wait", {"ms": [busy[m] for m in modes], "mode": modes}),
            run("q3", "pipe_variants", {"passes": passes, "unavailable": unavailable or {}}),
            run("k4", "alloc_control", {"ms": [SLOW] * 24})]


def test_the_idle_side_and_two_threads_helping_read_in_interval_everywhere():
    out = qs.score(runs())
    assert all(x == "in interval" for x in out["verdict"].values()), out["verdict"]
    assert abs(out["q1_l50_ms"] - 1.9) < 1e-9


def test_the_frequency_side_falsifies_q2_q3_q4_q5():
    v = qs.score(runs(held=lambda L: False, busy={"spin30": FAST, "spin60": FAST}))["verdict"]
    assert v["Q-2"] == v["Q-4"] == v["Q-5"] == "past falsifier" and v["Q-3"] == "past falsifier"


def test_a_deeper_pipeline_paying_falsifies_q7_and_no_thread_gain_q8():
    v = qs.score(runs(walls={"nt_s4_t1": 0.30, "nt_s2_t2": 0.43, "memcpy_s2_t2": 0.35}))["verdict"]
    assert v["Q-7"] == v["Q-8"] == v["Q-9"] == "past falsifier" and v["Q-6"] == "in interval"


def test_a_drifted_replicate_falsifies_q6_and_a_missing_mode_is_not_testable():
    assert qs.score(runs(walls={"nt_s2_t1": 0.50}))["verdict"]["Q-6"] == "past falsifier"
    v = qs.score(runs(unavailable={"nt": "CPUID does not report nt"}))["verdict"]
    assert v["Q-6"] == v["Q-7"] == v["Q-8"] == v["Q-9"] == "not testable" and v["Q-10"] == "outside interval"


def test_controls_that_do_not_separate_stop_the_scoring_and_a_failed_attempt_fails_q10():
    r = runs()
    r[1]["raw"]["batch_first"]["ms"] = [SLOW] * 24
    assert qs.score(r)["verdict"] == {"Q-1": "past falsifier"}
    assert qs.score(runs(), failed_attempts=1)["verdict"]["Q-10"] == "past falsifier"


def test_the_scorer_reads_the_raw_samples_the_rehearsal_cells_write(tmp_path):
    pytest.importorskip("torch")
    import suite

    bench = pr.PinRouteBench(rehearsal_cpu=True)
    r = runs()
    for i, g in ((2, "q1"), (3, "q2"), (4, "q3")):
        gates, failures, raw = {"roundtrip": True, "sanity": True}, [], {}
        pr.run_group(bench, g, "r", suite.Deadline(120), gates, failures, raw, small=True, work_dir=str(tmp_path))
        assert failures == [] and gates["roundtrip"]
        r[i]["raw"] = json.loads(json.dumps(raw))
    out = qs.score(r)
    assert out["verdict"]["Q-1"] == "in interval" and set(out["verdict"]) == {f"Q-{i}" for i in range(1, 11)}
    json.dumps(out)
