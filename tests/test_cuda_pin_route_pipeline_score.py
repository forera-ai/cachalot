"""Follow-up P scorer (benchmarks/cuda/pin_route_pipeline_score.py) on synthetic runs: a sharp and a banded threshold, the fill rankings,
a mode the instance could not run, and the rehearsal's raw samples read through the scorer."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "cuda"))

import pin_route as pr  # noqa: E402
import pin_route_pipeline_score as ps  # noqa: E402

SLOW, FAST = 0.359, 0.182
MAN = {"status": "complete", "gates": {"roundtrip": True, "sanity": True, "failed_cells": []}, "limits": {"elapsed_s": 1.0}}
# per mode: (serial fill ms, serial flush ms, copy ms, pipelined wall per record ms)
GOOD = {"aged": (None, None, FAST, 0.20), "memcpy": (0.40, None, SLOW, 0.42), "flush": (0.40, 0.38, FAST, 0.80), "nt": (0.30, None, FAST, 0.31)}


def held_at(p):
    return p <= 1.8


def train(period, held, n=150):
    ms = [SLOW] * n if held else [SLOW] * 8 + [0.225] * (n - 8)
    return {"ms": ms, "t_ms": [period * i for i in range(n)], "period_ms": period}


def passes(modes, recipe):
    out = []
    for k, m in enumerate(modes):
        fill, flush, copy, wall = recipe[m]
        for kind in ("serial", "pipelined"):
            per = (fill or 0) + (flush or 0) + copy if kind == "serial" else wall
            out.append({"mode": m, "kind": kind, "records": 64, "wall_ms": per * 64, "copy_ms": [copy] * 64,
                        "fill_ms": [fill] * 64 if fill else [], "flush_ms": [flush] * 64 if flush else [], "content_ok": True, "pass": k})
    return out


def runs(held=held_at, recipe=None, unavailable=None):
    recipe = {**GOOD, **(recipe or {})}
    modes = [m for m in pr.PIPE_ORDER if m not in (unavailable or {})]

    def run(group, name, res):
        return {"group": group, "manifest": MAN, "raw": {name: res}, "dir": group}

    return [run("k4", "alloc_control", {"ms": [SLOW] * 24}), run("l1", "batch_first", {"ms": [FAST] * 24}),
            run("p1", "period_train", {"trains": [train(p, held(p)) for p in pr.PERIODS_MS]}),
            run("p2", "fill_pipeline", {"passes": passes(modes, recipe), "unavailable": unavailable or {}}),
            run("k4", "alloc_control", {"ms": [SLOW] * 24})]


def test_a_sharp_threshold_and_nt_winning_read_in_interval_everywhere():
    out = ps.score(runs())
    assert all(x == "in interval" for x in out["verdict"].values()), out["verdict"]
    assert abs(out["p1_p50_ms"] - 1.9) < 1e-9 and out["p1_mixed_periods"] == 0


def test_a_band_of_mixed_periods_falsifies_p4():
    def banded(p):
        return {1.4: True, 1.6: True, 2.4: False}.get(p)

    r = runs(held=lambda p: True)
    trains = []
    for i, p in enumerate(pr.PERIODS_MS):
        h = banded(p)
        trains.append(train(p, h if h is not None else (i % 2 == 0)))
    r[2]["raw"]["period_train"]["trains"] = trains
    out = ps.score(r)
    assert out["p1_mixed_periods"] >= 3 and out["verdict"]["P-4"] == "past falsifier"


def test_memcpy_winning_falsifies_p8_and_a_hidden_penalty_p10():
    out = ps.score(runs(recipe={"memcpy": (0.40, None, SLOW, 0.21), "nt": (0.30, None, FAST, 0.40)}))
    assert out["verdict"]["P-8"] == "past falsifier" and out["verdict"]["P-10"] == "past falsifier"


def test_threshold_outside_the_sweep_falsifies_p2_and_p3():
    v = ps.score(runs(held=lambda p: True))["verdict"]
    assert v["P-3"] == "past falsifier" and v["P-2"] == "past falsifier"
    assert ps.threshold_p50({1.4: 1.0, 1.6: 0.0}) == 1.5 and ps.threshold_p50({1.4: 0.0}) == 1.4 and ps.threshold_p50({1.4: 1.0}) is None


def test_a_model_far_off_falsifies_p9():
    v = ps.score(runs(recipe={"flush": (0.40, 0.38, FAST, 1.40)}))["verdict"]
    assert v["P-9"] == "past falsifier"


def test_a_mode_the_instance_cannot_run_is_not_testable_and_p11_is_outside():
    v = ps.score(runs(unavailable={"nt": "CPUID does not report nt", "flush": "no compiler"}))["verdict"]
    assert v["P-5"] == v["P-6"] == v["P-8"] == v["P-9"] == "not testable" and v["P-7"] == v["P-10"] == "in interval"
    assert v["P-11"] == "outside interval"


def test_controls_that_do_not_separate_stop_the_scoring_and_a_failed_attempt_fails_p11():
    r = runs()
    r[1]["raw"]["batch_first"]["ms"] = [SLOW] * 24
    assert ps.score(r)["verdict"] == {"P-1": "past falsifier"}
    assert ps.score(runs(), failed_attempts=1)["verdict"]["P-11"] == "past falsifier"


def test_the_scorer_reads_the_raw_samples_the_rehearsal_cells_write(tmp_path):
    """Schema check: p1 and p2 run through the real run_group on the CPU rehearsal, their raw dicts fed to the scorer after a json round trip
    (the controls are synthetic so every section runs; the rehearsal's times are not CUDA numbers)."""
    pytest.importorskip("torch")
    import suite

    bench = pr.PinRouteBench(rehearsal_cpu=True)
    r = runs()
    for i, g in ((2, "p1"), (3, "p2")):
        gates, failures, raw = {"roundtrip": True, "sanity": True}, [], {}
        pr.run_group(bench, g, "r", suite.Deadline(120), gates, failures, raw, small=True, work_dir=str(tmp_path))
        assert failures == [] and gates["roundtrip"]
        r[i]["raw"] = json.loads(json.dumps(raw))
    out = ps.score(r)
    assert out["verdict"]["P-1"] == "in interval" and set(out["verdict"]) == {f"P-{i}" for i in range(1, 12)}
    json.dumps(out)
