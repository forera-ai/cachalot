"""Follow-up L scoring script (Mac only; record docs/CUDA-PIN-ROUTE-FOLLOWUP-RECORD.md): planted runs and every verdict branch."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))

import backing_probe as bp  # noqa: E402
import pin_route as pr  # noqa: E402
import pin_route_followup_score as fs  # noqa: E402

NB = bp.RECORD_BYTES
SLOW, MID, FAST = 0.362, NB / 33.8e9 * 1e3, 0.194


def ms_for_gbps(g):
    return NB / (g * 1e9) * 1e3


def make(tmp, spec, elapsed=10.0, ok=True):
    """spec: one (group, [ms...]) per run in session order."""
    tmp.mkdir(exist_ok=True)
    for i, (g, ms) in enumerate(spec):
        d = tmp / f"2026102110{i:02d}00Z-pinroute-{g}-abc{i}"
        d.mkdir()
        (d / "raw-samples.json").write_text(json.dumps({"run": d.name, "samples": {fs.RAW_NAME[g]: {"ms": ms, "nbytes": NB, "roundtrip_ok": True}}}))
        (d / "run-manifest.json").write_text(json.dumps({"status": "complete", "limits": {"elapsed_s": elapsed},
                                                          "gates": {"roundtrip": ok, "sanity": True, "failed_cells": []}}))
    return tmp


def spec(l1=SLOW, l2=SLOW, l3=SLOW, l4=FAST, k1=33.848, k2=34.521, k3=31.26, k4a=SLOW, k4b=SLOW):
    r = lambda x: [x] * 24  # noqa: E731
    return [("k4", r(k4a)), ("k3", r(ms_for_gbps(k3))), ("k2", r(ms_for_gbps(k2))), ("k1", r(ms_for_gbps(k1))),
            ("l1", r(l1)), ("l2", r(l2)), ("l3", r(l3)), ("l4", r(l4)), ("k4", r(k4b))]


def verdict(tmp, **kw):
    return fs.score(fs.load_runs(make(tmp, spec(**kw))))["verdict"]


def test_one_factor_dominating_scores_the_consistent_predictions_in_interval(tmp_path):
    """L-2 and L-3 cannot both hold with L-5 (at most one factor fast alone), so the registered set is hedged: batch allocation alone
    fast with the other two slow scores every prediction in interval except L-3."""
    v = verdict(tmp_path, l1=FAST)
    assert set(v) == {f"L-{i}" for i in range(1, 10)}
    assert {k for k, x in v.items() if x != "in interval"} == {"L-3"} and v["L-3"] == "past falsifier", v


def test_the_arena_analogue_not_reproduced_is_past_the_falsifier(tmp_path):
    v = verdict(tmp_path, l4=SLOW)
    assert v["L-1"] == "past falsifier" and v["L-5"].startswith("not testable")


def test_each_single_factor_branch(tmp_path):
    assert verdict(tmp_path / "a", l1=FAST)["L-2"] == "in interval" and verdict(tmp_path / "b", l1=FAST)["L-5"] == "in interval"
    assert verdict(tmp_path / "c", l2=FAST)["L-3"] == "in interval"
    assert verdict(tmp_path / "d", l3=FAST)["L-4"] == "past falsifier"
    v = verdict(tmp_path / "e", l1=FAST, l2=FAST)
    assert v["L-5"] == "past falsifier"                                      # two factors suffice alone


def test_order_and_control_branches(tmp_path):
    assert verdict(tmp_path / "a", k1=33.848 * 1.15)["L-6"] == "outside interval"
    assert verdict(tmp_path / "b", k3=31.26 * 1.30)["L-6"] == "past falsifier"
    assert verdict(tmp_path / "c", k4b=0.25)["L-7"] == "past falsifier"
    assert verdict(tmp_path / "d", k4b=0.3615)["L-7"] == "in interval"
    assert verdict(tmp_path / "e", k4a=ms_for_gbps(36.0))["L-8"] == "past falsifier"


def test_completion_gate(tmp_path):
    runs = fs.load_runs(make(tmp_path, spec(), ok=False))
    assert fs.score(runs)["verdict"]["L-9"] == "past falsifier"
    runs = fs.load_runs(make(tmp_path / "x", spec(), elapsed=60.0))
    assert fs.score(runs)["verdict"]["L-9"] == "outside interval"            # nine runs of 60 s are past the 300 s budget, no failed gate


def test_cli_writes_json(tmp_path, capsys):
    folder = make(tmp_path / "f", spec())
    out = tmp_path / "o.json"
    assert fs.main([str(folder), "--json", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["order"] == list(pr.FOLLOWUP_ORDER) and "L-1" in capsys.readouterr().out
