"""Follow-up M scoring script (Mac only; record docs/CUDA-PIN-ORDER-RECORD.md): planted runs and every verdict branch."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))

import backing_probe as bp  # noqa: E402
import pin_route as pr  # noqa: E402
import pin_route_order_score as os_  # noqa: E402

NB = bp.RECORD_BYTES
SLOW, FAST = 0.362, 0.192
PLATEAU_FAST = [64 * 2**20 / 52e9 * 1e3] * 3


def ms_for(g):
    return NB / (g * 1e9) * 1e3


def smaps(huge):
    reg = 10 * 2**20
    return {"make": {"anon_huge_kb": (reg // 1024) if huge else 0, "size_kb": reg // 1024, "rss_kb": reg // 1024, "thp_eligible": 1}}


def spec(l1=FAST, k4=SLOW, m1=(FAST, SLOW), m4=SLOW, m6=FAST, m6_huge=24, m7=FAST, m8=FAST, m0=FAST, m9=FAST, plateau=PLATEAU_FAST):
    r = lambda x, n=24: [x] * n  # noqa: E731
    cls = lambda x: {"ms": r(x, 12), "nbytes": NB, "plateau_ms": plateau, "plateau_bytes": pr.PLATEAU_BYTES, "roundtrip_ok": True}  # noqa: E731
    reg = 10 * 2**20
    return {"m0": cls(m0), "k4": {"ms": r(k4), "nbytes": NB}, "l1": {"ms": r(l1), "nbytes": NB},
            "m1": {"ms": r(m1[0], 12) + r(m1[1], 12), "nbytes": NB, "split": 12}, "m4": {"ms": r(m4), "nbytes": NB},
            "m6": {"ms": r(m6), "nbytes": NB, "registered": reg, "smaps": [smaps(i < m6_huge) for i in range(24)]},
            "m7": {"ms": r(m7), "nbytes": NB, "registered": reg, "smaps": [smaps(False)] * 24}, "m8": {"ms": r(m8), "nbytes": NB}, "m9": cls(m9)}


def make(tmp, sp, elapsed=10.0, ok=True):
    tmp.mkdir(exist_ok=True)
    for i, g in enumerate(pr.ORDER_M):
        d = tmp / f"2026102111{i:02d}00Z-pinroute-{g}-abc{i}"
        d.mkdir()
        (d / "raw-samples.json").write_text(json.dumps({"run": d.name, "samples": {os_.RAW_NAME[g]: sp[g]}}))
        (d / "run-manifest.json").write_text(json.dumps({"status": "complete", "limits": {"elapsed_s": elapsed},
                                                          "gates": {"roundtrip": ok, "failed_cells": []}}))
    return tmp


def verdict(tmp, arena=0.194, **kw):
    return os_.score(os_.load_runs(make(tmp, spec(**kw))), arena)["verdict"]


def test_the_expected_fast_outcome_scores_in_interval(tmp_path):
    v = verdict(tmp_path)
    assert set(v) == {f"M-{i}" for i in range(1, 10)}
    assert all(x == "in interval" for k, x in v.items() if k not in ("M-3",)), v


def test_a_copy_before_allocation_slows_later_buffers_branch(tmp_path):
    v = verdict(tmp_path, m1=(FAST, SLOW), m4=SLOW)
    assert v["M-2"] == "in interval" and v["M-3"] == "in interval"
    v = verdict(tmp_path / "b", m1=(FAST, FAST), m4=FAST)
    assert v["M-2"] == "past falsifier" and v["M-3"] == "past falsifier"


def test_the_controls_not_separating_makes_the_rest_not_testable(tmp_path):
    v = verdict(tmp_path, l1=SLOW)
    assert v["M-1"] == "past falsifier"
    assert all(v[k].startswith("not testable") for k in ("M-2", "M-3", "M-4", "M-5", "M-6", "M-7"))
    v = verdict(tmp_path / "b", k4=FAST)
    assert v["M-1"] == "past falsifier"


def test_huge_pages_in_the_fast_pattern_branches(tmp_path):
    assert verdict(tmp_path / "a", m6_huge=10)["M-4"] == "outside interval"                  # too few verified
    assert verdict(tmp_path / "b", m6=SLOW)["M-4"] == "past falsifier"                       # verified buffers slow
    assert verdict(tmp_path / "c", m7=SLOW)["M-5"] == "past falsifier"
    assert verdict(tmp_path / "d", m8=ms_for(51.8 * 0.6))["M-6"] == "past falsifier"
    assert verdict(tmp_path / "e", m8=ms_for(51.8 * 0.9))["M-6"] == "in interval"


def test_classification_and_arena_agreement(tmp_path):
    assert verdict(tmp_path / "a", m9=SLOW)["M-7"] == "outside interval"                      # the end class differs
    v = verdict(tmp_path / "b", m0=SLOW, plateau=PLATEAU_FAST)
    assert v["M-7"] == "past falsifier"                                                      # m0 slow while the arena (0.194) is fast
    v = verdict(tmp_path / "c", plateau=[64 * 2**20 / 30e9 * 1e3] * 3)
    assert v["M-7"] == "past falsifier"                                                      # plateau below 40 GB/s
    assert verdict(tmp_path / "d", arena=0.50)["M-8"] == "past falsifier"
    assert verdict(tmp_path / "e", arena=0.205)["M-8"] == "in interval"


def test_completion_and_cli(tmp_path, capsys):
    runs = os_.load_runs(make(tmp_path / "a", spec(), ok=False))
    assert os_.score(runs)["verdict"]["M-9"] == "past falsifier"
    folder = make(tmp_path / "b", spec())
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    (smoke / "rows.jsonl").write_text(json.dumps({"field": "source_copy_ms_median", "value": 0.194}) + "\n")
    out = tmp_path / "o.json"
    assert os_.main([str(folder), "--smoke", str(smoke), "--json", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["order"] == list(pr.ORDER_M) and d["verdict"]["M-8"] == "in interval" and "M-1" in capsys.readouterr().out
