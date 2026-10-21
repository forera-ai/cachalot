"""Session K scoring script (Mac only; record docs/CUDA-PIN-ROUTE-RECORD.md): planted runs, every verdict branch, the run finder and the CLI."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))

import backing_probe as bp  # noqa: E402
import pin_route as pr  # noqa: E402
import pin_route_score as ps  # noqa: E402

NB = bp.RECORD_BYTES
FAST, SLOW = NB / 54.6e9 * 1e3, NB / 27.7e9 * 1e3
PLATEAU_MS = [64 * 2**20 / 52e9 * 1e3, 64 * 2**20 / 51e9 * 1e3, 64 * 2**20 / 50e9 * 1e3]
LINK5 = {"samples": 40, "gens": {"5": 40}, "widths": {"16": 40}, "min_gen": 5, "max_gen": 5}


def classify(ms, stream=1517.0, link=LINK5):
    return {"ms": ms, "roundtrip_ok": True, "nbytes": NB, "kept": len(ms), "plateau_ms": PLATEAU_MS, "plateau_bytes": pr.PLATEAU_BYTES,
            "stream": {"ms": 2 * pr.STREAM_BYTES / (stream * 1e9) * 1e3, "gbps": stream, "nbytes": pr.STREAM_BYTES}, "link": link}


def route(ms, link=None):
    r = {"ms": ms, "roundtrip_ok": True, "nbytes": NB, "errors": []}
    if link:
        r["link"] = link
    return r


def pin_run(tmp, group, cell, result, elapsed=20.0, ok=True):
    d = tmp / f"20261022T0000{int(group[1]):02d}Z-pinroute-{group}-abc123"
    d.mkdir()
    (d / "raw-samples.json").write_text(json.dumps({"run": d.name, "samples": {ps.CELLS[group]: result}}))
    (d / "run-manifest.json").write_text(json.dumps({"status": "complete", "limits": {"elapsed_s": elapsed},
                                                      "gates": {"roundtrip": ok, "sanity": True, "failed_cells": []}}))
    (d / "rows.jsonl").write_text("")
    return d


def make_pin(tmp, k0, k1, k2, k3, k4, k5):
    for g, r in (("k0", k0), ("k1", k1), ("k2", k2), ("k3", k3), ("k4", k4), ("k5", k5)):
        pin_run(tmp, g, ps.CELLS[g], r)
    return {g: ps.load_run(d) for g, d in ps.find_pin_runs(tmp).items()}


def slow_instance(tmp, routes=(SLOW, SLOW, SLOW, SLOW)):
    return make_pin(tmp, classify([SLOW] * 12), route([routes[0]] * 24), route([routes[1]] * 24), route([routes[2]] * 24),
                    route([routes[3]] * 24, LINK5), classify([SLOW] * 12, stream=1500.0))


def test_a_slow_instance_with_no_route_effect_scores_every_pin_prediction_in_interval(tmp_path):
    runs = slow_instance(tmp_path)
    res = ps.score(runs, [])
    v = res["verdict"]
    assert res["pin"]["instance_class_k0"] == "slow" and res["pin"]["instance_class_k5"] == "slow"
    assert res["pin"]["plateau_gbps_k0"] == pytest.approx(52.0, rel=1e-3)
    for k in ("K-1", "K-2", "K-3", "K-4", "K-5", "K-6", "K-7", "K-11"):
        assert v[k] == "in interval", (k, v)


def test_a_fast_instance_makes_the_route_question_untestable_not_confirmed(tmp_path):
    runs = make_pin(tmp_path, classify([FAST] * 12), route([FAST] * 24), route([FAST] * 24), route([FAST] * 24), route([FAST] * 24, LINK5),
                    classify([FAST] * 12))
    v = ps.score(runs, [])["verdict"]
    assert v["K-1"] == "in interval" and v["K-5"].startswith("not testable")


def test_a_route_that_moves_the_class_is_past_the_falsifier(tmp_path):
    runs = slow_instance(tmp_path, routes=(SLOW, SLOW, FAST, SLOW))        # ordinary memory copies at the fast rate while the control is slow
    v = ps.score(runs, [])["verdict"]
    assert v["K-5"] == "past falsifier" and v["K-7"] == "past falsifier"


def test_the_flag_and_class_stability_falsifiers(tmp_path):
    runs = make_pin(tmp_path, classify([SLOW] * 12), route([SLOW] * 24), route([SLOW * 0.8] * 24), route([SLOW] * 24), route([SLOW] * 24, LINK5),
                    classify([FAST] * 12, stream=1500.0 * 0.7))
    v = ps.score(runs, [])["verdict"]
    assert v["K-1"] == "past falsifier" and v["K-6"] == "past falsifier" and v["K-4"] == "past falsifier"


def test_plateau_agreement_and_stream_intervals(tmp_path):
    runs = slow_instance(tmp_path)
    runs["k0"]["raw"]["classify_start"]["plateau_ms"] = [64 * 2**20 / 30e9 * 1e3] * 3         # a 30 GB/s plateau
    assert ps.score(runs, [])["verdict"]["K-2"] == "past falsifier"


def test_unavailable_ordinary_registration_is_part_of_the_prediction(tmp_path):
    runs = slow_instance(tmp_path)
    runs["k3"]["raw"]["ordinary_flag0"] = {"ms": [], "roundtrip_ok": True, "nbytes": NB, "errors": ["RuntimeError: cudaHostRegister returned 1"]}
    assert ps.score(runs, [])["verdict"]["K-7"] == "in interval"


def test_h_link_is_supported_only_when_the_slow_cell_shows_the_lower_generation(tmp_path):
    low = {"samples": 40, "gens": {"4": 10, "5": 30}, "widths": {"16": 40}, "min_gen": 4, "max_gen": 5}
    runs = make_pin(tmp_path, classify([FAST] * 12), route([FAST] * 24), route([FAST] * 24), route([FAST] * 24), route([SLOW] * 24, low),
                    classify([FAST] * 12))
    assert ps.score(runs, [])["verdict"]["K-11"] == "past falsifier (H-link supported)"
    other = tmp_path / "b"
    other.mkdir()
    runs = make_pin(other, classify([FAST] * 12), route([FAST] * 24), route([FAST] * 24), route([FAST] * 24), route([FAST] * 24, low),
                    classify([FAST] * 12))
    assert ps.score(runs, [])["verdict"]["K-11"] == "outside interval"                    # a lower sample with a fast cell proves nothing


def test_the_pin_gates_and_time_feed_k12(tmp_path):
    runs = slow_instance(tmp_path)
    assert ps.score(runs, [])["verdict"]["K-12"] == "in interval"
    runs["k2"]["manifest"]["gates"]["roundtrip"] = False
    assert ps.score(runs, [])["verdict"]["K-12"] == "past falsifier"


# ------------------------------------------------------------------------------------------------ the guard

def guard_run(tmp, name, calls=0, planted_hits=40, tokens=40, host=1.0, guard_host=1.0, sources=None):
    raw = {}
    for level, c in ((3, "1.0"), (4, "1.0"), (4, "0.3")):
        for role in ("N0", "D", "P0L", "N1"):
            plain = f"p17k.L{level}.C{c}.{role}"
            raw[plain] = {"wall_ms": [5.0] * 6, "host_ms": [host] * 6}
            raw[f"p17k.L{level}.C{c}.G.{role}"] = {"wall_ms": [5.0] * 6, "host_ms": [guard_host] * 6,
                                                   "sync_guard": {"calls": calls if role in ("D", "P0L") else 0, "guarded_rows": 100, "tokens": tokens,
                                                                  "tokens_with_hit": 0, "planted": False, "planted_calls": 0, "sources": {}, "token_hits": []}}
    for role in ("N0", "P0L", "N1"):
        raw[f"p17k.L4.C0.3.GP.{role}"] = {"wall_ms": [5.0] * 6, "host_ms": [guard_host] * 6,
                                          "sync_guard": {"calls": planted_hits if role == "P0L" else 0, "guarded_rows": 100, "tokens": tokens,
                                                         "tokens_with_hit": planted_hits if role == "P0L" else 0, "planted": True, "planted_calls": tokens,
                                                         "sources": sources if sources is not None else {"overlap_arena.py:1": planted_hits}, "token_hits": []}}
    d = tmp / name
    d.mkdir()
    (d / "raw-samples.json").write_text(json.dumps({"run": name, "samples": raw}))
    (d / "run-manifest.json").write_text(json.dumps({"status": "complete", "limits": {"elapsed_s": 300.0}, "gates": {"event_hash": True, "failed_cells": []}}))
    (d / "rows.jsonl").write_text("")
    return d


def test_a_clean_guard_scores_k8_k9_k10_in_interval(tmp_path):
    g = ps.score_guard([ps.load_run(guard_run(tmp_path, "g1"))])
    assert g["guard_sync_calls"] and all(x == 0 for x in g["guard_sync_calls"].values())
    assert g["guard_host_cost"] == {"L4.C1.0": 0.0, "L4.C0.3": 0.0} and g["planted"]["tokens_with_hit"] == 40
    v = ps.score(slow_instance(_sub(tmp_path)), [ps.load_run(guard_run(tmp_path, "g2"))])["verdict"]
    assert v["K-8"] == "in interval" and v["K-9"] == "in interval" and v["K-10"] == "in interval" and v["K-12"] == "in interval"


def _sub(tmp):
    d = tmp / "pin"
    d.mkdir(exist_ok=True)
    return d


def test_a_counted_synchronisation_in_a_guarded_cell_is_the_finding(tmp_path):
    v = ps.score(slow_instance(_sub(tmp_path)), [ps.load_run(guard_run(tmp_path, "g", calls=3))])["verdict"]
    assert v["K-8"] == "past falsifier" and v["K-10"] == "in interval"


def test_a_blind_guard_voids_k8(tmp_path):
    v = ps.score(slow_instance(_sub(tmp_path)), [ps.load_run(guard_run(tmp_path, "g", planted_hits=5))])["verdict"]
    assert v["K-10"] == "past falsifier" and v["K-8"].startswith("void")


def test_guard_cost_intervals(tmp_path):
    v = ps.score(slow_instance(_sub(tmp_path)), [ps.load_run(guard_run(tmp_path, "g", guard_host=1.15))])["verdict"]
    assert v["K-9"] == "outside interval"
    other = tmp_path / "x"
    other.mkdir()
    v = ps.score(slow_instance(_sub(other)), [ps.load_run(guard_run(tmp_path, "h", guard_host=1.5))])["verdict"]
    assert v["K-9"] == "past falsifier"


def test_cell_ids_parse_with_and_without_the_guard_tag():
    gc = ps.guard_cells({"p17k.L4.C0.3.G.P0L": {"x": 1}, "p17k.L3.C1.0.N0": {"x": 2}, "p17k.L4.C0.3.GP.P0L": {"x": 3}, "p17k.L4.C0.05.N1": {"x": 4}})
    assert gc[(4, 0.3, "guard")]["P0L"] == {"x": 1} and gc[(3, 1.0, "plain")]["N0"] == {"x": 2}
    assert gc[(4, 0.3, "planted")]["P0L"] == {"x": 3} and gc[(4, 0.05, "plain")]["N1"] == {"x": 4}


def test_the_finder_takes_the_newest_run_of_each_group_and_the_cli_writes_json(tmp_path, capsys):
    root = _sub(tmp_path)
    slow_instance(root)
    old = root / "20250101T000000Z-pinroute-k0-old"
    old.mkdir()
    assert ps.find_pin_runs(root)["k0"].name.startswith("20261022") and set(ps.find_pin_runs(root)) == set(ps.CELLS)
    gd = guard_run(tmp_path, "g")
    out = tmp_path / "o.json"
    assert ps.main(["--pin-root", str(root), "--guard", str(gd), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"]["K-8"] == "in interval"
    assert "K-1" in capsys.readouterr().out
