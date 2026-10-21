"""
Session K (Mac only; record `docs/CUDA-PIN-ROUTE-RECORD.md`, predictions K-1 to K-12): the pinning route (DC-3), the issue-loop synchronisation
guard (DC-5) and the committed classification of the instance's copy population (DC-6).

Reads the six `pinroute` run directories (groups k0 to k5; the newest run of each name under `--pin-root`) and the `n3k` run directory(ies)
of the guard profile (`--guard`, smoke and full), each holding `raw-samples.json`, `rows.jsonl` and `run-manifest.json`, and prints each
registered quantity beside its interval and falsifier. It judges nothing for the writer: the verdict column says only whether a number
is inside the registered interval, past its falsifier, or between. Scoring is by field name (record section 8).

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/pin_route_score.py --pin-root ~/cachalot-private/cuda-results/k1 \
        --guard RUN_DIR [--guard RUN_DIR2] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backing_probe as bp  # noqa: E402
import pin_route as pr  # noqa: E402

CELLS = {"k0": "classify_start", "k1": "register_portable", "k2": "register_flag0", "k3": "ordinary_flag0", "k4": "alloc_control", "k5": "classify_end"}
GUARD_GROUP = "p17k"


def load_run(d) -> dict:
    d = Path(d)
    rows = [json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines() if x.strip()] if (d / "rows.jsonl").exists() else []
    raw = json.loads((d / "raw-samples.json").read_text()).get("samples", {}) if (d / "raw-samples.json").exists() else {}
    man = json.loads((d / "run-manifest.json").read_text()) if (d / "run-manifest.json").exists() else {}
    return {"dir": str(d), "rows": rows, "raw": raw, "manifest": man}


def find_pin_runs(root) -> dict[str, Path]:
    """The newest `*-pinroute-kN-*` run directory of each group under `root`."""
    out: dict[str, Path] = {}
    for g in CELLS:
        found = sorted(Path(root).glob(f"*Z-pinroute-{g}-*"))
        if found:
            out[g] = found[-1]
    return out


def value(run: dict, arm_suffix: str, field: str):
    for r in run["rows"]:
        if r.get("field") == field and str(r.get("arm", "")).endswith(arm_suffix) and r.get("value") is not None:
            return r["value"]
    return None


def put(v: dict, key: str, ok: bool, fal: bool) -> None:
    v[key] = "in interval" if ok else ("past falsifier" if fal else "outside interval")


def median_rate(res: dict | None) -> float | None:
    if not res or not res.get("ms"):
        return None
    return bp.gbps(res["nbytes"], statistics.median(res["ms"]))


def route_slow_share(res: dict | None, plateau_gbps: float | None) -> float | None:
    if not res or not res.get("ms") or plateau_gbps is None:
        return None
    return sum(1 for m in res["ms"] if bp.gbps(res["nbytes"], m) < pr.SLOW_FRACTION * plateau_gbps) / len(res["ms"])


def guard_cells(raw: dict) -> dict:
    """Guard cell results by (level, c, kind) -> {role: raw cell}; kind is 'plain', 'guard' or 'planted'."""
    out: dict = {}
    for cid, cell in raw.items():
        parts = cid.split(".")
        if len(parts) < 5 or parts[0] != GUARD_GROUP or not parts[1].startswith("L"):
            continue
        level = int(parts[1][1:])
        c = float(parts[2][1:] + "." + parts[3])
        tail = parts[4:]
        kind = "plain"
        if tail[0] in ("G", "GP"):
            kind, tail = ("guard" if tail[0] == "G" else "planted"), tail[1:]
        out.setdefault((level, c, kind), {})[tail[0]] = cell
    return out


def score_pin(runs: dict[str, dict]) -> dict:
    out: dict = {"cells": {}}
    res = {g: runs[g]["raw"].get(CELLS[g]) if g in runs else None for g in CELLS}
    k0, k5 = res["k0"], res["k5"]
    cls = {}
    for g, r in (("k0", k0), ("k5", k5)):
        if r:
            cls[g] = pr.classify_buffers(r["ms"], r["plateau_ms"], r["nbytes"], r["plateau_bytes"])
    plateau = max([c["plateau_gbps"] for c in cls.values() if c["plateau_gbps"]], default=None)
    out["plateau_gbps_k0"] = cls.get("k0", {}).get("plateau_gbps")
    out["plateau_gbps_session"] = plateau
    out["instance_class_k0"], out["instance_class_k5"] = cls.get("k0", {}).get("instance_class"), cls.get("k5", {}).get("instance_class")
    agree = []
    for r, c in ((k0, cls.get("k0")), (k5, cls.get("k5"))):
        if r and c and c["slow"]:
            agree += [a == b for a, b in zip(c["slow"], bp.slow_flags(r["ms"], bp.SLOW_MS_9_95MB), strict=True)]
    out["label_agreement_pooled"] = sum(agree) / len(agree) if agree else None
    out["stream_gbps_k0"] = k0["stream"]["gbps"] if k0 else None
    out["stream_gbps_k5"] = k5["stream"]["gbps"] if k5 else None
    out["stream_changed"] = pr.stream_changed(out["stream_gbps_k0"], out["stream_gbps_k5"]) if k0 and k5 else None
    for g in ("k1", "k2", "k3", "k4"):
        r = res[g]
        out["cells"][g] = {"median_gbps": median_rate(r), "slow_share_relative": route_slow_share(r, plateau),
                           "slow_share_absolute": bp.share(bp.slow_flags(r["ms"], bp.SLOW_MS_9_95MB)) if r and r.get("ms") else None,
                           "n": len(r["ms"]) if r else 0, "errors": (r or {}).get("errors")}
    out["link"] = {g: (res[g] or {}).get("link") for g in ("k0", "k4", "k5")}
    out["slow_share_by_cell"] = {"k0": (cls["k0"]["slow_count"] / len(cls["k0"]["slow"])) if "k0" in cls and cls["k0"]["slow"] else None,
                                 "k5": (cls["k5"]["slow_count"] / len(cls["k5"]["slow"])) if "k5" in cls and cls["k5"]["slow"] else None,
                                 "k4": out["cells"]["k4"]["slow_share_relative"]}
    out["elapsed_s"] = {g: runs[g]["manifest"].get("limits", {}).get("elapsed_s") for g in runs}
    out["gates"] = {g: {k: bool(v) for k, v in runs[g]["manifest"].get("gates", {}).items() if k != "failed_cells"} |
                    {"no_failed_cells": not runs[g]["manifest"].get("gates", {}).get("failed_cells")} for g in runs}
    out["status"] = {g: runs[g]["manifest"].get("status") for g in runs}
    return out


def host_p50(cell: dict | None) -> float | None:
    if not cell or len(cell.get("host_ms", [])) < 3:
        return None
    return statistics.median(cell["host_ms"][1:])


def score_guard(runs: list[dict]) -> dict:
    out: dict = {"cells": {}, "gates": [], "status": [], "elapsed_s": []}
    for run in runs:
        gc = guard_cells(run["raw"])
        for (level, c, kind), roles in gc.items():
            for role, cell in roles.items():
                key = f"L{level}.C{c}.{kind}.{role}"
                sg = cell.get("sync_guard")
                out["cells"][key] = {"host_p50_ms": host_p50(cell), **({"sync_calls": sg["calls"], "tokens": sg["tokens"], "tokens_with_hit": sg["tokens_with_hit"],
                                                                       "sources": sg["sources"]} if sg else {})}
        g = run["manifest"].get("gates", {})
        out["gates"].append({k: bool(v) for k, v in g.items() if k != "failed_cells"} | {"no_failed_cells": not g.get("failed_cells")})
        out["status"].append(run["manifest"].get("status"))
        out["elapsed_s"].append(run["manifest"].get("limits", {}).get("elapsed_s"))
    cells = out["cells"]
    guarded = [k for k in cells if ".guard." in k and k.rsplit(".", 1)[1] in ("D", "P0L") and "sync_calls" in cells[k]]
    out["guard_sync_calls"] = {k: cells[k]["sync_calls"] for k in guarded}
    cost = {}
    for level, c in ((4, 1.0), (4, 0.3)):
        a, b = cells.get(f"L{level}.C{c}.plain.P0L", {}).get("host_p50_ms"), cells.get(f"L{level}.C{c}.guard.P0L", {}).get("host_p50_ms")
        if a and b:
            cost[f"L{level}.C{c}"] = abs(b - a) / a
    out["guard_host_cost"] = cost
    planted = [cells[k] for k in cells if ".planted." in k and k.endswith(".P0L") and cells[k].get("tokens")]
    out["planted"] = planted[0] if planted else None
    return out


def verdicts(pin: dict, guard: dict | None) -> dict:
    v: dict = {}
    a, b = pin.get("instance_class_k0"), pin.get("instance_class_k5")
    if a and b and "unavailable" not in (a, b):
        put(v, "K-1", a == b, a != b)
    p = pin.get("plateau_gbps_k0")
    if p is not None:
        put(v, "K-2", 48.0 <= p <= 57.0, p < 40.0 or p > 62.0)
    ag = pin.get("label_agreement_pooled")
    if ag is not None:
        put(v, "K-3", ag >= 0.90, ag < 0.75)
    s0, changed = pin.get("stream_gbps_k0"), pin.get("stream_changed")
    if s0 is not None and changed is not None:
        put(v, "K-4", 1300.0 <= s0 <= 1700.0 and not changed, not (1300.0 <= s0 <= 1700.0) or changed)
    cells = pin.get("cells", {})
    ctrl = cells.get("k4", {}).get("slow_share_relative")
    if ctrl is not None and ctrl < 0.2:
        v["K-5"] = "not testable (the control has no slow class)"
    elif ctrl is not None:
        diffs = [abs(cells[g]["slow_share_relative"] - ctrl) for g in ("k1", "k2", "k3") if cells.get(g, {}).get("slow_share_relative") is not None]
        if diffs:
            put(v, "K-5", max(diffs) <= 0.25, max(diffs) > 0.45)
    r1, r2 = cells.get("k1", {}).get("median_gbps"), cells.get("k2", {}).get("median_gbps")
    if r1 and r2:
        d = abs(r1 - r2) / min(r1, r2)
        put(v, "K-6", d < 0.05, d > 0.15)
    r3, r4 = cells.get("k3", {}).get("median_gbps"), cells.get("k4", {}).get("median_gbps")
    if r4 and not r3 and cells.get("k3", {}).get("errors"):
        v["K-7"] = "in interval"                                            # unavailable is part of the prediction
    elif r3 and r4:
        put(v, "K-7", abs(r3 - r4) / r4 <= 0.10, r3 > 1.30 * r4)
    if guard:
        calls = guard.get("guard_sync_calls", {})
        pl = guard.get("planted")
        if pl and pl["tokens"]:
            frac = pl["tokens_with_hit"] / pl["tokens"]
            put(v, "K-10", frac >= 0.95 and len(pl.get("sources", {})) == 1, frac < 0.50)
            if calls and v["K-10"] != "past falsifier":
                put(v, "K-8", all(x == 0 for x in calls.values()), any(x > 0 for x in calls.values()))
            elif calls:
                v["K-8"] = "void (the guard is blind: K-10 past falsifier)"
        cost = guard.get("guard_host_cost", {})
        if cost:
            put(v, "K-9", all(x < 0.10 for x in cost.values()), any(x > 0.25 for x in cost.values()))
    link_by = {g: x for g, x in pin.get("link", {}).items() if x and not x.get("unavailable") and x.get("samples")}
    if link_by:
        at_max = all(x["min_gen"] == x["max_gen"] for x in link_by.values()) and len({x["max_gen"] for x in link_by.values()}) == 1
        shares = pin.get("slow_share_by_cell", {})
        low = [g for g, x in link_by.items() if x["min_gen"] < x["max_gen"] and shares.get(g) is not None]
        ok_cells = [g for g, x in link_by.items() if x["min_gen"] == x["max_gen"] and shares.get(g) is not None]
        supports = any(shares[g] >= 0.5 for g in low) and any(shares[g] < 0.5 for g in ok_cells) and not any(shares[g] < 0.5 for g in low)
        v["K-11"] = "in interval" if at_max else ("past falsifier (H-link supported)" if supports else "outside interval")
    gates_ok = all(all(g.values()) for g in pin.get("gates", {}).values()) if pin.get("gates") else None
    if guard and guard.get("gates"):
        gates_ok = (gates_ok is not False) and all(all(g.values()) for g in guard["gates"])
    if gates_ok is not None:
        pin_s = sum(x for x in pin.get("elapsed_s", {}).values() if x)
        guard_s = sum(x for x in (guard or {}).get("elapsed_s", []) if x)
        complete = all(s == "complete" for s in list(pin.get("status", {}).values()) + list((guard or {}).get("status", [])))
        put(v, "K-12", gates_ok and complete and pin_s < 600 and guard_s < 900, not gates_ok or not complete)
    return v


def score(pin_runs: dict[str, dict], guard_runs: list[dict]) -> dict:
    pin = score_pin(pin_runs)
    guard = score_guard(guard_runs) if guard_runs else None
    return {"pin": pin, "guard": guard, "verdict": verdicts(pin, guard)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pin-root", required=True)
    ap.add_argument("--guard", action="append", default=[])
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    pin_runs = {g: load_run(d) for g, d in find_pin_runs(args.pin_root).items()}
    res = score(pin_runs, [load_run(d) for d in args.guard])
    text = json.dumps(res, indent=2, sort_keys=True, default=float)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
