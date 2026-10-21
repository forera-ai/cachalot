"""
Session K follow-up L (Mac only; record `docs/CUDA-PIN-ROUTE-FOLLOWUP-RECORD.md`, predictions L-1 to L-9): what makes the arena process's pinned
copies fast when a fresh process's are slow?

Reads the nine run directories of `session.sh pinroute2` (one folder; the runs are taken in the order of their names, which is the order of
`pin_route.FOLLOWUP_ORDER`: k4, k3, k2, k1, l1, l2, l3, l4, k4) and prints each registered quantity beside its interval and falsifier. It
judges nothing for the writer: the verdict column says only whether a number is inside the registered interval, past its falsifier, or between.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/pin_route_followup_score.py FOLLOWUP_DIR [--json OUT]
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

RAW_NAME = {"k4": "alloc_control", "k3": "ordinary_flag0", "k2": "register_flag0", "k1": "register_portable",
            "l1": "batch_first", "l2": "arena_dest", "l3": "side_stream", "l4": "arena_full"}
FIRST_RUN_GBPS = {"k1": 33.848, "k2": 34.521, "k3": 31.260}          # medians of the first run (record section 3, L-6)
FIRST_RUN_K4_MS = 0.362


def load_runs(folder) -> list[dict]:
    dirs = sorted(d for d in Path(folder).iterdir() if d.is_dir() and "-pinroute-" in d.name)
    out = []
    for d in dirs:
        man = json.loads((d / "run-manifest.json").read_text()) if (d / "run-manifest.json").exists() else {}
        raw = json.loads((d / "raw-samples.json").read_text()).get("samples", {}) if (d / "raw-samples.json").exists() else {}
        out.append({"dir": d.name, "manifest": man, "raw": raw, "group": d.name.split("-pinroute-")[1].split("-")[0]})
    return out


def cell(run: dict) -> dict | None:
    res = run["raw"].get(RAW_NAME.get(run["group"], ""))
    if not res or not res.get("ms"):
        return None
    ms = res["ms"]
    return {"group": run["group"], "n": len(ms), "median_ms": statistics.median(ms), "median_gbps": bp.gbps(res["nbytes"], statistics.median(ms)),
            "slow_share": bp.share(bp.slow_flags(ms, bp.SLOW_MS_9_95MB)), "fast_share": sum(1 for x in ms if x <= 0.25) / len(ms)}


def put(v: dict, key: str, ok: bool, fal: bool) -> None:
    v[key] = "in interval" if ok else ("past falsifier" if fal else "outside interval")


def score(runs: list[dict]) -> dict:
    cells: dict[str, dict] = {}
    seen_k4 = 0
    for r in runs:
        c = cell(r)
        if c is None:
            continue
        key = r["group"]
        if key == "k4":
            key = "k4_first" if seen_k4 == 0 else "k4_last"
            seen_k4 += 1
        cells[key] = c
    out = {"cells": cells, "order": [r["group"] for r in runs], "elapsed_s": [r["manifest"].get("limits", {}).get("elapsed_s") for r in runs],
           "gates": [{k: bool(x) for k, x in r["manifest"].get("gates", {}).items() if k != "failed_cells"}
                     | {"no_failed_cells": not r["manifest"].get("gates", {}).get("failed_cells")} for r in runs],
           "status": [r["manifest"].get("status") for r in runs]}
    v: dict = {}
    c = cells
    if "l4" in c:
        put(v, "L-1", c["l4"]["slow_share"] <= 0.2 and c["l4"]["median_ms"] < 0.25, c["l4"]["slow_share"] >= 0.9)
    if "l1" in c:
        put(v, "L-2", c["l1"]["slow_share"] <= 0.5, c["l1"]["slow_share"] >= 0.9)
    if "l2" in c:
        put(v, "L-3", c["l2"]["slow_share"] <= 0.5, c["l2"]["slow_share"] >= 0.9)
    if "l3" in c:
        put(v, "L-4", c["l3"]["slow_share"] >= 0.9, c["l3"]["slow_share"] <= 0.5)
    if {"l1", "l2", "l3", "l4"} <= set(c):
        if c["l4"]["slow_share"] <= 0.2:
            alone = sum(1 for g in ("l1", "l2", "l3") if c[g]["slow_share"] <= 0.5)
            put(v, "L-5", alone <= 1, alone >= 2)
        else:
            v["L-5"] = "not testable (l4 is not fast, L-1 falsified or missed)"
    rel = {g: abs(c[g]["median_gbps"] / FIRST_RUN_GBPS[g] - 1.0) for g in FIRST_RUN_GBPS if g in c}
    if len(rel) == 3:
        put(v, "L-6", all(x <= 0.08 for x in rel.values()), any(x > 0.20 for x in rel.values()))
    out["route_rate_change_vs_first_run"] = rel
    if "k4_first" in c and "k4_last" in c:
        a, b = c["k4_first"], c["k4_last"]
        same = abs(a["median_ms"] / FIRST_RUN_K4_MS - 1) <= 0.01 and abs(b["median_ms"] / FIRST_RUN_K4_MS - 1) <= 0.01 and a["slow_share"] == b["slow_share"] == 1.0
        put(v, "L-7", same, a["median_ms"] < 0.30 or b["median_ms"] < 0.30 or abs(a["median_ms"] / b["median_ms"] - 1) > 0.05)
    if {"k1", "k2", "k3", "k4_first"} <= set(c):
        r = {g: c[g]["median_gbps"] for g in ("k1", "k2", "k3", "k4_first")}
        put(v, "L-8", min(r["k1"], r["k2"]) > r["k3"] > r["k4_first"], r["k4_first"] != min(r.values()))
    if runs:
        ok = all(all(g.values()) for g in out["gates"]) and all(s == "complete" for s in out["status"])
        total = sum(x for x in out["elapsed_s"] if x)
        put(v, "L-9", ok and total < 300 and len(runs) == len(pr.FOLLOWUP_ORDER), not ok)
    out["verdict"] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    res = score(load_runs(args.folder))
    text = json.dumps(res, indent=2, sort_keys=True, default=float)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
