"""
Follow-up M (Mac only; record `docs/CUDA-PIN-ORDER-RECORD.md`, predictions M-1 to M-9): why is a pinned buffer that is allocated and then copied
slow, and what do the huge-page, route and classification answers look like in the fast (batch-first) pattern?

Reads the nine run directories of `session.sh pinroute3` (one folder; taken in the order of their names, which is `pin_route.ORDER_M`: m0, k4,
l1, m1, m4, m6, m7, m8, m9) and optionally the `n3ksmoke` run directory (`--smoke`, for the arena cross-check M-8), and prints each registered
quantity beside its interval and falsifier. It judges nothing for the writer: the verdict column says only whether a number is inside the
registered interval, past its falsifier, or between. M-2 to M-6 are scored only if the two controls separate (M-1).

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/pin_route_order_score.py ORDER_DIR [--smoke N3K_SMOKE_DIR] [--json OUT]
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

RAW_NAME = {"m0": "classify_batch_start", "k4": "alloc_control", "l1": "batch_first", "m1": "half_copy_half", "m4": "copy_first",
            "m6": "private_thp_batch", "m7": "private_nothp_batch", "m8": "ordinary_batch", "m9": "classify_batch_end"}


def load_runs(folder) -> list[dict]:
    dirs = sorted(d for d in Path(folder).iterdir() if d.is_dir() and "-pinroute-" in d.name)
    out = []
    for d in dirs:
        man = json.loads((d / "run-manifest.json").read_text()) if (d / "run-manifest.json").exists() else {}
        raw = json.loads((d / "raw-samples.json").read_text()).get("samples", {}) if (d / "raw-samples.json").exists() else {}
        out.append({"dir": d.name, "manifest": man, "raw": raw, "group": d.name.split("-pinroute-")[1].split("-")[0]})
    return out


def arena_source_median(smoke_dir) -> float | None:
    p = Path(smoke_dir) / "rows.jsonl"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("field") == "source_copy_ms_median" and r.get("value") is not None:
                return float(r["value"])
    return None


def put(v: dict, key: str, ok: bool, fal: bool) -> None:
    v[key] = "in interval" if ok else ("past falsifier" if fal else "outside interval")


def stats(res: dict | None) -> dict | None:
    if not res or not res.get("ms"):
        return None
    ms = res["ms"]
    return {"n": len(ms), "median_ms": statistics.median(ms), "median_gbps": bp.gbps(res["nbytes"], statistics.median(ms)),
            "slow_share": bp.share(bp.slow_flags(ms, bp.SLOW_MS_9_95MB))}


def huge_verified(res: dict) -> tuple[int, int]:
    """(huge-page-verified buffers at the make point, how many of them are slow)."""
    pts, ms, reg = res.get("smaps", []), res["ms"], res["registered"]
    n = slow = 0
    for p, m in zip(pts, ms, strict=False):
        f = bp.huge_fraction(p.get("make"), reg)
        if f is not None and f >= 0.9:
            n += 1
            slow += m >= bp.SLOW_MS_9_95MB
    return n, slow


def score(runs: list[dict], arena_median: float | None = None) -> dict:
    cells: dict[str, dict] = {}
    raws: dict[str, dict] = {}
    for r in runs:
        g = r["group"]
        res = r["raw"].get(RAW_NAME.get(g, ""))
        if res is None:
            continue
        raws[g] = res
        s = stats(res)
        if s:
            cells[g] = s
    out: dict = {"cells": cells, "order": [r["group"] for r in runs], "elapsed_s": [r["manifest"].get("limits", {}).get("elapsed_s") for r in runs],
                 "gates": [{k: bool(x) for k, x in r["manifest"].get("gates", {}).items() if k != "failed_cells"}
                           | {"no_failed_cells": not r["manifest"].get("gates", {}).get("failed_cells")} for r in runs],
                 "status": [r["manifest"].get("status") for r in runs], "arena_source_median_ms": arena_median}
    v: dict = {}
    c = cells
    controls = {"l1", "k4"} <= set(c)
    separate = controls and c["l1"]["slow_share"] <= 0.2 and c["k4"]["slow_share"] >= 0.9
    if controls:
        put(v, "M-1", separate, not separate)
    testable = bool(separate)
    nt = "not testable (the controls do not separate on this instance)"
    if "m1" in raws:
        ms, k = raws["m1"]["ms"], raws["m1"]["split"]
        first, last = bp.share(bp.slow_flags(ms[:k], bp.SLOW_MS_9_95MB)), bp.share(bp.slow_flags(ms[k:], bp.SLOW_MS_9_95MB))
        out["m1_first_half_slow_share"], out["m1_second_half_slow_share"] = first, last
        if testable:
            put(v, "M-2", first <= 0.2 and last >= 0.8, last <= 0.2)
        else:
            v["M-2"] = nt
    if "m4" in c:
        if testable:
            put(v, "M-3", c["m4"]["slow_share"] >= 0.8, c["m4"]["slow_share"] <= 0.2)
        else:
            v["M-3"] = nt
    if "m6" in raws:
        n, slow = huge_verified(raws["m6"])
        out["m6_huge_verified"], out["m6_huge_verified_slow"] = n, slow
        if testable and "l1" in c:
            near = abs(c["m6"]["median_gbps"] / c["l1"]["median_gbps"] - 1) <= 0.10
            put(v, "M-4", n >= 14 and near, n > 0 and slow / n >= 0.5)
        else:
            v["M-4"] = nt
    if "m7" in c and "m6" in c:
        if testable:
            rel = c["m7"]["median_gbps"] / c["m6"]["median_gbps"]
            put(v, "M-5", c["m7"]["slow_share"] <= 0.2 and abs(rel - 1) <= 0.10, c["m7"]["slow_share"] >= 0.5 or rel < 0.70)
        else:
            v["M-5"] = nt
    if "m8" in c and "l1" in c:
        if testable:
            rel = c["m8"]["median_gbps"] / c["l1"]["median_gbps"]
            put(v, "M-6", abs(rel - 1) <= 0.15, rel < 0.70)
        else:
            v["M-6"] = nt
    cls = {}
    for g in ("m0", "m9"):
        if g in raws and raws[g].get("plateau_ms"):
            cls[g] = pr.classify_buffers(raws[g]["ms"], raws[g]["plateau_ms"], raws[g]["nbytes"], raws[g]["plateau_bytes"])
    out["classification"] = {g: {"class": x["instance_class"], "plateau_gbps": x["plateau_gbps"], "slow_count": x["slow_count"]} for g, x in cls.items()}
    if len(cls) == 2:
        if testable:
            arena_fast = arena_median is not None and arena_median <= 0.25
            put(v, "M-7", cls["m0"]["instance_class"] == "fast" == cls["m9"]["instance_class"] and 48.0 <= cls["m0"]["plateau_gbps"] <= 57.0,
                cls["m0"]["plateau_gbps"] < 40.0 or (cls["m0"]["instance_class"] == "slow" and arena_fast))
        else:
            v["M-7"] = nt
    if arena_median is not None and "m0" in c:
        d = abs(arena_median / c["m0"]["median_ms"] - 1)
        put(v, "M-8", d <= 0.10, d > 0.25)
    if runs:
        ok = all(all(g.values()) for g in out["gates"]) and all(s == "complete" for s in out["status"])
        total = sum(x for x in out["elapsed_s"] if x)
        put(v, "M-9", ok and total < 300 and len(runs) == len(pr.ORDER_M), not ok)
    out["verdict"] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--smoke")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    res = score(load_runs(args.folder), arena_source_median(args.smoke) if args.smoke else None)
    text = json.dumps(res, indent=2, sort_keys=True, default=float)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
