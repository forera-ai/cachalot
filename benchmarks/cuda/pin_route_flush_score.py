"""
Follow-up O (Mac only; record `docs/CUDA-PIN-FLUSH-RECORD.md`, predictions O-1 to O-15): is the half-rate pinned copy the CPU's dirty cache lines
(H-dirty) or a write-keyed state outside the caches (H-page), how long do the lines take to clear, and does an O_DIRECT fill avoid it?

Reads the seven run directories of `session.sh pinroute5` (one folder, taken in the order of their names, which is `pin_route.ORDER_O`: k4, l1,
o1, o2, o3, o4, k4) and prints each registered quantity beside its interval and falsifier. It judges nothing for the writer: the verdict says only
whether a number is inside the registered interval, past its falsifier, or between; a prediction whose mode the instance could not run is
"not testable". O-2 to O-14 are scored only if the controls separate (O-1). `--failed-attempts N` counts earlier attempts that stopped (O-15).

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/pin_route_flush_score.py FLUSH_DIR [--failed-attempts N] [--json OUT]
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
from pin_route_order_score import load_runs, put  # noqa: E402
from pin_route_settle_score import by_key, slow_share  # noqa: E402

RAW_NAME = {"k4": "alloc_control", "l1": "batch_first", "o1": "fine_wait", "o2": "gap_train", "o3": "flush", "o4": "direct_read"}
MID_WAITS = (7.5, 10, 12.5, 15, 17.5)


def settle_t50(shares: dict) -> float | None:
    """The first T at which the slow share falls to 0.5 or below, interpolated linearly from the T before it; None if it never does."""
    ts = sorted(shares)
    for i, t in enumerate(ts):
        if shares[t] <= 0.5:
            if i == 0:
                return float(t)
            tp, sp = ts[i - 1], shares[ts[i - 1]]
            return tp + (sp - 0.5) / (sp - shares[t]) * (t - tp)
    return None


def put_modes(v: dict, key: str, shares: dict, need: tuple, ok, fal) -> None:
    """`put` when every mode in `need` was timed, otherwise 'not testable'."""
    if any(shares.get(m) is None for m in need):
        v[key] = "not testable"
        return
    put(v, key, ok(shares), fal(shares))


def score(runs: list[dict], failed_attempts: int = 0) -> dict:
    res = {r["group"]: r for r in runs if r["group"] != "k4"}
    k4s = [r for r in runs if r["group"] == "k4"]
    raw = {g: r["raw"].get(RAW_NAME[g]) for g, r in res.items()}
    out: dict = {"order": [r["group"] for r in runs], "status": [r["manifest"].get("status") for r in runs],
                 "gates": [r["manifest"].get("gates") for r in runs], "elapsed_s": [r["manifest"].get("limits", {}).get("elapsed_s") for r in runs]}
    v: dict = {}
    k4_share = [slow_share(r["raw"]["alloc_control"]["ms"]) for r in k4s]
    l1 = raw.get("l1") or {"ms": []}
    l1_share = slow_share(l1["ms"])
    fast_median = statistics.median(l1["ms"]) if l1["ms"] else None
    out["controls"] = {"k4_slow_share": k4_share, "l1_slow_share": l1_share, "l1_median_ms": fast_median}
    sep = len(k4_share) == 2 and all(s is not None and s >= 0.9 for s in k4_share) and l1_share is not None and l1_share <= 0.1
    put(v, "O-1", sep, not sep)
    if not sep:
        out["verdict"] = v
        return out

    o1 = by_key(raw["o1"]["ms"], raw["o1"]["wait_ms"])
    shares = {t: slow_share(m) for t, m in o1.items()}
    out["o1"] = {pr.wait_key(t): {"slow_share": shares[t], "median_ms": statistics.median(m), "n": len(m)} for t, m in sorted(o1.items())}
    s = lambda t: shares.get(t)  # noqa: E731
    ends_ok = all(s(t) is not None for t in (0, 5, 20, 50)) and s(0) >= 0.75 and s(5) >= 0.75 and s(20) <= 0.25 and s(50) <= 0.25
    put(v, "O-2", ends_ok, (s(5) is not None and s(5) <= 0.25) or (s(20) is not None and s(20) >= 0.75))
    t50 = settle_t50(shares)
    out["o1_t50_ms"] = t50
    put(v, "O-3", t50 is not None and 7.5 <= t50 <= 15, t50 is None or t50 <= 5 or t50 > 20)
    between = lambda x: 0.20 < x < bp.SLOW_MS_9_95MB  # noqa: E731
    mid = sum(1 for t in MID_WAITS for x in o1.get(t, []) if between(x))
    out["o1_mid_between"] = mid
    put(v, "O-4", mid >= 3, not any(between(x) for x in raw["o1"]["ms"]))

    gaps = [pr.gap_summary(t) for t in raw["o2"]["trains"]]
    out["o2"] = gaps
    short = [g for g in gaps if g["gap_ms"] in (1, 2)]
    put(v, "O-5", bool(short) and all(g["second_half_slow_share"] is not None and g["second_half_slow_share"] >= 0.9 for g in short),
        any(g["settle_ms"] is not None for g in short))
    long_ = [g for g in gaps if g["gap_ms"] == 20]
    put(v, "O-6", bool(long_) and all(g["after_first_fast_share"] is not None and g["after_first_fast_share"] >= 0.9 for g in long_),
        any(g["after_first_fast_share"] is not None and g["after_first_fast_share"] <= 0.5 for g in long_))

    o3 = by_key(raw["o3"]["ms"], raw["o3"]["mode"])
    f = {m: slow_share(x) for m, x in o3.items()}
    out["o3"] = {m: {"slow_share": f[m], "median_ms": statistics.median(x), "n": len(x)} for m, x in o3.items()}
    out["o3_unavailable"] = raw["o3"].get("unavailable", {})
    put_modes(v, "O-7", f, ("none", "memcpy"), lambda d: d["none"] >= 0.8 and d["memcpy"] >= 0.8, lambda d: d["none"] <= 0.2 or d["memcpy"] <= 0.2)
    for key, mode in (("O-8", "clflushopt"), ("O-9", "nt"), ("O-10", "clwb")):
        put_modes(v, key, f, (mode,), lambda d, m=mode: d[m] <= 0.2, lambda d, m=mode: d[m] >= 0.8)
    if f.get("clflushopt") is None or f.get("nt") is None:
        v["O-14"] = "not testable"
    else:
        dirty = f["clflushopt"] <= 0.2 and f["nt"] <= 0.2
        page = f["clflushopt"] >= 0.8 and f["nt"] >= 0.8
        split = (f["clflushopt"] <= 0.2 and f["nt"] >= 0.8) or (f["clflushopt"] >= 0.8 and f["nt"] <= 0.2)
        put(v, "O-14", dirty or page, split)

    o4 = by_key(raw["o4"]["ms"], raw["o4"]["mode"])
    r = {m: slow_share(x) for m, x in o4.items()}
    reads = by_key(raw["o4"]["read_ms"], raw["o4"]["mode"])
    out["o4"] = {m: {"slow_share": r[m], "median_ms": statistics.median(x), "read_ms_median": statistics.median(reads[m]), "n": len(x)}
                 for m, x in o4.items()}
    out["o4_unavailable"] = raw["o4"].get("unavailable", {})
    put_modes(v, "O-11", r, ("none", "buffered", "direct_wait"), lambda d: d["none"] <= 0.1 and d["direct_wait"] <= 0.2 and d["buffered"] >= 0.8,
              lambda d: d["none"] >= 0.5 or d["buffered"] <= 0.2)
    put_modes(v, "O-12", r, ("direct",), lambda d: d["direct"] <= 0.2, lambda d: d["direct"] >= 0.8)

    ratios = {}
    for name, ms in (("k4", [x for k in k4s for x in k["raw"]["alloc_control"]["ms"]]), ("o1", raw["o1"]["ms"]),
                     ("o2", [x for t in raw["o2"]["trains"] for x in t["ms"]]), ("o3", raw["o3"]["ms"]), ("o4", raw["o4"]["ms"])):
        slow = [x for x in ms if x >= bp.SLOW_MS_9_95MB]
        if slow and fast_median:
            ratios[name] = statistics.median(slow) / fast_median
    out["slow_over_fast"] = ratios
    put(v, "O-13", bool(ratios) and all(1.8 <= x <= 2.2 for x in ratios.values()), any(x < 1.6 or x > 2.4 for x in ratios.values()))

    gpu_s = sum(e or 0 for e in out["elapsed_s"])
    clean = (all(st == "complete" for st in out["status"]) and failed_attempts == 0
             and all(g and all(x for k, x in g.items() if k != "failed_cells") and not g.get("failed_cells") for g in out["gates"]))
    every_mode = not out["o3_unavailable"] and not out["o4_unavailable"]
    out["gpu_s"] = round(gpu_s, 3)
    put(v, "O-15", clean and every_mode and gpu_s < 60, not clean)
    out["verdict"] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flush_dir")
    ap.add_argument("--failed-attempts", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    out = score(load_runs(args.flush_dir), args.failed_attempts)
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
