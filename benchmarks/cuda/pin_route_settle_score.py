"""
Follow-up N (Mac only; record `docs/CUDA-PIN-SETTLE-RECORD.md`, predictions N-1 to N-8): is a just-allocated pinned buffer slow for a time after
allocation (H-time), or while its freshly written bytes are still in the CPU's caches (H-cache)?

Reads the seven run directories of `session.sh pinroute4` (one folder, taken in the order of their names, which is `pin_route.ORDER_N`: k4, l1,
n1, n2, n3, n4, k4) and prints each registered quantity beside its interval and falsifier. It judges nothing for the writer: the verdict says only
whether a number is inside the registered interval, past its falsifier, or between. N-2 to N-7 are scored only if the controls separate (N-1).
`--failed-attempts N` counts earlier attempts of the session that stopped (N-8 counts a failed cell of any attempt).

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/pin_route_settle_score.py SETTLE_DIR [--failed-attempts N] [--json OUT]
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

RAW_NAME = {"k4": "alloc_control", "l1": "batch_first", "n1": "settle_train", "n2": "wait_sweep", "n3": "evict", "n4": "retouch"}


def slow_share(ms: list[float]) -> float | None:
    return bp.share(bp.slow_flags(ms, bp.SLOW_MS_9_95MB)) if ms else None


def by_key(ms: list[float], keys: list) -> dict:
    out: dict = {}
    for x, k in zip(ms, keys, strict=True):
        out.setdefault(k, []).append(x)
    return out


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
    put(v, "N-1", len(k4_share) == 2 and all(s is not None and s >= 0.9 for s in k4_share) and l1_share is not None and l1_share <= 0.1,
        not (len(k4_share) == 2 and all(s is not None and s >= 0.9 for s in k4_share) and l1_share is not None and l1_share <= 0.1))
    if v["N-1"] != "in interval":
        out["verdict"] = v
        return out

    n2 = by_key(raw["n2"]["ms"], raw["n2"]["wait_ms"])
    out["n2"] = {str(t): {"slow_share": slow_share(m), "median_ms": statistics.median(m), "n": len(m)} for t, m in sorted(n2.items())}
    s500, s0 = slow_share(n2.get(500, [])), slow_share(n2.get(0, []))
    put(v, "N-2", s500 is not None and s500 >= 0.8 and s0 is not None and s0 >= 0.8, s500 is not None and s500 <= 0.2)

    n3 = by_key(raw["n3"]["ms"], [e // pr.MIB for e in raw["n3"]["evict_bytes"]])
    fills = by_key(raw["n3"]["fill_ms"], [e // pr.MIB for e in raw["n3"]["evict_bytes"]])
    out["n3"] = {str(e): {"slow_share": slow_share(m), "median_ms": statistics.median(m), "fill_ms_median": statistics.median(fills[e]), "n": len(m)}
                 for e, m in sorted(n3.items())}
    s256, se0 = slow_share(n3.get(256, [])), slow_share(n3.get(0, []))
    fill256 = statistics.median(fills[256]) if 256 in fills else None
    put(v, "N-3", s256 is not None and s256 <= 0.2 and se0 is not None and se0 >= 0.8,
        s256 is not None and s256 >= 0.8 and fill256 is not None and fill256 < 20)

    n4 = by_key(raw["n4"]["ms"], raw["n4"]["mode"])
    out["n4"] = {m: {"slow_share": slow_share(x), "median_ms": statistics.median(x), "n": len(x)} for m, x in n4.items()}
    snone, swrite = slow_share(n4.get("none", [])), slow_share(n4.get("write", []))
    put(v, "N-4", snone is not None and snone <= 0.1 and swrite is not None and swrite >= 0.8, swrite is not None and swrite <= 0.2)

    trains = [pr.train_summary(t) for t in raw["n1"]["trains"]]
    out["n1"] = trains
    stay = sum(1 for t in trains if t["tail_slow_share"] is not None and t["tail_slow_share"] >= 0.5)
    settled = [t["settle_ms"] for t in trains if t["settle_ms"] is not None]
    out["n1_stay_slow"] = stay
    put(v, "N-5", stay >= 5, len(settled) == len(trains) and all(s <= 150 for s in settled) and s500 is not None and s500 >= 0.8)

    slow_medians = {}
    for name, ms in (("k4", [x for r in k4s for x in r["raw"]["alloc_control"]["ms"]]), ("n1", [x for t in raw["n1"]["trains"] for x in t["ms"]]),
                     ("n2", raw["n2"]["ms"]), ("n3", raw["n3"]["ms"]), ("n4", raw["n4"]["ms"])):
        slow = [x for x in ms if x >= bp.SLOW_MS_9_95MB]
        if slow and fast_median:
            slow_medians[name] = statistics.median(slow) / fast_median
    out["slow_over_fast"] = slow_medians
    put(v, "N-6", bool(slow_medians) and all(1.8 <= r <= 2.2 for r in slow_medians.values()),
        any(r < 1.6 or r > 2.4 for r in slow_medians.values()))

    cache_side = {"N-2": s500 is not None and s500 >= 0.8, "N-3": s256 is not None and s256 <= 0.2, "N-4": swrite is not None and swrite >= 0.8}
    time_side = {"N-2": s500 is not None and s500 <= 0.2, "N-3": s256 is not None and s256 >= 0.8, "N-4": swrite is not None and swrite <= 0.2}
    out["sides"] = {"cache": cache_side, "time": time_side}
    together = all(cache_side.values()) or all(time_side.values())
    put(v, "N-7", together, not together)

    gpu_s = sum(e or 0 for e in out["elapsed_s"])
    clean = (all(s == "complete" for s in out["status"]) and all(g and all(x for k, x in g.items() if k != "failed_cells") and not g.get("failed_cells")
                                                               for g in out["gates"]) and failed_attempts == 0)
    out["gpu_s"] = round(gpu_s, 3)
    put(v, "N-8", clean and gpu_s < 60, not clean)
    out["verdict"] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("settle_dir")
    ap.add_argument("--failed-attempts", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    out = score(load_runs(args.settle_dir), args.failed_attempts)
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
