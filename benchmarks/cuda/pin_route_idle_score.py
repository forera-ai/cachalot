"""
Follow-up Q (Mac only; record `docs/CUDA-PIN-IDLE-RECORD.md`, predictions Q-1 to Q-10): does a continuous host idle of about 2 ms clear a
just-written pinned buffer's dirty lines (q1, q2), and do a deeper or a two-thread staging pipeline help (q3)?

Reads the six run directories of `session.sh pinroute7` (one folder, taken in the order of their names, which is `pin_route.ORDER_Q`: k4, l1,
q1, q2, q3, k4) and prints each registered quantity beside its interval and falsifier. It judges nothing for the writer. Q-2 to Q-9 are scored
only if the controls separate (Q-1). `--failed-attempts N` counts earlier attempts that stopped (Q-10).

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/pin_route_idle_score.py IDLE_DIR [--failed-attempts N] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pin_route as pr  # noqa: E402
from pin_route_flush_score import put_modes  # noqa: E402
from pin_route_order_score import load_runs, put  # noqa: E402
from pin_route_pipeline_score import threshold_p50  # noqa: E402
from pin_route_settle_score import by_key, slow_share  # noqa: E402

RAW_NAME = {"k4": "alloc_control", "l1": "batch_first", "q1": "idle_stretch", "q2": "busy_wait", "q3": "pipe_variants"}
P_NT_WALL_MS = 0.420                        # follow-up P's (nt, 2 buffers, 1 thread) per record, docs/CUDA-PIN-PIPELINE-RESULTS.md


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
    out["controls"] = {"k4_slow_share": k4_share, "l1_slow_share": l1_share}
    sep = len(k4_share) == 2 and all(s is not None and s >= 0.9 for s in k4_share) and l1_share is not None and l1_share <= 0.1
    put(v, "Q-1", sep, not sep)
    if not sep:
        out["verdict"] = v
        return out

    trains = [pr.idle_summary(t) for t in raw["q1"]["trains"]]
    out["q1"] = trains
    by_l: dict = {}
    for t in trains:
        by_l.setdefault(t["sleep_ms"], []).append(t)
    hold = {L: sum(t["held"] for t in ts) / len(ts) for L, ts in by_l.items()}
    out["q1_hold_share"] = {str(L): h for L, h in sorted(hold.items())}
    out["q1_slept_ms"] = {str(L): statistics.median([t["slept_ms_median"] for t in ts]) for L, ts in sorted(by_l.items())}
    out["q1_settle_ms"] = {str(L): sorted(t["settle_ms"] for t in ts if t["settle_ms"] is not None) for L, ts in sorted(by_l.items())}
    short = by_l.get(1.0, []) + by_l.get(1.5, [])
    put(v, "Q-2", bool(short) and all(t["held"] for t in short) and bool(by_l.get(2.5)) and not any(t["held"] for t in by_l[2.5]),
        bool(by_l.get(1.0)) and not any(t["held"] for t in by_l[1.0]))
    l50 = threshold_p50(hold)
    out["q1_l50_ms"] = l50
    put(v, "Q-3", l50 is not None and 1.8 <= l50 <= 2.2, l50 is None or l50 <= 1.5 or l50 > 2.5)

    q2 = by_key(raw["q2"]["ms"], raw["q2"]["mode"])
    b = {m: slow_share(x) for m, x in q2.items()}
    out["q2"] = {m: {"slow_share": b[m], "median_ms": statistics.median(x), "n": len(x)} for m, x in q2.items()}
    put_modes(v, "Q-4", b, ("spin30", "sleep30"), lambda d: d["spin30"] >= 0.8 and d["sleep30"] <= 0.2, lambda d: d["spin30"] <= 0.2)
    put_modes(v, "Q-5", b, ("spin60",), lambda d: d["spin60"] >= 0.8, lambda d: d["spin60"] <= 0.2)

    summ = pr.variants_summary(raw["q3"]["passes"])
    out["q3"] = summ
    out["q3_unavailable"] = raw["q3"].get("unavailable", {})
    w = {k: s["wall_per_record_ms"] for k, s in summ.items()}
    g = lambda k: w.get(k)  # noqa: E731
    base = g("nt_s2_t1")
    if base is None:
        v["Q-6"] = v["Q-7"] = "not testable"
    else:
        put(v, "Q-6", abs(base / P_NT_WALL_MS - 1) <= 0.05, abs(base / P_NT_WALL_MS - 1) > 0.15)
        deep = [g("nt_s3_t1"), g("nt_s4_t1")]
        if None in deep:
            v["Q-7"] = "not testable"
        else:
            put(v, "Q-7", all(abs(x / base - 1) <= 0.10 for x in deep), any(x <= 0.8 * base for x in deep))
    two = g("nt_s2_t2")
    if two is None:
        v["Q-8"] = "not testable"
    else:
        put(v, "Q-8", 0.25 <= two <= 0.36, two >= 0.40)
    mtwo = g("memcpy_s2_t2")
    if two is None or mtwo is None:
        v["Q-9"] = "not testable"
    else:
        put(v, "Q-9", mtwo > two, mtwo <= 0.9 * two)

    gpu_s = sum(e or 0 for e in out["elapsed_s"])
    clean = (all(st == "complete" for st in out["status"]) and failed_attempts == 0
             and all(gt and all(x for k, x in gt.items() if k != "failed_cells") and not gt.get("failed_cells") for gt in out["gates"]))
    out["gpu_s"] = round(gpu_s, 3)
    put(v, "Q-10", clean and not out["q3_unavailable"] and gpu_s < 60, not clean)
    out["verdict"] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("idle_dir")
    ap.add_argument("--failed-attempts", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    out = score(load_runs(args.idle_dir), args.failed_attempts)
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
