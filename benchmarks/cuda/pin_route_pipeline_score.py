"""
Follow-up P (Mac only; record `docs/CUDA-PIN-PIPELINE-RECORD.md`, predictions P-1 to P-11): the copy period that holds a just-written pinned
buffer slow, and what the dirty-line effect costs a staging pipeline filled from cold sources (memcpy, memcpy plus clflushopt, non-temporal).

Reads the five run directories of `session.sh pinroute6` (one folder, taken in the order of their names, which is `pin_route.ORDER_P`: k4, l1,
p1, p2, k4) and prints each registered quantity beside its interval and falsifier. It judges nothing for the writer: the verdict says only
whether a number is inside the registered interval, past its falsifier, or between; a prediction whose mode the instance could not run is
"not testable". P-2 to P-10 are scored only if the controls separate (P-1). `--failed-attempts N` counts earlier attempts that stopped (P-11).

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/pin_route_pipeline_score.py PIPELINE_DIR [--failed-attempts N] [--json OUT]
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
from pin_route_settle_score import slow_share  # noqa: E402

RAW_NAME = {"k4": "alloc_control", "l1": "batch_first", "p1": "period_train", "p2": "fill_pipeline"}
FILLS = ("memcpy", "flush", "nt")


def threshold_p50(hold: dict) -> float | None:
    """The first period at which the hold share falls to 0.5 or below, linear from the period before it; None if it never does."""
    ps = sorted(hold)
    for i, p in enumerate(ps):
        if hold[p] <= 0.5:
            if i == 0:
                return float(p)
            pp, hp = ps[i - 1], hold[ps[i - 1]]
            return pp + (hp - 0.5) / (hp - hold[p]) * (p - pp)
    return None


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
    out["controls"] = {"k4_slow_share": k4_share, "l1_slow_share": l1_share, "l1_median_ms": statistics.median(l1["ms"]) if l1["ms"] else None}
    sep = len(k4_share) == 2 and all(s is not None and s >= 0.9 for s in k4_share) and l1_share is not None and l1_share <= 0.1
    put(v, "P-1", sep, not sep)
    if not sep:
        out["verdict"] = v
        return out

    trains = [pr.period_summary(t) for t in raw["p1"]["trains"]]
    out["p1"] = trains
    by_p: dict = {}
    for t in trains:
        by_p.setdefault(t["period_ms"], []).append(t)
    hold = {p: sum(t["held"] for t in ts) / len(ts) for p, ts in by_p.items()}
    out["p1_hold_share"] = {str(p): h for p, h in sorted(hold.items())}
    out["p1_realised_period_ms"] = {str(p): statistics.median([t["realised_period_ms"] for t in ts if t["realised_period_ms"] is not None] or [0])
                                    for p, ts in sorted(by_p.items())}
    out["p1_settle_ms"] = {str(p): sorted(t["settle_ms"] for t in ts if t["settle_ms"] is not None) for p, ts in sorted(by_p.items())}
    lo, hi = by_p.get(1.4, []), by_p.get(2.4, [])
    near = lambda p: abs(out["p1_realised_period_ms"].get(str(p), 0) - p) <= 0.05  # noqa: E731
    put(v, "P-2", bool(lo) and bool(hi) and all(t["held"] for t in lo) and not any(t["held"] for t in hi),
        ((any(not t["held"] for t in lo) and near(1.4)) or (any(t["held"] for t in hi) and near(2.4))))
    p50 = threshold_p50(hold)
    out["p1_p50_ms"] = p50
    put(v, "P-3", p50 is not None and 1.6 <= p50 <= 2.2, p50 is None or p50 <= 1.4 or p50 > 2.4)
    mixed = sum(1 for h in hold.values() if 0 < h < 1)
    out["p1_mixed_periods"] = mixed
    put(v, "P-4", mixed <= 1, mixed >= 3)

    summ = pr.pipe_summary(raw["p2"]["passes"])
    out["p2"] = {f"{m}_{k}": s for (m, k), s in sorted(summ.items())}
    out["p2_unavailable"] = raw["p2"].get("unavailable", {})
    ser = {m: summ.get((m, "serial")) for m in ("aged", *FILLS)}
    pip = {m: summ.get((m, "pipelined")) for m in ("aged", *FILLS)}
    put_modes(v, "P-5", ser, ("aged", *FILLS),
              lambda d: d["memcpy"]["copy_slow_share"] >= 0.9 and all(d[m]["copy_fast_share"] >= 0.9 for m in ("flush", "nt", "aged")),
              lambda d: d["memcpy"]["copy_fast_share"] >= 0.5 or d["nt"]["copy_slow_share"] >= 0.5 or d["flush"]["copy_slow_share"] >= 0.5)
    put_modes(v, "P-6", ser, FILLS,
              lambda d: 0.25 <= d["memcpy"]["fill_ms"] <= 0.60 and 0.30 <= d["flush"]["flush_ms"] <= 0.50 and d["nt"]["fill_ms"] <= d["memcpy"]["fill_ms"],
              lambda d: d["memcpy"]["fill_ms"] < 0.15 or d["memcpy"]["fill_ms"] > 1.0 or d["flush"]["flush_ms"] > 0.8)
    put_modes(v, "P-7", pip, ("aged",), lambda d: 0.18 <= d["aged"]["wall_per_record_ms"] <= 0.23, lambda d: d["aged"]["wall_per_record_ms"] > 0.30)
    w = lambda d, m: d[m]["wall_per_record_ms"]  # noqa: E731
    put_modes(v, "P-8", pip, FILLS, lambda d: w(d, "nt") < min(w(d, "memcpy"), w(d, "flush")) and w(d, "flush") > w(d, "memcpy"),
              lambda d: w(d, "nt") >= 1.1 * w(d, "memcpy"))
    model = {}
    for m in FILLS:
        if ser[m] and pip[m]:
            host = (ser[m]["fill_ms"] or 0) + (ser[m]["flush_ms"] or 0)
            pred = max(host, ser[m]["copy_ms"])
            model[m] = {"host_ms": host, "copy_ms": ser[m]["copy_ms"], "predicted_ms": pred, "measured_ms": pip[m]["wall_per_record_ms"],
                        "error": (pip[m]["wall_per_record_ms"] - pred) / pred}
    out["p2_model"] = model
    if len(model) < len(FILLS):
        v["P-9"] = "not testable"
    else:
        put(v, "P-9", all(abs(x["error"]) <= 0.25 for x in model.values()), any(abs(x["error"]) > 0.5 for x in model.values()))
    put_modes(v, "P-10", pip, ("aged", "memcpy"), lambda d: 1.5 <= w(d, "memcpy") / w(d, "aged") <= 2.5, lambda d: w(d, "memcpy") / w(d, "aged") < 1.2)

    gpu_s = sum(e or 0 for e in out["elapsed_s"])
    clean = (all(st == "complete" for st in out["status"]) and failed_attempts == 0
             and all(g and all(x for k, x in g.items() if k != "failed_cells") and not g.get("failed_cells") for g in out["gates"]))
    out["gpu_s"] = round(gpu_s, 3)
    put(v, "P-11", clean and not out["p2_unavailable"] and gpu_s < 60, not clean)
    out["verdict"] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pipeline_dir")
    ap.add_argument("--failed-attempts", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    out = score(load_runs(args.pipeline_dir), args.failed_attempts)
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
