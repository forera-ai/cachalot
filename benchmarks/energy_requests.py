"""E3g: the server's own energy report for each request against the logger's PSTR integral over the same window (docs/E3G-GLM-MINIMAX-RECORD.md).

Reads the energy driver's `request` events (each carries the server's `usage.cachalot.energy`) and the logger's once-a-second rows. For each request the decode
window is the last `decode_seconds` before the request ended; the logger's system energy over it is the trapezoid of `PSTR`, linearly interpolated at both ends.
Requests with a decode window of at least `--min-window` seconds are scored (median and maximum disagreement); shorter ones are listed, not scored. The component
joules, when the server reported them, are compared with the logger's IOReport differences over the same window.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ~/venvs/deepseek-v41/bin/python benchmarks/energy_requests.py \
        --log benchmarks/results/energy/e3g-glm-e3b-arms.jsonl --driver benchmarks/results/energy/e3g-glm-arms-driver.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics as st

COMPS = ("cpu", "gpu", "ane", "dram", "dcs", "amcc")


def integral(rows, t0, t1):
    """System joules between two epoch times from rows with `t` and `pstr` (None-safe); None when the window leaves the rows."""
    pts = [(r["t"], r["pstr"]) for r in rows if r.get("pstr") is not None]
    if not pts or t0 < pts[0][0] or t1 > pts[-1][0]:
        return None

    def w_at(t):
        for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                return wa + (wb - wa) * (t - ta) / (tb - ta)
        return pts[-1][1]

    knots = [(t0, w_at(t0))] + [p for p in pts if t0 < p[0] < t1] + [(t1, w_at(t1))]
    return sum(0.5 * (a[1] + b[1]) * (b[0] - a[0]) for a, b in zip(knots, knots[1:]))


def comp_delta(rows, t0, t1, comp):
    pts = [(r["t"], r["joules"].get(comp)) for r in rows if r.get("joules") and r["joules"].get(comp) is not None]
    if not pts or t0 < pts[0][0] or t1 > pts[-1][0]:
        return None

    def j_at(t):
        for (ta, ja), (tb, jb) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                return ja + (jb - ja) * (t - ta) / (tb - ta)
        return pts[-1][1]

    return j_at(t1) - j_at(t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--min-window", type=float, default=10.0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.log)]
    reqs = [json.loads(l) for l in open(a.driver) if l.startswith("{") and '"request"' in l]
    scored, comp_diffs = [], []
    print("request end   tokens  decode_s | meter J    logger J   diff  | meter J/tok  logger J/tok | components (meter vs logger, dram / gpu J)")
    for q in reqs:
        e = q["energy"]
        if e is None:
            continue  # an arm with the meter off
        if not e.get("available") or not e.get("decode"):
            print(f"{q['end']:.0f}  {q['tokens']:5d} {q['decode_seconds'] or 0:8.1f} | meter unavailable: {e.get('reason', 'no decode window')}")
            continue
        d = e["decode"]
        t1 = q["end"]
        t0 = t1 - d["seconds"]
        lj = integral(rows, t0, t1)
        if lj is None:
            print(f"{q['end']:.0f}  {q['tokens']:5d} {d['seconds']:8.1f} | the logger has no rows for this window")
            continue
        diff = (d["system_joules"] - lj) / lj
        cs = ""
        if d.get("components_joules"):
            ld, lg = comp_delta(rows, t0, t1, "dram"), comp_delta(rows, t0, t1, "gpu")
            if ld and lg:
                cs = f"{d['components_joules']['dram']:.0f}/{ld:.0f}  {d['components_joules']['gpu']:.0f}/{lg:.0f}"
                comp_diffs += [(d["components_joules"]["dram"] - ld) / ld, (d["components_joules"]["gpu"] - lg) / lg]
        flag = "" if d["seconds"] >= a.min_window else "  (not scored: window under %.0f s)" % a.min_window
        print(f"{q['end']:.0f}  {q['tokens']:5d} {d['seconds']:8.1f} | {d['system_joules']:8.1f} {lj:10.1f} {diff:+6.1%} | {d['system_joules_per_token']:9.3f}  {lj / max(1, q['tokens']):11.3f} | {cs}{flag}")
        if d["seconds"] >= a.min_window:
            scored.append(abs(diff))
    if scored:
        print(f"\nscored requests: {len(scored)}  median |diff| {st.median(scored):.1%}  max {max(scored):.1%}   (prediction: median under 4 %, every one within 8 %)")
    if comp_diffs:
        print(f"component joules (dram, gpu) against the logger: median |diff| {st.median(abs(x) for x in comp_diffs):.1%}  max {max(abs(x) for x in comp_diffs):.1%}   (prediction: within 8 %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
