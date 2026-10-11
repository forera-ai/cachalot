"""E3h analysis (docs/E3H-METER-RERUN-RECORD.md): the meter's cost on a decode token (the ABBA arms off, on, on, off) and the component tail fix against the logger.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ~/venvs/deepseek-v41/bin/python benchmarks/meter_ab_analyze.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import energy_requests as er  # noqa: E402

D = Path(__file__).resolve().parent / "results" / "energy"
ARMS = ["1-off", "2-on", "3-on", "4-off"]


def load(arm):
    ev = [json.loads(l) for l in open(D / f"e3h-{arm}-arms-driver.jsonl") if l.startswith("{")]
    warm = next(i for i, e in enumerate(ev) if e.get("event") == "warm")
    reqs = [e for e in ev[warm:] if e.get("event") == "request"]
    # the warm-up requests come before the "warm" event; these are the all-resident requests of phase B
    return reqs


def main() -> int:
    per = {}
    for arm in ARMS:
        r = load(arm)
        ms = [1000.0 * q["decode_seconds"] / q["tokens"] for q in r if q["decode_seconds"] and q["tokens"] >= 100]
        per[arm] = ms
        print(f"{arm:<7} requests {len(ms):3d}  ms/token  median {st.median(ms):6.2f}  mean {st.mean(ms):6.2f}  sd {st.pstdev(ms):5.2f}  first-third {st.mean(ms[:len(ms)//3]):6.2f}  last-third {st.mean(ms[-(len(ms)//3):]):6.2f}")
    off = per["1-off"] + per["4-off"]
    on = per["2-on"] + per["3-on"]
    d = st.mean(on) - st.mean(off)
    dm = st.median(on) - st.median(off)
    print(f"\nmeter on minus off: mean {d:+.2f} ms/token ({d / st.mean(off):+.1%}), median {dm:+.2f} ms")
    print(f"drift check (A/A): off1 - off4 {st.mean(per['1-off']) - st.mean(per['4-off']):+.2f} ms;  on2 - on3 {st.mean(per['2-on']) - st.mean(per['3-on']):+.2f} ms")
    pair = [st.mean(per['2-on']) - st.mean(per['1-off']), st.mean(per['3-on']) - st.mean(per['4-off'])]
    print(f"pair differences (on - the neighbouring off): {pair[0]:+.2f} ms, {pair[1]:+.2f} ms")
    print("\nthe tail fix, the meter against the logger (windows of 8 s or more, the two on arms pooled):")
    sysd, compd = [], []
    for arm in ("2-on", "3-on"):
        rows = [json.loads(l) for l in open(D / f"e3h-{arm}-e3b-arms.jsonl")]
        for q in load(arm):
            e = q["energy"]
            if not e or not e.get("available") or not e.get("decode") or e["decode"]["seconds"] < 8.0:
                continue
            t1 = q["end"]; t0 = t1 - e["decode"]["seconds"]
            lj = er.integral(rows, t0, t1)
            if lj is None:
                continue
            sysd.append((e["decode"]["system_joules"] - lj) / lj)
            if e["decode"].get("components_joules"):
                for c in ("dram", "gpu"):
                    lc = er.comp_delta(rows, t0, t1, c)
                    if lc:
                        compd.append((e["decode"]["components_joules"][c] - lc) / lc)
    if sysd:
        print(f"  system:     n {len(sysd)}  median |diff| {st.median(abs(x) for x in sysd):.1%}  max |diff| {max(abs(x) for x in sysd):.1%}")
    if compd:
        print(f"  components: n {len(compd)}  median |diff| {st.median(abs(x) for x in compd):.1%}  signed median {st.median(compd):+.1%}  max |diff| {max(abs(x) for x in compd):.1%}  (before the fix, MiniMax 5 s windows: median 7.3 %, signed -7.9 %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
