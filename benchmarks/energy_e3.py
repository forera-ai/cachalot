"""E3 analysis: joules a token with every component, per phase of the energy arms. Record: docs/E3-JOULES-TOKEN-RECORD.md.

Reads the logger's per-second rows (benchmarks/energy_logger.py) and the driver's phase clock times (benchmarks/energy_arms_driver.py output).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/energy_e3.py --log benchmarks/results/energy/e3-log.jsonl --driver benchmarks/results/energy/arms-driver.jsonl

Per phase: IOReport watts by component, SMC `PSTR`, the E1b fit's prediction of PSTR and its error, system energy a token (PSTR times seconds a
token) and its split by fit term, store counters, and the package-only (cpu+gpu+ane) energy a token for comparison with 0.62.29.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time

COMPS = ("cpu", "gpu", "ane", "dram", "dcs", "amcc", "display_media")
FIT = {"icpt": 18.4, "cpu": 1.27, "gpu": 1.24, "dram": 0.34, "mem": 1.04}  # HANDOFF 18.120


def clock_to_epoch(day_epoch: float, hms: str) -> float:
    d = time.localtime(day_epoch)
    h, m, s = (int(x) for x in hms.split(":"))
    return time.mktime((d.tm_year, d.tm_mon, d.tm_mday, h, m, s, 0, 0, -1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--skip", type=float, default=3.0, help="seconds dropped at the start of each phase")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.log)]
    ev = [json.loads(l) for l in open(a.driver) if l.startswith("{")]
    phases = [e for e in ev if e.get("event") == "phase"]
    out = {}
    print("phase           s    tok  ms/tok | cpu   gpu  dram  dcs+amcc | PSTR (sd)   fit  err | J/tok sys  pkg(cpu+gpu+ane) | GB/s ssd  misses/tok")
    for p in phases:
        s, e = clock_to_epoch(rows[0]["t"], p["start"]), clock_to_epoch(rows[0]["t"], p["end"])
        rs = [r for r in rows if s + a.skip <= r["t"] <= e]
        if len(rs) < 10:
            print(f"{p['name']:<14} too few rows ({len(rs)})")
            continue
        dt = rs[-1]["t"] - rs[0]["t"]
        w = {c: (rs[-1]["joules"].get(c, 0) - rs[0]["joules"].get(c, 0)) / dt for c in COMPS}
        pv = [r["pstr"] for r in rs if r["pstr"] is not None]
        pstr = st.mean(pv)
        fit = FIT["icpt"] + FIT["cpu"] * w["cpu"] + FIT["gpu"] * w["gpu"] + FIT["dram"] * w["dram"] + FIT["mem"] * (w["dcs"] + w["amcc"])
        tok = p["tokens"]
        spt = (e - s) / tok if tok else None
        srs = [r for r in rs if "stats" in r and r["stats"].get("ssd_bytes_read") is not None]
        sts = [r["stats"] for r in srs]
        ssd = (sts[-1]["ssd_bytes_read"] - sts[0]["ssd_bytes_read"]) / (srs[-1]["t"] - srs[0]["t"]) / 1e9 if len(sts) > 1 else None
        miss = ((sts[-1]["expert_misses"] - sts[0]["expert_misses"]) / max(1, sts[-1]["tokens_generated"] - sts[0]["tokens_generated"])) if len(sts) > 1 else None
        pkg = w["cpu"] + w["gpu"] + w["ane"]
        out[p["name"]] = {"pstr": pstr, "fit": fit, "err": pstr - fit, "spt": spt, "w": w, "pkg": pkg, "tok": tok}
        jt = f"{pstr * spt:6.2f}  {pkg * spt:6.2f}" if spt else "     -       -"
        print(f"{p['name']:<14}{e - s:5.0f} {tok:6d} {1000 * spt if spt else 0:6.0f} | {w['cpu']:4.1f} {w['gpu']:5.1f} {w['dram']:5.1f} {w['dcs'] + w['amcc']:7.1f} | "
              f"{pstr:5.1f} ({st.pstdev(pv):.1f}) {fit:5.1f} {pstr - fit:+5.1f} | {jt} | "
              f"{'-' if ssd is None else f'{ssd:5.2f}'}  {'-' if miss is None else f'{miss:5.1f}'}")
    B, C = out.get("B all-resident"), out.get("C read-bound")
    if B and C:
        print("\nsystem energy a token (PSTR x seconds a token) split by the fit's terms, J:")
        for name, o in (("B all-resident", B), ("C read-bound", C)):
            w, t = o["w"], o["spt"]
            parts = {"baseline": FIT["icpt"], "cpu": FIT["cpu"] * w["cpu"], "gpu": FIT["gpu"] * w["gpu"], "dram": FIT["dram"] * w["dram"],
                     "dcs+amcc": FIT["mem"] * (w["dcs"] + w["amcc"]), "unfitted (SSD, noise)": o["err"]}
            print(f"  {name:<15} total {o['pstr'] * t:5.2f} = " + " + ".join(f"{k} {v * t:.2f}" for k, v in parts.items()))
        print(f"\nratio C over B: system {C['pstr'] * C['spt'] / (B['pstr'] * B['spt']):.2f}, package (cpu+gpu+ane) {C['pkg'] * C['spt'] / (B['pkg'] * B['spt']):.2f}")
        print(f"fit error (PSTR - fit): B {B['err']:+.1f} W, C {C['err']:+.1f} W; (C - B) = {C['err'] - B['err']:+.1f} W is the mean power of whatever the fit does not name, "
              "chiefly the drive reading in C (the fit's noise is 1.3-1.4 W)")
    for name in ("A idle-server", "D idle-server"):
        if name in out:
            print(f"{name}: PSTR {out[name]['pstr']:.1f} W, fit {out[name]['fit']:.1f} W")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
