"""E3b analysis: which Energy Model channels or SMC keys carry the 13-14 W the E1b fit misses under real decode. Record: docs/E3B-REFIT-RECORD.md.

Inputs: the synthetic steps (`residual_test.py --full TAG` -> e1b-TAG.jsonl + .header.json) and the decode arms (`E_FULL=1 energy_arms.sh` ->
e3b-arms.jsonl + .header.json, and the driver's phase clock times in arms-driver.jsonl).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/refit_analyze.py --steps benchmarks/results/energy/e1b-refit.jsonl \\
        --arms benchmarks/results/energy/e3b-arms.jsonl --driver benchmarks/results/energy/arms-driver.jsonl

Prints: per window PSTR, the old fit and its error (P3, P4); the sum of the channels the old aggregates do not use, idle against the
all-resident phase (P1); the families and SMC keys that moved most; a fit on the synthetic steps with an extra regressor, tested on decode.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import power_sources as ps  # noqa: E402

COMPS = ("cpu", "gpu", "ane", "dram", "dcs", "amcc", "display_media")
FIT = {"icpt": 18.4, "cpu": 1.27, "gpu": 1.24, "dram": 0.34, "mem": 1.04}


def load(path, skip_header=False):
    rows = [json.loads(l) for l in open(path)]
    hdr = json.loads(Path(str(path).replace(".jsonl", ".header.json") if not Path(str(path) + ".header.json").exists() else str(path) + ".header.json").read_text())
    return rows, hdr


def clock_epoch(day_epoch, hms):
    d = time.localtime(day_epoch)
    h, m, s = (int(x) for x in hms.split(":"))
    return time.mktime((d.tm_year, d.tm_mon, d.tm_mday, h, m, s, 0, 0, -1))


class Win:
    """One window of rows with per-channel watts, mean PSTR, mean SMC keys."""

    def __init__(self, name, rows, hdr, skip):
        rows = [r for r in rows if r.get("raw")]
        rs = rows[int(skip):]
        self.name, self.ok = name, len(rs) >= 5
        if not self.ok:
            return
        dt = rs[-1]["t"] - rs[0]["t"]
        self.dt, self.t0 = dt, rs[0]["t"]
        self.chan = []
        for (g, n, u, f), a, b in zip(hdr["channels"], rs[0]["raw"], rs[-1]["raw"]):
            sc = ps._UNIT_TO_J.get(u)
            self.chan.append((g, n, (b - a) * sc / dt) if (sc and a is not None and b is not None) else (g, n, None))
        p = [r["pstr"] for r in rs if r["pstr"] is not None]
        self.pstr, self.pstr_sd = st.mean(p), st.pstdev(p)
        sm = [r["smc"] for r in rs if r.get("smc")]
        self.smc = {k: st.mean(s[k] for s in sm if k in s) for k in (sm[0] if sm else {})}
        self.comp = {c: 0.0 for c in COMPS}
        self.unmapped = 0.0
        self.fam = {}
        for g, n, w in self.chan:
            if w is None:
                continue
            cm = ps._component(g, n)
            if cm in self.comp:
                self.comp[cm] += w
            elif cm is None:
                self.unmapped += w
            fam = re.sub(r"\d+", "#", n)
            self.fam[fam] = self.fam.get(fam, 0.0) + w
        c = self.comp
        self.fit = FIT["icpt"] + FIT["cpu"] * c["cpu"] + FIT["gpu"] * c["gpu"] + FIT["dram"] * c["dram"] + FIT["mem"] * (c["dcs"] + c["amcc"])
        self.err = self.pstr - self.fit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--skip", type=float, default=5.0)
    a = ap.parse_args()
    srows, shdr = load(a.steps)
    arows, ahdr = load(a.arms)
    wins = []
    steps = {}
    for r in srows:
        steps.setdefault(r["step"], []).append(r)
    for name, rs in steps.items():
        if name != "warmup":
            wins.append(("syn", Win(name, rs, shdr, a.skip)))
    for e in (json.loads(l) for l in open(a.driver) if l.startswith("{")):
        if e.get("event") == "phase":
            s, t = clock_epoch(arows[0]["t"], e["start"]), clock_epoch(arows[0]["t"], e["end"])
            wins.append(("arm", Win(e["name"], [r for r in arows if s <= r["t"] <= t], ahdr, 3.0)))
            if e["name"].startswith("B"):
                body = [r for r in arows if s + 3 <= r["t"] <= t]
                wins.append(("arm", Win("B first 30 s", body[:30], ahdr, 0)))
                wins.append(("arm", Win("B last 30 s", body[-30:], ahdr, 0)))
    wins = [(k, w) for k, w in wins if w.ok]
    print("window                  PSTR (sd)   fit    err | unmapped W | fan RPM  Tmax")
    for k, w in wins:
        fans = [v for kk, v in w.smc.items() if kk.startswith("F") and kk.endswith("Ac")]
        temps = [v for kk, v in w.smc.items() if kk[0] == "T" and 10 < v < 120]
        print(f"{k}:{w.name:<18} {w.pstr:6.1f} ({w.pstr_sd:4.1f}) {w.fit:6.1f} {w.err:+6.1f} | {w.unmapped:8.1f}   | {max(fans) if fans else 0:6.0f} {max(temps) if temps else 0:5.1f}")
    byname = {f"{k}:{w.name}": w for k, w in wins}
    idle = [w for k, w in wins if w.name in ("A idle-server",) or w.name.endswith(":idle")]
    B = next((w for k, w in wins if w.name == "B all-resident"), None)
    C = next((w for k, w in wins if w.name == "C read-bound"), None)
    syn = [w for k, w in wins if k == "syn"]
    if syn:
        print("\nP3 synthetic steps: max |error| %.1f W over %d steps (E1b rms was 1.3 W)" % (max(abs(w.err) for w in syn), len(syn)))
    f30, l30 = byname.get("arm:B first 30 s"), byname.get("arm:B last 30 s")
    if f30 and l30:
        print(f"P4 first/last 30 s of B: error {f30.err:+.1f} vs {l30.err:+.1f} W (difference {l30.err - f30.err:+.1f})")
    if idle and B:
        iu = st.mean(w.unmapped for w in idle)
        print(f"P1 unmapped channels: idle {iu:.1f} W, B {B.unmapped:.1f} W, rise {B.unmapped - iu:+.1f} W" + (f", C {C.unmapped:.1f} W (rise {C.unmapped - iu:+.1f})" if C else ""))
        print("\nfamilies that moved most from idle to B (W, summed over their channels; unmapped means not in the old aggregates):")
        fams = set(B.fam)
        rows = []
        for f in fams:
            i = st.mean(w.fam.get(f, 0.0) for w in idle)
            rows.append((B.fam[f] - i, f, i, B.fam[f]))
        for d, f, i, b in sorted(rows, reverse=True)[:14]:
            print(f"  {d:+7.2f}  {f:<32} idle {i:6.2f}  B {b:6.2f}")
        print("\nSMC keys that moved most from idle to B (power P* keys, W; and fan/temperature extremes):")
        mv = [(B.smc[k] - st.mean(w.smc.get(k, 0) for w in idle), k) for k in B.smc if k[0] == "P"]
        for d, k in sorted(mv, reverse=True)[:8]:
            print(f"  {d:+7.2f}  {k}  idle {st.mean(w.smc.get(k, 0) for w in idle):6.2f}  B {B.smc[k]:6.2f}")
    # refit on synthetic steps with an extra regressor, test on decode
    dec = [w for w in (B, C) if w]
    if dec and len(syn) >= 10:
        def X(ws, extra):
            return np.array([[1.0, w.comp["cpu"], w.comp["gpu"], w.comp["dram"], w.comp["dcs"] + w.comp["amcc"]] + ([w.unmapped] if extra else []) for w in ws])
        y = np.array([w.pstr for w in syn])
        print("\nfit on synthetic steps, predict decode (error = measured - predicted, W):")
        for extra, label in ((False, "old aggregates"), (True, "+ unmapped channels")):
            co, *_ = np.linalg.lstsq(X(syn, extra), y, rcond=None)
            res = y - X(syn, extra) @ co
            pr = X(dec, extra) @ co
            print(f"  {label:<22} coefficients {np.round(co, 2).tolist()}  train rms {np.sqrt((res ** 2).mean()):.2f}  " +
                  "  ".join(f"{w.name} {w.pstr - p:+.1f}" for w, p in zip(dec, pr)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
