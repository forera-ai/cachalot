"""E3c: does irregular GPU access reproduce the real-decode power excess? Record: docs/E3C-GATHER-RECORD.md.

No model. A 4 GiB bf16 buffer is read with `mx.sum` over slices: sequential 9.5 MiB blocks (the reference), random 9.5 MiB blocks (an expert's
size), random 1 MiB blocks, and groups of 8 random blocks each evaluated before the next (decode's per-layer synchronization). Every Energy Model channel
and the SMC keys are logged once a second; the analysis reports, per step, the E1b fit's error against SMC `PSTR` and the residuals of the rails
`PVCC`, `PSVR`, `PMVR` against a fit on the sequential and idle steps. Needs a root `powermetrics` sampling beside it:

    # terminal 1 (Hamed): never the `tasks` sampler
    sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 700 -o /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/energy/e3c-sampler.txt
    # terminal 2 (no sudo)
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ~/venvs/deepseek-v41/bin/python benchmarks/gather_test.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dram_calibration as dc  # noqa: E402
import power_sources as ps  # noqa: E402

RES = Path(__file__).resolve().parent / "results" / "energy"
GIB = 1 << 30
ELEMS = 2 * GIB // 2  # 2 Gi elements of bf16 = 4 GiB
B95 = int(9.5 * (1 << 20)) // 2  # elements in a 9.5 MiB block
B1 = (1 << 20) // 2
# (name, block elements, order, duty, group size per evaluation)
STEPS = [("idle", 0, "none", 0, 0), ("SEQ25", B95, "seq", 0.25, 64), ("SEQ50", B95, "seq", 0.5, 64), ("SEQ100", B95, "seq", 1.0, 64),
         ("R95-100", B95, "rand", 1.0, 64), ("R95-50", B95, "rand", 0.5, 64), ("R1-100", B1, "rand", 1.0, 64), ("DL-100", B95, "rand", 1.0, 8)]


def run_step(name, blk, order, duty, group, seconds, buf, state, mx, rng):
    if name == "idle":
        time.sleep(seconds)
        return
    nblk = ELEMS // blk
    pos = 0
    end = time.time() + seconds
    while time.time() < end:
        w0 = time.time()
        busy_until = w0 + duty
        while time.time() < busy_until:
            if order == "seq":
                idx = [(pos + i) % nblk for i in range(group)]
                pos = (pos + group) % nblk
            else:
                idx = rng.randint(0, nblk, size=group).tolist()
            r = [mx.sum(buf[i * blk:(i + 1) * blk]) for i in idx]
            mx.eval(r)
            state["bytes"] += group * blk * 2
        rest = 1.0 - (time.time() - w0)
        if rest > 0:
            time.sleep(rest)


def analyze(rows, hdr, skip=5.0):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import refit_analyze as ra
    steps = {}
    for r in rows:
        steps.setdefault(r["step"], []).append(r)
    W = []
    for name, rs in steps.items():
        if name == "warmup":
            continue
        w = ra.Win(name, rs, hdr, skip)
        if w.ok:
            dt = rs[-1]["t"] - rs[int(skip)]["t"]
            w.gbs = (rs[-1]["bytes"] - rs[int(skip)]["bytes"]) / dt / 1e9
            W.append(w)
    kind = lambda w: w.name.split(":", 1)[1]  # noqa: E731
    ref = [w for w in W if kind(w) in ("idle", "SEQ25", "SEQ50", "SEQ100")]
    X = lambda ws: np.array([[1.0, w.comp["gpu"], w.comp["dram"], w.comp["dcs"] + w.comp["amcc"], w.comp["cpu"]] for w in ws])  # noqa: E731
    fits = {}
    for key in ("PSTR", "PVCC", "PSVR", "PMVR"):
        y = np.array([(w.pstr if key == "PSTR" else w.smc.get(key, 0.0)) for w in ref])
        fits[key] = np.linalg.lstsq(X(ref), y, rcond=None)[0]
    print("step          GB/s |  PSTR   old-fit err | rail residuals vs the sequential/idle fit (W): PSTR  PVCC  PSVR  PMVR | fan RPM")
    out = {}
    for w in W:
        res = {}
        for key, co in fits.items():
            val = w.pstr if key == "PSTR" else w.smc.get(key, 0.0)
            res[key] = val - (X([w]) @ co)[0]
        fans = [v for k, v in w.smc.items() if k.startswith("F") and k.endswith("Ac")]
        print(f"{w.name:<13}{w.gbs:6.1f} | {w.pstr:6.1f} {w.err:+9.1f}   |                                           {res['PSTR']:+5.1f} {res['PVCC']:+5.1f} {res['PSVR']:+5.1f} {res['PMVR']:+5.1f} | {max(fans) if fans else 0:5.0f}")
        out.setdefault(kind(w), []).append((w.err, res))
    print("\nmean per condition (old-fit error; reference-fit residuals):")
    for k, v in out.items():
        e = st.mean(x[0] for x in v)
        r = {key: st.mean(x[1][key] for x in v) for key in ("PSTR", "PVCC", "PSVR", "PMVR")}
        print(f"  {k:<8} old-fit err {e:+5.1f}   residuals PSTR {r['PSTR']:+5.1f} PVCC {r['PVCC']:+5.1f} PSVR {r['PSVR']:+5.1f} PMVR {r['PMVR']:+5.1f}")
    seq = [x[0] for k in ("SEQ25", "SEQ50", "SEQ100") for x in out.get(k, [])]
    if seq:
        print(f"\nsequential steps' old-fit error: mean {st.mean(seq):+.1f} W (range {min(seq):+.1f} to {max(seq):+.1f}); decode (E3b): +12.6 / +13.9 W")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-seconds", type=float, default=30.0)
    ap.add_argument("--skip", type=float, default=5.0)
    ap.add_argument("--tag", default="gather")
    ap.add_argument("--no-check", action="store_true", help="smoke test only")
    ap.add_argument("--analyze", metavar="JSONL")
    a = ap.parse_args()
    if a.analyze:
        hdr = json.loads(Path(a.analyze.replace(".jsonl", ".header.json")).read_text())
        analyze([json.loads(l) for l in open(a.analyze)], hdr, a.skip)
        return 0
    import mlx.core as mx
    mx.set_cache_limit(0)
    buf = mx.ones((ELEMS,), dtype=mx.bfloat16)
    mx.eval(buf)
    probe = ps.IOReport("Energy Model")
    j0 = dc.joules_by_component(probe)
    time.sleep(2.0)
    j1 = dc.joules_by_component(probe)
    if j1.get("dram", 0.0) == j0.get("dram", 0.0) and not a.no_check:
        print("STOP: the DRAM counter did not move in 2 s. Start the root powermetrics sampler first (command in this file's docstring).")
        return 2
    hdr_path = RES / f"e3c-{a.tag}{'-smoke' if a.no_check else ''}.header.json"
    state = {"step": "warmup", "bytes": 0}
    lg = dc.Logger(state, hdr_path)
    lg.start()
    mx.eval(mx.sum(buf))
    rng = np.random.RandomState(20261010)
    seq = [("A", STEPS), ("B", list(reversed(STEPS)))]
    total = 2 * len(STEPS) * a.step_seconds
    t0 = time.time()
    for pn, steps in seq:
        for name, blk, order, duty, group in steps:
            state["step"] = f"{pn}:{name}"
            print(f"{time.strftime('%H:%M:%S')} step {state['step']}  (~{max(0, total - (time.time() - t0)) / 60:.1f} min left)", flush=True)
            run_step(name, blk, order, duty, group, a.step_seconds, buf, state, mx, rng)
    lg.stop.set()
    lg.join()
    out = RES / f"e3c-{a.tag}{'-smoke' if a.no_check else ''}.jsonl"
    with open(out, "w") as f:
        for r in lg.rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out} ({len(lg.rows)} rows)")
    analyze(lg.rows, json.loads(hdr_path.read_text()), a.skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
