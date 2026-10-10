"""E1b: which component is mis-counted in the system-power residual? Record: docs/E1B-RESIDUAL-RECORD.md.

E1 loaded the machine with one source (a GPU stream), so IOReport's components moved together and SMC `PSTR` minus their sum could not be
attributed. This run loads the CPU alone, the GPU compute alone (no DRAM traffic), a CPU memory copy, the GPU stream, and a mix, then
regresses PSTR on the components. Needs a root `powermetrics` sampling for the whole run (the counters are frozen without it):

    # terminal 1 (Hamed): the sampler, never the `tasks` sampler
    sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 800 -o /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/energy/e1b-sampler.txt
    # terminal 2: this instrument (no sudo)
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ~/venvs/deepseek-v41/bin/python benchmarks/residual_test.py

Output: benchmarks/results/energy/e1b-residual.jsonl and a table on stdout; `--analyze FILE` re-reads a saved run.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dram_calibration as dc  # noqa: E402
import power_sources as ps  # noqa: E402

RES = Path(__file__).resolve().parent / "results" / "energy"
GIB = 1 << 30
COMPS = ("cpu", "gpu", "ane", "dram", "dcs", "amcc", "display_media")
# (name, kind, parameter)
STEPS = [("idle", "idle", 0), ("cpu4", "cpu", 4), ("cpu16", "cpu", 16), ("cpumem8", "cpumem", 8), ("gpumm", "gpumm", 1),
         ("gpustream50", "gpustream", 0.5), ("gpustream100", "gpustream", 1.0), ("gpu100+cpu16", "combo", 16)]


class CpuLoad:
    """n threads of cache-resident np.sin (the ufunc releases the GIL) or a 256 MiB memory copy."""

    def __init__(self, kind: str, n: int):
        self.stop, self.threads, self.bytes = threading.Event(), [], 0
        self.lock = threading.Lock()
        for _ in range(n):
            t = threading.Thread(target=self._sin if kind == "cpu" else self._copy, daemon=True)
            self.threads.append(t)

    def _sin(self):
        x = np.random.rand(65536).astype(np.float32)
        y = np.empty_like(x)
        while not self.stop.is_set():
            for _ in range(20):
                np.sin(x, out=y)

    def _copy(self):
        a = np.ones(256 * (1 << 20) // 4, dtype=np.float32)
        b = np.empty_like(a)
        while not self.stop.is_set():
            np.copyto(b, a)
            with self.lock:
                self.bytes += a.nbytes * 2  # read + write

    def __enter__(self):
        for t in self.threads:
            t.start()
        return self

    def __exit__(self, *e):
        self.stop.set()
        for t in self.threads:
            t.join()


def run_step(kind: str, param, seconds: float, state: dict, mx, bufs, mm) -> None:
    end = time.time() + seconds
    if kind == "idle":
        time.sleep(seconds)
    elif kind in ("cpu", "cpumem"):
        with CpuLoad(kind, int(param)) as c:
            while time.time() < end:
                time.sleep(0.25)
                state["bytes"] = state["base"] + c.bytes
        state["base"] = state["bytes"]
    elif kind == "gpumm":
        a, b = mm
        while time.time() < end:
            r = a
            for _ in range(40):
                r = mx.matmul(r, b)
            mx.eval(r)
    elif kind == "gpustream":
        dc.stream(param, seconds, bufs, state, mx)
    elif kind == "combo":
        with CpuLoad("cpu", int(param)):
            dc.stream(1.0, seconds, bufs, state, mx)


def analyze(rows, skip: float):
    steps: dict[str, list[dict]] = {}
    for r in rows:
        steps.setdefault(r["step"], []).append(r)
    table = []
    for name, rs in steps.items():
        if name == "warmup" or len(rs) < skip + 5:
            continue
        rs = rs[int(skip):]
        dt = rs[-1]["t"] - rs[0]["t"]
        w = {c: (rs[-1]["joules"].get(c, 0) - rs[0]["joules"].get(c, 0)) / dt for c in COMPS}
        p = [r["pstr"] for r in rs if r["pstr"] is not None]
        table.append({"step": name, "w": w, "gbs": (rs[-1]["bytes"] - rs[0]["bytes"]) / dt / 1e9,
                      "pstr": st.mean(p), "pstr_sd": st.pstdev(p) if len(p) > 1 else 0.0})
    print("\nstep           GB/s |   cpu    gpu   dram    dcs   amcc  other |  sum  PSTR (sd) residual")
    for r in table:
        w = r["w"]
        tot = sum(w.values())
        print(f"{r['step']:<14}{r['gbs']:6.1f} | {w['cpu']:5.1f} {w['gpu']:6.1f} {w['dram']:6.1f} {w['dcs']:6.1f} {w['amcc']:6.1f} {w['ane'] + w['display_media']:6.1f} | "
              f"{tot:5.1f} {r['pstr']:5.1f} ({r['pstr_sd']:.1f}) {r['pstr'] - tot:7.1f}")
    # joint fit: PSTR ~ c + b_cpu cpu + b_gpu gpu + b_dram dram + b_mem (dcs+amcc) + b_other (ane+display)
    cols = [("cpu", lambda w: w["cpu"]), ("gpu", lambda w: w["gpu"]), ("dram", lambda w: w["dram"]),
            ("dcs+amcc", lambda w: w["dcs"] + w["amcc"]), ("ane+disp", lambda w: w["ane"] + w["display_media"])]
    X = np.array([[1.0] + [f(r["w"]) for _, f in cols] for r in table])
    y = np.array([r["pstr"] for r in table])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\njoint fit over {len(table)} steps: PSTR = {coef[0]:.1f} W + " + " + ".join(f"{c:.2f} x {n}" for c, (n, _) in zip(coef[1:], cols)) + f"   R^2 {r2:.3f}")
    print("(a coefficient of 1.0 means PSTR rises by one watt per IOReport watt of that component; ane+disp has almost no variation and is poorly determined)")
    # simple: single-source residual change
    idle = [r for r in table if r["step"] == "idle"]
    if idle:
        ir = st.mean(r["pstr"] - sum(r["w"].values()) for r in idle)
        print(f"idle residual {ir:.1f} W; residual change by step:")
        for r in table:
            print(f"  {r['step']:<14} {r['pstr'] - sum(r['w'].values()) - ir:+6.1f} W")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-seconds", type=float, default=30.0)
    ap.add_argument("--skip", type=float, default=5.0)
    ap.add_argument("--no-check", action="store_true", help="smoke test only")
    ap.add_argument("--full", metavar="TAG", help="log every Energy Model channel and the SMC keys; outputs e1b-TAG.jsonl and e1b-TAG.header.json")
    ap.add_argument("--analyze", metavar="JSONL")
    a = ap.parse_args()
    if a.analyze:
        analyze([json.loads(l) for l in open(a.analyze)], a.skip)
        return 0
    import mlx.core as mx
    mx.set_cache_limit(0)
    bufs = [mx.ones((GIB // 2,), dtype=mx.bfloat16) for _ in range(4)]
    mm = (mx.random.normal((1024, 1024)).astype(mx.bfloat16), (mx.random.normal((1024, 1024)) * 0.03).astype(mx.bfloat16))
    mx.eval(bufs, mm)
    probe = ps.IOReport("Energy Model")
    j0 = dc.joules_by_component(probe)
    time.sleep(2.0)
    j1 = dc.joules_by_component(probe)
    if j1.get("dram", 0.0) == j0.get("dram", 0.0) and not a.no_check:
        print("STOP: the DRAM counter did not move in 2 s. Start the root powermetrics sampler first (command in this file's docstring).")
        return 2
    state = {"step": "warmup", "bytes": 0, "base": 0}
    lg = dc.Logger(state, RES / f"e1b-{a.full}.header.json" if a.full else None)
    lg.start()
    mx.eval([mx.sum(b) for b in bufs])
    seq = [("A", STEPS), ("B", list(reversed(STEPS)))]
    total = 2 * len(STEPS) * a.step_seconds
    t0 = time.time()
    for pn, steps in seq:
        for name, kind, param in steps:
            state["step"] = f"{pn}:{name}"
            print(f"{time.strftime('%H:%M:%S')} step {state['step']}  (~{max(0, total - (time.time() - t0)) / 60:.1f} min left)", flush=True)
            run_step(kind, param, a.step_seconds, state, mx, bufs, mm)
    lg.stop.set()
    lg.join()
    out = RES / (f"e1b-{a.full}.jsonl" if a.full else "e1b-residual-smoke.jsonl" if a.no_check else "e1b-residual.jsonl")
    with open(out, "w") as f:
        for r in lg.rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out} ({len(lg.rows)} rows)")
    # strip the pass prefix so A and B of a step are averaged only through the fit, not merged: keep them separate rows
    analyze(lg.rows, a.skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
