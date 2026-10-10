"""E1 of the power-accounting plan: DRAM (and DCS, AMCC) energy against bytes moved. Record: docs/E1-DRAM-CALIBRATION-RECORD.md.

A GPU stream over distinct 1 GiB bf16 buffers (4 GiB in all, so no launch reads what the last one left in the system-level cache) is
duty-cycled to hit target read rates. A user-level reader logs the IOReport "Energy Model" counters and the SMC `PSTR` rail once a
second. The counters only advance while a root `powermetrics` is sampling (HANDOFF 18.118), so the run needs Hamed's sampler up for its
whole length (about 10 minutes) and refuses to start if the DRAM counter does not move.

    # terminal 1 (Hamed): the sampler, never the `tasks` sampler
    sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 700 -o /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/energy/e1-sampler.txt
    # terminal 2: this instrument (no sudo)
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ~/venvs/deepseek-v41/bin/python benchmarks/dram_calibration.py

Output: benchmarks/results/energy/e1-dram.jsonl (one row a second) and a table on stdout: per duty step the achieved GB/s and the mean watts
of each component, the fit of DRAM watts on GB/s (pJ per byte, intercept, R^2), and the PSTR cross-check.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import power_sources as ps  # noqa: E402

GIB = 1 << 30
OUT = Path(__file__).resolve().parent / "results" / "energy" / "e1-dram.jsonl"
SMOKE = Path(__file__).resolve().parent / "results" / "energy" / "e1-dram-smoke.jsonl"
COMPONENTS = ("cpu", "gpu", "ane", "dram", "dcs", "amcc", "display_media")


def joules_by_component(rep: "ps.IOReport") -> dict[str, float]:
    out: dict[str, float] = {}
    for ch in rep.sample():
        cm = ps._component(ch["group"], ch["name"])
        sc = ps._UNIT_TO_J.get(ch["unit"])
        if cm and sc and ch["value"] is not None:
            out[cm] = out.get(cm, 0.0) + ch["value"] * sc
    return out


class Logger(threading.Thread):
    """Once a second: cumulative joules by component, PSTR watts, the stream's cumulative bytes and the current step label."""

    def __init__(self, state: dict, full_header_path: "Path | None" = None):
        super().__init__(daemon=True)
        self.state, self.rows, self.stop = state, [], threading.Event()
        self.rep, self.smc = ps.IOReport("Energy Model"), ps.SMC()
        self.full = None
        if full_header_path is not None:
            self.full = ps.FullSampler()
            Path(full_header_path).write_text(json.dumps(self.full.header()))

    def run(self):
        while not self.stop.is_set():
            t = time.time()
            try:
                pstr = self.smc.read("PSTR")[2]
            except Exception:
                pstr = None
            row = {"t": t, "step": self.state["step"], "bytes": self.state["bytes"], "pstr": pstr, "joules": joules_by_component(self.rep)}
            if self.full is not None:
                row.update(self.full.row())
            self.rows.append(row)
            self.stop.wait(max(0.0, 1.0 - (time.time() - t)))


def stream(duty: float, seconds: float, bufs, state: dict, mx) -> None:
    """For each 1 s window: stream for duty seconds, idle for the rest."""
    end = time.time() + seconds
    while time.time() < end:
        w0 = time.time()
        busy_until = w0 + duty
        while time.time() < busy_until:
            r = [mx.sum(b) for b in bufs]
            mx.eval(r)
            state["bytes"] += sum(b.size * 2 for b in bufs)
        rest = 1.0 - (time.time() - w0)
        if rest > 0:
            time.sleep(rest)


def fit(xs, ys):
    n, mx_, my = len(xs), st.mean(xs), st.mean(ys)
    sxx = sum((x - mx_) ** 2 for x in xs)
    slope = sum((x - mx_) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    icpt = my - slope * mx_
    ss_res = sum((y - (icpt + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return slope, icpt, (1 - ss_res / ss_tot) if ss_tot else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-seconds", type=float, default=30.0)
    ap.add_argument("--skip", type=float, default=5.0, help="seconds of each step excluded from its mean")
    ap.add_argument("--duties", default="0,0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--no-check", action="store_true", help="smoke test only: skip the refusal when the DRAM counter does not move")
    ap.add_argument("--analyze", metavar="JSONL", help="re-analyse a saved run instead of running")
    a = ap.parse_args()
    duties = [float(x) for x in a.duties.split(",")]
    if a.analyze:
        rows = [json.loads(l) for l in open(a.analyze)]
    else:
        import mlx.core as mx
        mx.set_cache_limit(0)
        bufs = [mx.ones((GIB // 2,), dtype=mx.bfloat16) for _ in range(4)]
        mx.eval(bufs)
        probe = ps.IOReport("Energy Model")
        j0 = joules_by_component(probe)
        time.sleep(2.0)
        j1 = joules_by_component(probe)
        if j1.get("dram", 0.0) == j0.get("dram", 0.0) and not a.no_check:
            print("STOP: the DRAM counter did not move in 2 s. Start `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 700 "
                  "-o <file>` first (docs/POWER-ACCOUNTING-PLAN.md section 8).")
            return 2
        state = {"step": "warmup", "bytes": 0}
        lg = Logger(state)
        lg.start()
        mx.eval([mx.sum(b) for b in bufs])
        order = [(i + 1, d) for i, d in enumerate(duties)]
        passes = [("A", order), ("B", list(reversed(order)))]
        t_start = time.time()
        total = 2 * len(duties) * a.step_seconds
        for pname, seq in passes:
            for idx, d in seq:
                state["step"] = f"{pname}{idx}:{d}"
                print(f"{time.strftime('%H:%M:%S')} step {state['step']} duty {d:.2f}  (~{max(0, total - (time.time() - t_start)) / 60:.1f} min left)", flush=True)
                stream(d, a.step_seconds, bufs, state, mx)
        lg.stop.set()
        lg.join()
        rows = lg.rows
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out = SMOKE if a.no_check else OUT
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {out} ({len(rows)} rows)")
    # ---- analysis: per step means, excluding the first `skip` seconds
    steps: dict[str, list[dict]] = {}
    for r in rows:
        steps.setdefault(r["step"], []).append(r)
    table = []
    for name, rs in steps.items():
        if name == "warmup" or len(rs) < a.skip + 5:
            continue
        rs = rs[int(a.skip):]
        dt = rs[-1]["t"] - rs[0]["t"]
        gbs = (rs[-1]["bytes"] - rs[0]["bytes"]) / dt / 1e9
        w = {c: (rs[-1]["joules"].get(c, 0) - rs[0]["joules"].get(c, 0)) / dt for c in COMPONENTS}
        ps_ = [r["pstr"] for r in rs if r["pstr"] is not None]
        table.append({"step": name, "duty": float(name.split(":")[1]), "gbs": gbs, "w": w,
                      "pstr": st.mean(ps_) if ps_ else None, "pstr_sd": st.pstdev(ps_) if len(ps_) > 1 else None})
    table.sort(key=lambda r: (r["gbs"], r["step"]))
    print("\nstep      duty   GB/s |  dram    dcs   amcc    cpu    gpu    ane  disp |  PSTR (sd)")
    for r in table:
        w = r["w"]
        print(f"{r['step']:<8} {r['duty']:5.2f} {r['gbs']:6.1f} | {w['dram']:5.2f} {w['dcs']:6.2f} {w['amcc']:6.2f} {w['cpu']:6.2f} {w['gpu']:6.2f} {w['ane']:6.2f} {w['display_media']:5.2f} | "
              f"{r['pstr']:.1f} ({r['pstr_sd']:.1f})" if r["pstr"] is not None else f"{r['step']} no PSTR")
    xs = [r["gbs"] for r in table]
    print("\nfit of watts on GB/s (slope in pJ per byte = W per GB/s x 1000):")
    for c in ("dram", "dcs", "amcc"):
        s, i, r2 = fit(xs, [r["w"][c] for r in table])
        print(f"  {c:<5} slope {s * 1000:7.2f} pJ/B  intercept {i:6.2f} W  R^2 {r2:.3f}")
    comb = [r["w"]["dram"] + r["w"]["dcs"] + r["w"]["amcc"] for r in table]
    s, i, r2 = fit(xs, comb)
    print(f"  dram+dcs+amcc slope {s * 1000:7.2f} pJ/B  intercept {i:6.2f} W  R^2 {r2:.3f}")
    soc = [r["w"]["cpu"] + r["w"]["gpu"] + r["w"]["ane"] for r in table]
    s, i, r2 = fit(xs, soc)
    print(f"  cpu+gpu+ane slope {s * 1000:7.2f} pJ/B  intercept {i:6.2f} W  R^2 {r2:.3f}   (the GPU doing the streaming)")
    pst = [(x, r["pstr"]) for x, r in zip(xs, table) if r["pstr"] is not None]
    if len(pst) > 3:
        s, i, r2 = fit([p[0] for p in pst], [p[1] for p in pst])
        print(f"  PSTR  slope {s * 1000:7.2f} pJ/B  intercept {i:6.2f} W  R^2 {r2:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
