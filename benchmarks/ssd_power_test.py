"""E2: what a drive costs in system power when it reads (power-accounting plan step E2). Record: docs/E2-SSD-POWER-RECORD.md.

Reads 8 MiB blocks with F_NOCACHE from large files on one drive, duty-cycled per second (the drive busy for duty seconds of each second), in
a sequence that alternates idle and load steps. Each block is read at most once in the whole run (a shuffled list used without replacement),
so no read can come from the page cache. A user-level reader logs the IOReport energy counters and SMC `PSTR` each second; each load step
is compared with the mean of its two neighbouring idle steps, and the change in PSTR is corrected with the E1b coefficients
(PSTR = c + 1.27 cpu + 1.24 gpu + 0.34 dram + 1.04 (dcs + amcc), HANDOFF 18.120) for the change in the IOReport components.

Needs a root `powermetrics` sampling for the whole run (the counters are frozen without it):

    # terminal 1 (Hamed): the sampler, never the `tasks` sampler
    sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 700 -o /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/energy/e2-sampler.txt
    # terminal 2: this instrument (no sudo)
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ~/venvs/deepseek-v41/bin/python benchmarks/ssd_power_test.py

Output: benchmarks/results/energy/e2-<label>.jsonl and a table on stdout; `--analyze FILE` re-reads a saved run.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import statistics as st
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dram_calibration as dc  # noqa: E402
import power_sources as ps  # noqa: E402

RES = Path(__file__).resolve().parent / "results" / "energy"
BLOCK = 8 << 20
COMPS = ("cpu", "gpu", "ane", "dram", "dcs", "amcc", "display_media")
COEF = {"cpu": 1.27, "gpu": 1.24, "dram": 0.34, "dcs": 1.04, "amcc": 1.04, "ane": 0.0, "display_media": 0.0}  # E1b fit (dcs+amcc share one 1.04)
SEQ = [0, 0.25, 0, 0.5, 0, 1.0, 0, 1.0, 0, 0.5, 0, 0.25, 0]
F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)


class Blocks:
    """A shuffled list of (path, offset) used without replacement; fd opened with F_NOCACHE per file."""

    def __init__(self, root: str, pattern: str, seed: int, wrap: bool = False):
        files = sorted(Path(root).glob(pattern))
        if not files:
            raise SystemExit(f"no files match {root}/{pattern}")
        self.fds, self.items = {}, []
        for f in files:
            fd = os.open(f, os.O_RDONLY)
            fcntl.fcntl(fd, F_NOCACHE, 1)
            self.fds[str(f)] = fd
            n = os.fstat(fd).st_size // BLOCK
            self.items += [(str(f), i * BLOCK) for i in range(n)]
        random.Random(seed).shuffle(self.items)
        self.next, self.lock, self.wrap = 0, threading.Lock(), wrap

    def take(self):
        with self.lock:
            if self.next >= len(self.items):
                if not self.wrap:
                    raise SystemExit("ran out of unread blocks (the run would repeat reads and risk cache hits)")
                self.next = 0  # --wrap: a block repeats only after the whole list (>> RAM) was read
            it = self.items[self.next]
            self.next += 1
        return it


def burst(blocks: Blocks, until: float, state: dict, buf_size=BLOCK) -> None:
    while time.time() < until:
        path, off = blocks.take()
        n = len(os.pread(blocks.fds[path], buf_size, off))
        with state["lock"]:
            state["bytes"] += n


def run_step(duty: float, seconds: float, blocks: Blocks, state: dict, threads: int, period: float = 1.0) -> None:
    end = time.time() + seconds
    while time.time() < end:
        w0 = time.time()
        if duty > 0:
            busy_until = w0 + duty * period
            ts = [threading.Thread(target=burst, args=(blocks, busy_until, state)) for _ in range(threads)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        rest = period - (time.time() - w0)
        if rest > 0:
            time.sleep(rest)


def step_stats(rows, skip):
    steps: dict[str, list[dict]] = {}
    for r in rows:
        steps.setdefault(r["step"], []).append(r)
    out = []
    for name in sorted((k for k in steps if k != "warmup"), key=lambda k: int(k.split(":")[0])):
        rs = steps[name][int(skip):]
        if len(rs) < 3:
            continue
        dt = rs[-1]["t"] - rs[0]["t"]
        w = {c: (rs[-1]["joules"].get(c, 0) - rs[0]["joules"].get(c, 0)) / dt for c in COMPS}
        p = [r["pstr"] for r in rs if r["pstr"] is not None]
        out.append({"step": name, "duty": float(name.split(":")[1]), "gbs": (rs[-1]["bytes"] - rs[0]["bytes"]) / dt / 1e9, "w": w,
                    "pstr": st.mean(p), "pstr_sd": st.pstdev(p), "n": len(p)})
    return out


def analyze(rows, skip, label=""):
    S = step_stats(rows, skip)
    print(f"\nstep         duty   GB/s |   cpu    gpu   dram  dcs+amcc |  PSTR (sd)   dPSTR  corrected  (n)")
    res = []
    for i, s in enumerate(S):
        if s["duty"] == 0:
            continue
        nb = [S[j] for j in (i - 1, i + 1) if 0 <= j < len(S) and S[j]["duty"] == 0]
        if not nb:
            continue
        base_p = st.mean(b["pstr"] for b in nb)
        dP = s["pstr"] - base_p
        corr = dP - sum(COEF[c] * (s["w"][c] - st.mean(b["w"][c] for b in nb)) for c in COMPS)
        se = (s["pstr_sd"] ** 2 / s["n"] + sum(b["pstr_sd"] ** 2 / b["n"] for b in nb) / len(nb) ** 2) ** 0.5
        res.append({"step": s["step"], "duty": s["duty"], "gbs": s["gbs"], "dP": dP, "corr": corr, "se": se})
        w = s["w"]
        print(f"{s['step']:<12} {s['duty']:4.2f} {s['gbs']:6.2f} | {w['cpu']:5.1f} {w['gpu']:6.1f} {w['dram']:6.1f} {w['dcs'] + w['amcc']:8.1f} | "
              f"{s['pstr']:5.1f} ({s['pstr_sd']:.1f}) {dP:+6.2f} {corr:+8.2f}  (se {se:.2f})")
    print("\nper level (mean of repeats):")
    by = {}
    for r in res:
        by.setdefault(r["duty"], []).append(r)
    for d in sorted(by):
        rs = by[d]
        m = st.mean(r["corr"] for r in rs)
        se = (sum(r["se"] ** 2 for r in rs) ** 0.5) / len(rs)
        print(f"  duty {d:4.2f}: {st.mean(r['gbs'] for r in rs):5.2f} GB/s  corrected dPSTR {m:+6.2f} W  (se {se:.2f}, n={len(rs)}; raw {st.mean(r['dP'] for r in rs):+.2f})")
    if 1.0 in by and 0.25 in by:
        full = st.mean(r["corr"] for r in by[1.0])
        q = st.mean(r["corr"] for r in by[0.25])
        print(f"  ratio duty 0.25 / duty 1.0 = {q / full:.2f}" if full else "")
    print(f"  top achieved rate {max(r['gbs'] for r in res):.2f} GB/s (a rate far above the drive's wall is a page-cache hit and voids the run)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q2g128")
    ap.add_argument("--glob", default="*.safetensors")
    ap.add_argument("--label", default="x10pro")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--step-seconds", type=float, default=30.0)
    ap.add_argument("--skip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--period", type=float, default=1.0, help="duty-cycle window in seconds (0.1 gives a steadier rate)")
    ap.add_argument("--wrap", action="store_true", help="cycle the block list instead of stopping when it is exhausted (a bank larger than RAM)")
    ap.add_argument("--seq", default="", help="comma list of duties replacing the default sequence (idle steps inserted between)")
    ap.add_argument("--no-check", action="store_true", help="smoke test only")
    ap.add_argument("--analyze", metavar="JSONL")
    a = ap.parse_args()
    if a.analyze:
        analyze([json.loads(l) for l in open(a.analyze)], a.skip)
        return 0
    blocks = Blocks(a.root, a.glob, a.seed, a.wrap)
    seq = SEQ
    if a.seq:
        ds = [float(x) for x in a.seq.split(',')]
        seq = [0]
        for d in ds:
            seq += [d, 0]
    need = int(sum(seq) * a.step_seconds * 1.3e9 / BLOCK)
    print(f"{len(blocks.items)} blocks of 8 MiB; the run needs about {need} at 1.3 GB/s ({'wrapping' if a.wrap else 'no wrap'})")
    probe = ps.IOReport("Energy Model")
    j0 = dc.joules_by_component(probe)
    time.sleep(2.0)
    j1 = dc.joules_by_component(probe)
    if j1.get("dram", 0.0) == j0.get("dram", 0.0) and not a.no_check:
        print("STOP: the DRAM counter did not move in 2 s. Start the root powermetrics sampler first (command in this file's docstring).")
        return 2
    state = {"step": "warmup", "bytes": 0, "lock": threading.Lock()}
    lg = dc.Logger(state)
    lg.start()
    t0, total = time.time(), len(seq) * a.step_seconds
    for i, d in enumerate(seq):
        state["step"] = f"{i + 1}:{d}"
        print(f"{time.strftime('%H:%M:%S')} step {state['step']}  (~{max(0, total - (time.time() - t0)) / 60:.1f} min left, {blocks.next} blocks read)", flush=True)
        run_step(d, a.step_seconds, blocks, state, a.threads, a.period)
    lg.stop.set()
    lg.join()
    out = RES / f"e2-{a.label}{'-smoke' if a.no_check else ''}.jsonl"
    with open(out, "w") as f:
        for r in lg.rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out} ({len(lg.rows)} rows)")
    analyze(lg.rows, a.skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
