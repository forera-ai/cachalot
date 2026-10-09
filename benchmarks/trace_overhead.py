"""
Price of a per-token critical-path tracer (charter L2, HANDOFF 18.114): a CPU micro-benchmark, no model loaded.

Layer 1 - what one event costs, single thread: perf_counter_ns alone, then a per-thread preallocated column store
(`array('q')`, four stores an event, no lock, no allocation), then the obvious alternatives (a tuple appended to a list, a
dict-per-event) for contrast.

Layer 2 - what a whole token costs: a simulated DeepSeek token (40 layers; per layer a router sync, a dispatch, ~1.1 demand
reads served by an 8-worker pool that sleeps like a pread; ~340 events a token with the tracer on) whose main-thread work is a
fixed number of Python iterations, so any event cost adds to the wall time instead of hiding in a deadline. Tokens alternate
tracer-off / tracer-on in random order (the same code path, one `is not None` test an event when off), with an off/off control
to show the noise floor. Reports the paired difference with a bootstrap 95 % interval.

The micro price does not include the tracer's effect on GPU overlap inside a real token; that is the in-situ check afterwards.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/trace_overhead.py --out benchmarks/results/l2-price
"""
from __future__ import annotations

import argparse
import json
import platform
import queue
import random
import statistics as st
import sys
import threading
import time
from array import array
from pathlib import Path

LAYERS = 40
READS_PER_LAYER = 1.14  # 45.6 reads a token (DS-PRED-USB: reads a token with prediction on)
WORKERS = 8
READ_SLEEP = 0.0004  # a pread of a 9.95 MB record on the internal drive is ~1.5 ms; shorter keeps more worker events inside the token
EVENTS_PER_LAYER_MAIN = 6  # sync start, sync end, dispatch, blocked start, blocked end, one more (submit)
TOKEN_ITERS_PER_LAYER = 20000  # calibrated at start to ~2 ms


class Trace:
    """Per-thread preallocated columns; merged after the request. `emit` is the whole hot path."""

    __slots__ = ("ts", "code", "a", "b", "n", "cap")

    def __init__(self, cap: int = 4096):
        self.ts, self.code, self.a, self.b = (array("q", bytes(8 * cap)) for _ in range(4))
        self.n, self.cap = 0, cap

    def emit(self, code: int, a: int = 0, b: int = 0) -> None:
        i = self.n
        if i < self.cap:
            self.ts[i] = time.perf_counter_ns()
            self.code[i] = code
            self.a[i] = a
            self.b[i] = b
            self.n = i + 1

    def reset(self) -> None:
        self.n = 0


class TupleTrace:
    """The alternative design: a list of tuples per thread (cheaper to write in CPython, more memory)."""

    __slots__ = ("ev", "n", "cap")

    def __init__(self, cap: int = 4096):
        self.ev: list = []
        self.n, self.cap = 0, cap

    def emit(self, code: int, a: int = 0, b: int = 0) -> None:
        self.ev.append((time.perf_counter_ns(), code, a, b))

    def reset(self) -> None:
        self.ev.clear()


def spin(iters: int) -> float:
    x = 1.0
    for _ in range(iters):
        x = x * 1.0000001 + 1e-9
    return x


def calibrate(target_s: float) -> int:
    n = 200000
    while True:
        t0 = time.perf_counter()
        spin(n)
        dt = time.perf_counter() - t0
        if dt > 0.05:
            return max(1000, int(n * target_s / dt))
        n *= 4


def per_event_costs(n: int) -> dict:
    out = {}
    t0 = time.perf_counter_ns()
    for _ in range(n):
        time.perf_counter_ns()
    out["perf_counter_ns alone"] = (time.perf_counter_ns() - t0) / n
    tr = Trace(cap=n + 1)
    t0 = time.perf_counter_ns()
    for i in range(n):
        tr.emit(1, i, i)
    out["column store (shipped design)"] = (time.perf_counter_ns() - t0) / n
    lst: list = []
    t0 = time.perf_counter_ns()
    for i in range(n):
        lst.append((time.perf_counter_ns(), 1, i, i))
    out["tuple appended to a list"] = (time.perf_counter_ns() - t0) / n
    lst2: list = []
    t0 = time.perf_counter_ns()
    for i in range(n):
        lst2.append({"t": time.perf_counter_ns(), "c": 1, "a": i, "b": i})
    out["dict per event"] = (time.perf_counter_ns() - t0) / n
    t0 = time.perf_counter_ns()
    none = None
    for i in range(n):
        if none is not None:
            none.emit(1, i, i)
    out["tracer off (the is-not-None test)"] = (time.perf_counter_ns() - t0) / n
    return out


class Sim:
    def __init__(self, iters: int):
        self.iters = iters
        self.jobs: queue.Queue = queue.Queue()
        self.done: queue.Queue = queue.Queue()
        self.tracer_main: Trace | TupleTrace | None = None
        self.worker_traces = [Trace() for _ in range(WORKERS)]
        self.on = False
        self.threads = [threading.Thread(target=self.worker, args=(i,), daemon=True) for i in range(WORKERS)]
        for t in self.threads:
            t.start()

    def worker(self, wid: int) -> None:
        tr = self.worker_traces[wid]
        while True:
            job = self.jobs.get()
            if job is None:
                return
            on = self.on
            if on:
                tr.emit(11, job)  # read started
            time.sleep(READ_SLEEP)
            if on:
                tr.emit(12, job)  # read completed
            self.done.put(job)

    def token(self, on: bool) -> float:
        self.on = on
        tr = self.tracer_main if on else None
        t0 = time.perf_counter()
        pending = 0
        carry = 0.0
        for layer in range(LAYERS):
            if tr is not None:
                tr.emit(1, layer)  # sync start
            spin(self.iters // 2)
            if tr is not None:
                tr.emit(2, layer)  # sync end
            carry += READS_PER_LAYER
            while carry >= 1.0:
                carry -= 1.0
                if tr is not None:
                    tr.emit(10, layer)  # read submitted
                self.jobs.put(layer)
                pending += 1
            if tr is not None:
                tr.emit(3, layer)  # dispatch
            spin(self.iters // 2)
            if tr is not None:
                tr.emit(4, layer)  # blocked start
            while pending:
                self.done.get()
                pending -= 1
            if tr is not None:
                tr.emit(5, layer)  # blocked end
                tr.emit(6, layer)
        dt = time.perf_counter() - t0
        if on:
            for w in self.worker_traces:
                w.reset()
            self.tracer_main.reset()
        return dt * 1000

    def set_kind(self, kind: str) -> None:
        cls = Trace if kind == "column" else TupleTrace
        self.tracer_main = cls()
        self.worker_traces[:] = [cls() for _ in range(WORKERS)]

    def events_per_token(self) -> int:
        self.tracer_main.reset()
        for w in self.worker_traces:
            w.reset()
        self.token(True)
        # the worker traces are reset inside token(); count from the constants instead
        return EVENTS_PER_LAYER_MAIN * LAYERS + int(2 * LAYERS * READS_PER_LAYER)


def boot_ci(d: list[float], n: int = 4000) -> tuple[float, float]:
    rng = random.Random(0)
    m = [st.mean(rng.choices(d, k=len(d))) for _ in range(n)]
    m.sort()
    return m[int(0.025 * n)], m[int(0.975 * n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=300, help="tokens a condition")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("== layer 1: one event, single thread", flush=True)
    costs = per_event_costs(1_000_000)
    for k, v in costs.items():
        print(f"  {k:40s} {v:8.1f} ns", flush=True)

    iters = calibrate(0.0018)  # spin() is called twice a layer (iters//2 each): ~1.8 ms of work a layer, ~72 ms + waits a token
    sim = Sim(iters)
    sim.set_kind("column")
    for _ in range(10):  # warm up
        sim.token(False)
        sim.token(True)
    ev = EVENTS_PER_LAYER_MAIN * LAYERS + int(2 * LAYERS * READS_PER_LAYER)
    print(f"\n== layer 2: a simulated token, ~{ev} events with the tracer on, {args.tokens} tokens a condition", flush=True)

    def run(label_a: bool, label_b: bool) -> list[float]:
        deltas = []
        for _ in range(args.tokens):
            order = [label_a, label_b]
            random.shuffle(order)
            r = {}
            for flag in order:
                r.setdefault(flag, []).append(sim.token(flag))
            # when both labels equal the dict collapses; run the pair explicitly instead
        return deltas

    def paired(a_on: bool, b_on: bool) -> tuple[list[float], list[float], list[float]]:
        a_t, b_t, d = [], [], []
        for _ in range(args.tokens):
            first_a = random.random() < 0.5
            if first_a:
                ta = sim.token(a_on)
                tb = sim.token(b_on)
            else:
                tb = sim.token(b_on)
                ta = sim.token(a_on)
            a_t.append(ta), b_t.append(tb), d.append(tb - ta)
        return a_t, b_t, d

    res = {"python": platform.python_version(), "events_per_token": ev, "per_event_ns": costs}
    for name, kind, (a_on, b_on) in (("control (off vs off)", "column", (False, False)),
                                    ("column store on vs off", "column", (False, True)),
                                    ("tuple list on vs off", "tuple", (False, True))):
        sim.set_kind(kind)
        a_t, b_t, d = paired(a_on, b_on)
        lo, hi = boot_ci(d)
        med = st.median(d)
        res[name] = {"off_ms": st.mean(a_t), "second_ms": st.mean(b_t), "delta_ms": st.mean(d), "median_delta_ms": med,
                     "ci95_ms": [lo, hi], "sd_ms": st.pstdev(d)}
        print(f"  {name:24s} token {st.mean(a_t):7.2f} / {st.mean(b_t):7.2f} ms  delta mean {st.mean(d):+7.3f} median {med:+7.3f} ms  95% [{lo:+.3f}, {hi:+.3f}]  sd {st.pstdev(d):.3f}", flush=True)
    on = res["column store on vs off"]
    per_event_us = 1000 * on["delta_ms"] / ev
    print(f"\n  tracer overhead {on['delta_ms']:.3f} ms a token = {per_event_us:.2f} us an event over {ev} events; "
          f"{100 * on['delta_ms'] / 80:.2f} % of an 80 ms floor; upper 95 % bound {on['ci95_ms'][1]:.3f} ms", flush=True)
    res["per_event_us_in_token"] = per_event_us
    tu = res["tuple list on vs off"]
    print(f"  tuple list: {tu['delta_ms']:.3f} ms a token ({100 * tu['delta_ms'] / 80:.2f} % of 80 ms), upper bound {tu['ci95_ms'][1]:.3f} ms", flush=True)
    (out / "result.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
