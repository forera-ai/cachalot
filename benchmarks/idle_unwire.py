"""Idle-gap un-wire microbench: first read pass after an idle gap, with and without keep-wired mechanisms.

Usage: PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/idle_unwire.py ARM [GIB]   (HANDOFF section 18.223)
  ARM: none | heartbeat | knob | both   (knob/both need MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS set by the caller)
Prints one JSON line per gap.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

import mlx.core as mx

ARM = sys.argv[1]
GIB = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
BUF_MIB = 8
GAPS = (3, 10, 30)


def wired_gib():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    wired = int(re.search(r"Pages wired down:\s+(\d+)", out).group(1))
    comp = int(re.search(r"Pages occupied by compressor:\s+(\d+)", out).group(1))
    return wired * page / 2**30, comp * page / 2**30


mx.set_wired_limit(int((GIB + 4) * 2**30))
n = int(GIB * 1024 / BUF_MIB)
elems = BUF_MIB * 2**20 // 2
bufs = []
for i in range(n):
    bufs.append(mx.full((elems,), float(i % 7), dtype=mx.bfloat16))
    if i % 256 == 255:
        mx.eval(bufs[-256:])
mx.eval(bufs)


def read_pass():
    t = time.perf_counter()
    parts = [mx.sum(b) for b in bufs]
    mx.eval(parts)
    return time.perf_counter() - t


last_work = [time.monotonic()]
stop = threading.Event()


def heartbeat():
    one = mx.array(1.0)
    while not stop.is_set():
        time.sleep(0.5)
        mx.eval(one + 1)


if ARM in ("heartbeat", "both"):
    threading.Thread(target=heartbeat, daemon=True).start()

warm = min(read_pass() for _ in range(3))
w0, c0 = wired_gib()
print(json.dumps({"arm": ARM, "mlx": mx.__version__, "knob_ms": os.environ.get("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", "0"),
                  "gib": GIB, "warm_s": round(warm, 4), "wired_gib": round(w0, 1), "compressor_gib": round(c0, 1)}), flush=True)
for gap in GAPS:
    read_pass()
    time.sleep(gap)
    w1, c1 = wired_gib()
    first = read_pass()
    second = read_pass()
    print(json.dumps({"arm": ARM, "gap_s": gap, "first_s": round(first, 4), "second_s": round(second, 4),
                      "wired_before_gib": round(w1, 1), "compressor_before_gib": round(c1, 1)}), flush=True)
stop.set()
