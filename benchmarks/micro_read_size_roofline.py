"""
Screen: what read bandwidth does this GPU give a streaming kernel, as a function of how many bytes one launch reads?

Section 18.75 found the floor's weight-streaming kernels at 29-45 % of the M3 Ultra's 819 GB/s (routed experts
~2.39 GB a token in 10.1 ms, the trunk GEMV family at ~367 GB/s), and section 9.27 that `mx.sum` over the shared
expert's 34 MiB reaches only 385 GB/s. Every decode launch reads one layer's worth of weights (one expert's
projection is ~3 MiB, a layer's six experts ~57 MiB, the shared expert ~34 MiB). If the achievable rate depends on
the bytes a launch reads, the "29-45 %" is the size of the launches, not the kernels.

Two arms per chunk size, each over distinct buffers totalling at least `--total-mib` (so no launch reads what the
last one left in the system-level cache), all chained inside one `mx.eval` so the per-eval drain is paid once:
  sum    `mx.sum` of a bf16 buffer (the simplest possible full read)
  gemv   a bf16 [rows, 4096] matrix times a 4096 vector (a batch-1 dense matvec, the decode shape)
Reports the best of `--reps` timings as GB/s and as a share of 819 GB/s.

No model is loaded. Run alone (one GPU user):
  cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/micro_read_size_roofline.py
"""
from __future__ import annotations

import argparse
import json
from time import perf_counter

import mlx.core as mx

PEAK_GBS = 819.0
COLS = 4096


def time_chain(fn, buffers, reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = perf_counter()
        mx.eval([fn(b) for b in buffers])
        best = min(best, perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-mib", default="1,2,4,8,16,32,64,128,256,512,1024")
    ap.add_argument("--total-mib", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rows = []
    vec = mx.random.normal((COLS,)).astype(mx.bfloat16)
    mx.eval(vec)
    for size_mib in [int(s) for s in args.sizes_mib.split(",")]:
        n = max(2, args.total_mib // size_mib)
        elems = size_mib * 2**20 // 2
        r = max(1, elems // COLS)
        bufs = [mx.random.normal((r, COLS)).astype(mx.bfloat16) for _ in range(n)]
        mx.eval(bufs)
        nbytes = n * r * COLS * 2
        out = {"chunk_mib": size_mib, "launches": n}
        for name, fn in (("sum", lambda b: mx.sum(b)), ("gemv", lambda b: b @ vec)):
            fn(bufs[0]).item() if name == "sum" else mx.eval(fn(bufs[0]))
            s = time_chain(fn, bufs, args.reps)
            out[f"{name}_gbs"] = round(nbytes / s / 1e9, 1)
            out[f"{name}_share"] = round(nbytes / s / 1e9 / PEAK_GBS, 3)
            out[f"{name}_us_per_launch"] = round(1e6 * s / n, 1)
        rows.append(out)
        print(json.dumps(out), flush=True)
        del bufs
        mx.clear_cache()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
