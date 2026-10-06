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
  q2     `mx.quantized_matmul` of a 4096 vector by a 2-bit affine, group-128 [rows, 4096] weight (the routed-expert
         bank's format), packed weights plus bf16 scales and biases counted as the bytes read (0.61.1)
  fp8    the trunk's FP8 GEMV (`fp8_linear_quantized`, E4M3 uint8 [rows, 4096] weight with its block scales, the
         activation quantized once outside the timing), weight plus scale bytes counted (0.61.1)
The q2 and fp8 arms read the same bytes a launch as the bf16 arms (rows scale with the bytes a row costs), so the gap
between gemv and q2 / fp8 at one chunk size is the kernel's dequantization, not the launch size.
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
GROUP = 128


def time_chain(fn, buffers, reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = perf_counter()
        mx.eval([fn(b) for b in buffers])
        best = min(best, perf_counter() - t0)
    return best


def _record(out: dict, name: str, seconds: float, nbytes: int, launches: int) -> None:
    out[f"{name}_gbs"] = round(nbytes / seconds / 1e9, 1)
    out[f"{name}_share"] = round(nbytes / seconds / 1e9 / PEAK_GBS, 3)
    out[f"{name}_us_per_launch"] = round(1e6 * seconds / launches, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-mib", default="1,2,4,8,16,32,64,128,256,512,1024")
    ap.add_argument("--total-mib", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--arms", default="sum,gemv,q2,fp8")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rows = []
    vec = mx.random.normal((COLS,)).astype(mx.bfloat16)
    xq2 = vec.reshape(1, COLS)
    qx = None
    if "fp8" in args.arms:
        from cachalot.model.fp8_linear_metal import fp8_linear_quantized, quantize_fp8_activation

        qx = quantize_fp8_activation(vec)
        mx.eval(qx.values, qx.scales, *([qx.decoded] if qx.decoded is not None else []))
    mx.eval(vec)
    arms = args.arms.split(",")
    for size_mib in [int(s) for s in args.sizes_mib.split(",")]:
        n = max(2, args.total_mib // size_mib)
        out = {"chunk_mib": size_mib, "launches": n}
        if "sum" in arms or "gemv" in arms:
            elems = size_mib * 2**20 // 2
            r = max(1, elems // COLS)
            bufs = [mx.random.normal((r, COLS)).astype(mx.bfloat16) for _ in range(n)]
            mx.eval(bufs)
            nbytes = n * r * COLS * 2
            for name, fn in (("sum", lambda b: mx.sum(b)), ("gemv", lambda b: b @ vec)):
                if name not in arms:
                    continue
                mx.eval(fn(bufs[0]))
                _record(out, name, time_chain(fn, bufs, args.reps), nbytes, n)
            del bufs
            mx.clear_cache()
        if "q2" in arms:
            # a 2-bit row is 1,024 bytes of packed weights plus 32 groups x (scale, bias) in bf16
            r = max(GROUP, (size_mib * 2**20 // (COLS // 4 + 4 * COLS // GROUP)) // GROUP * GROUP)
            bufs = []
            for _ in range(n):
                w, sc, bi = mx.quantize(mx.random.normal((r, COLS)).astype(mx.bfloat16), group_size=GROUP, bits=2)
                bufs.append((w, sc, bi))
                mx.eval(w, sc, bi)
            nbytes = sum(a.nbytes for t in bufs for a in t)
            fn = lambda t: mx.quantized_matmul(xq2, *t, transpose=True, group_size=GROUP, bits=2)
            mx.eval(fn(bufs[0]))
            _record(out, "q2", time_chain(fn, bufs, args.reps), nbytes, n)
            out["q2_rows"] = r
            del bufs
            mx.clear_cache()
        if "fp8" in arms:
            # an FP8 row is 4,096 bytes; its block scales are one byte per 32 rows x 32 columns
            r = max(32, (size_mib * 2**20 // COLS) // 32 * 32)
            bufs = []
            for _ in range(n):
                w = mx.random.randint(0, 0x7E, (r, COLS)).astype(mx.uint8)  # finite E4M3 codes only
                sc = mx.full((r // 32, COLS // 32), 127, dtype=mx.uint8)
                mx.eval(w, sc)
                bufs.append((w, sc))
            nbytes = sum(a.nbytes for t in bufs for a in t)
            fn = lambda t: fp8_linear_quantized(qx, *t)
            mx.eval(fn(bufs[0]))
            _record(out, "fp8", time_chain(fn, bufs, args.reps), nbytes, n)
            del bufs
            mx.clear_cache()
        rows.append(out)
        print(json.dumps(out), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
