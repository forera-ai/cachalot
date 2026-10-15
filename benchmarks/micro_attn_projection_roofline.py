"""
L4(f) step 2: what rate do the attention projections reach as the runtime issues them, against the machine's own streaming ceiling?

`floor_roofline.py` priced the DeepSeek floor's attention rows with the library kernels' rate tables (LEDGER MC-READ-SIZE-Q). The five
projections of an attention layer run through the project's own kernels at their own shapes, so this measures them directly:

  wq_a   fp8_linear  [1280, 5120] on the layer input             (shipped issues wq_a and wkv as two launches, 0.64.x)
  wkv    fp8_linear  [512, 5120]  on the layer input
  wq_b   fp8_linear  [32768, 1280] on the 1280-wide query latent
  wo_a   bf16 grouped matmul [8, 1024, 4096] x [8, 4096, 1]      (attention_compressed.py `Grouped BF16 wo_a`)
  wo_b   fp8_linear  [5120, 8192] on the 8192-wide low-rank output

Each arm is chained over `--layers` distinct random buffers inside one `mx.eval` (no launch reads what the last one left in the
system-level cache), best and median of `--reps` timings. `sum` is `mx.sum` over the same bytes: the machine's ceiling at that size.
`layer` chains all five projections per layer, in the order the runtime issues them.

No model is loaded. Run alone, under the guard (random buffers of about 7 GiB in all):
  cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 4 --max-seconds 900 --tag attn-roofline -- \\
      ~/venvs/deepseek-v41/bin/python benchmarks/micro_attn_projection_roofline.py --json benchmarks/results/l4f/attn-roofline-A.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cachalot.model.fp8_linear_metal import fp8_linear  # noqa: E402

PEAK_GBPS = 819.0
HIDDEN = 5120
BLOCK = 32
O_GROUPS, O_LORA, GROUP_IN = 8, 1024, 4096


def fp8_weight(rows: int, cols: int):
    w = mx.random.randint(0, 255, shape=(rows, cols)).astype(mx.uint8)
    s = mx.random.randint(120, 131, shape=((rows + BLOCK - 1) // BLOCK, cols // BLOCK)).astype(mx.uint8)
    mx.eval(w, s)
    return w, s


def timed(fn, n_layers: int, reps: int):
    mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append((perf_counter() - t0) / n_layers * 1e3)
    ts.sort()
    return ts[0], statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    L = args.layers

    x = mx.random.normal(shape=(HIDDEN,)).astype(mx.bfloat16)
    qr = mx.random.normal(shape=(1280,)).astype(mx.bfloat16)
    o_low = mx.random.normal(shape=(O_GROUPS * O_LORA,)).astype(mx.bfloat16)
    o_grp = mx.random.normal(shape=(O_GROUPS, GROUP_IN, 1)).astype(mx.bfloat16)
    mx.eval(x, qr, o_low, o_grp)

    shapes = {"wq_a": (1280, HIDDEN), "wkv": (512, HIDDEN), "wq_b": (32768, 1280), "wo_b": (HIDDEN, 8192)}
    fp8 = {name: [fp8_weight(*shape) for _ in range(L)] for name, shape in shapes.items()}
    wo_a = [mx.random.normal(shape=(O_GROUPS, O_LORA, GROUP_IN)).astype(mx.bfloat16) for _ in range(L)]
    mx.eval(wo_a)
    inputs = {"wq_a": x, "wkv": x, "wq_b": qr, "wo_b": mx.random.normal(shape=(8192,)).astype(mx.bfloat16)}
    mx.eval(inputs["wo_b"])

    def sum_over(bufs):
        def run():
            out = None
            for b in bufs:
                y = mx.sum(b[0]) + mx.sum(b[1]) if isinstance(b, tuple) else mx.sum(b)
                out = y if out is None else out + y
            return out
        return run

    results = []

    def report(name, nbytes, best, med, ceiling=None):
        gbps = nbytes / (best * 1e-3) / 1e9
        row = {"arm": name, "bytes": nbytes, "ms_best": best, "ms_median": med, "gbps": gbps, "pct_peak": gbps / PEAK_GBPS * 100}
        if ceiling is not None:
            row["ceiling_gbps"] = ceiling
            row["pct_of_ceiling"] = gbps / ceiling * 100
        results.append(row)
        extra = f"   {gbps / ceiling * 100:5.1f} % of the sum ceiling ({ceiling:.0f} GB/s)" if ceiling else ""
        print(f"{name:14s} {nbytes / 1e6:8.1f} MB/layer  best {best:7.4f} ms  median {med:7.4f}  {gbps:6.0f} GB/s  "
              f"{gbps / PEAK_GBPS * 100:5.1f} % of {PEAK_GBPS:.0f}{extra}", flush=True)

    layer_bytes = 0
    for name in ("wq_a", "wkv", "wq_b", "wo_b"):
        bufs = fp8[name]
        nbytes = sum(a.nbytes for a in bufs[0])

        def run(name=name, bufs=bufs):
            out = None
            for w, s in bufs:
                y = mx.sum(fp8_linear(inputs[name], w, s))
                out = y if out is None else out + y
            return out

        sb, sm = timed(sum_over(bufs), L, args.reps)
        ceiling = nbytes / (sb * 1e-3) / 1e9
        report(name + " sum", nbytes, sb, sm)
        b, m = timed(run, L, args.reps)
        report(name, nbytes, b, m, ceiling)
        layer_bytes += nbytes

    nbytes = wo_a[0].nbytes

    def run_wo_a():
        out = None
        for w in wo_a:
            y = mx.sum(mx.matmul(w, o_grp))
            out = y if out is None else out + y
        return out

    sb, sm = timed(sum_over(wo_a), L, args.reps)
    ceiling = nbytes / (sb * 1e-3) / 1e9
    report("wo_a sum", nbytes, sb, sm)
    b, m = timed(run_wo_a, L, args.reps)
    report("wo_a (bf16)", nbytes, b, m, ceiling)
    layer_bytes += nbytes

    def run_layer():
        out = None
        for i in range(L):
            q_a = fp8_linear(x, *fp8["wq_a"][i])
            kv = fp8_linear(x, *fp8["wkv"][i])
            q_b = fp8_linear(qr, *fp8["wq_b"][i])
            low = mx.matmul(wo_a[i], o_grp)
            o_b = fp8_linear(inputs["wo_b"], *fp8["wo_b"][i])
            y = mx.sum(q_a) + mx.sum(kv) + mx.sum(q_b) + mx.sum(low) + mx.sum(o_b)
            out = y if out is None else out + y
        return out

    b, m = timed(run_layer, L, args.reps)
    report("layer (all 5)", layer_bytes, b, m)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"layers": L, "reps": args.reps, "rows": results}, indent=2))


if __name__ == "__main__":
    main()
