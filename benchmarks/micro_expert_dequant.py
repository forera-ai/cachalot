"""
L4(f) step 5: how much of a 2-bit routed-expert GEMV launch is the dequantization arithmetic?

Experiment record: docs/L4F-EXPERT-DEQUANT-RECORD.md (prediction and stop rule written before this ran).

Per shape (w1: 2304 x 5120, w2: 5120 x 2304; 2-bit, group 128; random packed buffers; `--matrices` distinct
matrices chained in one `mx.eval`, best and median of `--reps`):
  sum    mx.sum over the same bytes (a reading reference, not a ceiling)
  qmm    mx.quantized_matmul as shipped
  dq     custom Metal GEMV with the real dequantization (checked numerically against qmm)
  nodq   the same kernel with the dequantization removed (wrong numerics; same loads and reduction): bounds any dequant lever

Run alone, under the guard:
  cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 4 --max-seconds 900 --tag expert-dequant -- \\
      ~/venvs/deepseek-v41/bin/python benchmarks/micro_expert_dequant.py --json benchmarks/results/l4f/expert-dequant-A.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from functools import cache
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PEAK_GBPS = 819.0
GROUP = 128
BITS = 2
CHUNK = 64  # values per uint4 load (4 words x 16 two-bit values)


@cache
def gemv_kernel(k: int, lanes_per_row: int, dequant: bool):
    """One simdgroup covers 32 / lanes_per_row rows; each lane walks 64-value chunks of its row."""
    chunks = k // CHUNK
    rows_per_sg = 32 // lanes_per_row
    groups = k // GROUP
    if dequant:
        body = """
            float qsum = 0.0f;
            float xsum = 0.0f;
            for (uint wi = 0; wi < 4; ++wi) {
                uint word = wv[wi];
                const device float4* xv4 = reinterpret_cast<const device float4*>(x + chunk * 64 + wi * 16);
                for (uint j = 0; j < 4; ++j) {
                    float4 xv = xv4[j];
                    uint nib = (word >> (8 * j)) & 0xFFu;
                    qsum += xv.x * float(nib & 3u) + xv.y * float((nib >> 2) & 3u)
                          + xv.z * float((nib >> 4) & 3u) + xv.w * float((nib >> 6) & 3u);
                    xsum += xv.x + xv.y + xv.z + xv.w;
                }
            }
            acc += scale * qsum + bias * xsum;
        """
    else:
        body = """
            float qsum = 0.0f;
            for (uint wi = 0; wi < 4; ++wi) {
                uint word = wv[wi];
                const device float4* xv4 = reinterpret_cast<const device float4*>(x + chunk * 64 + wi * 16);
                for (uint j = 0; j < 4; ++j) {
                    float4 xv = xv4[j];
                    uint w4 = (word >> (8 * j));
                    qsum += xv.x + xv.y + xv.z + xv.w;
                    acc += float(w4);
                }
            }
            acc += scale * qsum + bias;
        """
    source = f"""
        uint global_tid = thread_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = global_tid >> 5;
        uint sub = lane / {lanes_per_row};
        uint l = lane % {lanes_per_row};
        uint row = sg * {rows_per_sg} + sub;
        bool valid = row < n_rows[0];
        uint safe_row = valid ? row : 0;
        const device uint4* wrow = reinterpret_cast<const device uint4*>(packed + safe_row * {k // 16});
        float acc = 0.0f;
        for (uint chunk = l; chunk < {chunks}; chunk += {lanes_per_row}) {{
            uint4 w = wrow[chunk];
            uint wv[4] = {{w.x, w.y, w.z, w.w}};
            uint g = chunk >> 1;
            float scale = float(scales[safe_row * {groups} + g]);
            float bias = float(biases[safe_row * {groups} + g]);
            {body}
        }}
        for (uint off = {lanes_per_row // 2}; off > 0; off >>= 1) {{
            acc += simd_shuffle_xor(acc, off);
        }}
        if (valid && l == 0) {{
            out[row] = acc;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"expert_gemv_{'dq' if dequant else 'nodq'}_k{k}_l{lanes_per_row}",
        input_names=["x", "packed", "scales", "biases", "n_rows"],
        output_names=["out"],
        source=source,
    )


def gemv(x32, packed, scales, biases, dequant: bool, lanes: int, tg: int):
    n, kw = packed.shape
    k = kw * 16
    rows_per_sg = 32 // lanes
    n_sg = (n + rows_per_sg - 1) // rows_per_sg
    return gemv_kernel(k, lanes, dequant)(
        inputs=[x32, packed, scales, biases, mx.array([n], dtype=mx.uint32)],
        template=[],
        grid=(n_sg * 32, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]


def make(rows: int, cols: int):
    w = mx.random.normal(shape=(rows, cols)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=GROUP, bits=BITS)
    mx.eval(q, s, b)
    return q, s, b


def packed_like(rows: int, cols: int):
    q = mx.random.randint(0, 2**31 - 1, shape=(rows, cols * BITS // 32)).astype(mx.uint32)
    s = (mx.random.uniform(shape=(rows, cols // GROUP)) * 0.05).astype(mx.bfloat16)
    b = (mx.random.normal(shape=(rows, cols // GROUP)) * 0.05).astype(mx.bfloat16)
    mx.eval(q, s, b)
    return q, s, b


def timed(fn, n: int, reps: int):
    mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append((perf_counter() - t0) / n * 1e3)
    ts.sort()
    return ts[0], statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", type=int, default=240)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    N = args.matrices
    rows_out = []

    def report(shape, arm, nbytes, best, med, note=""):
        gbps = nbytes / (best * 1e-3) / 1e9
        rows_out.append({"shape": shape, "arm": arm, "bytes": nbytes, "ms_best": best, "ms_median": med,
                         "gbps": gbps, "note": note})
        print(f"{shape:3s} {arm:16s} best {best:7.4f} ms  median {med:7.4f}  {gbps:6.0f} GB/s "
              f"{gbps / PEAK_GBPS * 100:5.1f} %  {note}", flush=True)

    for name, (n, k) in {"w1": (2304, 5120), "w2": (5120, 2304)}.items():
        ws = [packed_like(n, k) for _ in range(N)]
        nbytes = sum(a.nbytes for a in ws[0])
        x = mx.random.normal(shape=(k,)).astype(mx.bfloat16)
        x32 = x.astype(mx.float32)
        mx.eval(x, x32)

        def chain(f):
            def run():
                out = None
                for q, s, b in ws:
                    y = mx.sum(f(q, s, b).astype(mx.float32))
                    out = y if out is None else out + y
                return out
            return run

        def sum_arm():
            out = None
            for q, s, b in ws:
                y = mx.sum(q) + mx.sum(s.astype(mx.float32)) + mx.sum(b.astype(mx.float32))
                out = y if out is None else out + y
            return out

        report(name, "sum", nbytes, *timed(sum_arm, N, args.reps))
        qmm = lambda q, s, b: mx.quantized_matmul(x, q, s, b, transpose=True, group_size=GROUP, bits=BITS)  # noqa: E731
        report(name, "qmm (shipped)", nbytes, *timed(chain(qmm), N, args.reps))

        lanes = 16 if k == 5120 else 4
        for tg in (64, 128, 256):
            q0, s0, b0 = ws[0]
            ref = qmm(q0, s0, b0).astype(mx.float32)
            got = gemv(x32, q0, s0, b0, True, lanes, tg)
            mx.eval(ref, got)
            err = float(mx.max(mx.abs(ref - got)).item()) / (float(mx.max(mx.abs(ref)).item()) + 1e-9)
            for label, dq in (("dq", True), ("nodq", False)):
                f = lambda q, s, b, dq=dq, tg=tg: gemv(x32, q, s, b, dq, lanes, tg)  # noqa: E731
                note = f"max rel err vs qmm {err:.1e}" if dq else "numerics removed"
                report(name, f"{label} tg{tg}", nbytes, *timed(chain(f), N, args.reps), note=note)
        del ws

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"matrices": N, "reps": args.reps, "rows": rows_out}, indent=2))


if __name__ == "__main__":
    main()
