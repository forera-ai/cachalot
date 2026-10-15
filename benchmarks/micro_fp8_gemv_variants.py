"""
L4(f) step 3: where does the `fp8_linear` time of `wo_b` and `wq_b` go, and does an arithmetic E4M3 decode beat the table lookup?

Experiment record: docs/L4F-ATTN-FP8-GEMV-RECORD.md (prediction and stop rule written before this ran).

Per shape (random buffers, `--layers` distinct layers chained in one `mx.eval`, best and median of `--reps`):
  full   fp8_linear as shipped (quantize kernel + v2 GEMV + bf16 cast + sum)
  quant  the quantize kernel alone, chained
  gemv   the shipped v2 GEMV on a pre-quantized activation (+ bf16 cast + sum)
  sum    mx.sum over the same bytes (ceiling)
  var    prototype GEMV variants, same cast and sum; each is compared bit for bit with the shipped GEMV first and skipped if different

Run alone, under the guard:
  cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 4 --max-seconds 900 --tag fp8-gemv-variants -- \\
      ~/venvs/deepseek-v41/bin/python benchmarks/micro_fp8_gemv_variants.py --json benchmarks/results/l4f/fp8-gemv-variants-A.json
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

from cachalot.model.fp8_fused_metal import (  # noqa: E402
    E4M3_TABLE_HEADER,
    BLOCK_SIZE,
    _lanes_per_row,
    fp8_gemv_decoded,
    quantize_activation_fp8_fused,
)
from cachalot.model.fp8_linear_metal import fp8_linear  # noqa: E402
from cachalot.model.kernel_consts import u32  # noqa: E402

PEAK_GBPS = 819.0
BLOCK = 32

# E4M3 byte -> float without a table: the low seven bits shifted into a half's exponent/mantissa give 2^(e-15)(1+m/8)
# (subnormals included, halves keep them), times 2^8 is 2^(e-7)(1+m/8); the sign goes into half bit 15. 0x7F/0xFF
# (the NaN code) decode to 480 this way; the table holds 0.0 there, so `nan_guard` restores it.
DECODE_HEADER = E4M3_TABLE_HEADER + r"""
inline float e4m3_half_decode(uint b) {
    ushort h = ushort(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8));
    return float(as_type<half>(h)) * 256.0f;
}
inline float e4m3_half_decode_guard(uint b) {
    return ((b & 0x7Fu) == 0x7Fu) ? 0.0f : e4m3_half_decode(b);
}
"""


@cache
def variant_kernel(in_features: int, lanes_per_row: int, decode: str):
    """Copy of fp8_fused_metal._gemv_v2_kernel with the byte decode swapped; loop and reduction order unchanged."""
    scale_k = in_features // BLOCK_SIZE
    rows_per_sg = 32 // lanes_per_row
    fn = {"table": "e4m3_table[{}]", "half": "e4m3_half_decode_guard({})", "half-nonan": "e4m3_half_decode({})"}[decode]
    d = lambda expr: fn.format(expr)  # noqa: E731
    source = f"""
        uint global_tid = thread_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = global_tid >> 5;
        uint sub = lane / {lanes_per_row};
        uint l = lane % {lanes_per_row};
        uint row = sg * {rows_per_sg} + sub;
        bool valid = row < n_rows[0];
        uint safe_row = valid ? row : 0;
        const device uint4* wrow = reinterpret_cast<const device uint4*>(weight + safe_row * {in_features});
        uint weight_scale_base = (safe_row >> 5) * {scale_k};
        float acc = 0.0f;
        for (uint block = l; block < {scale_k}; block += {lanes_per_row}) {{
            float act_scale = act_scales[block];
            float weight_scale = metal::exp2(float(int(weight_scales[weight_scale_base + block]) - 127));
            uint4 w0 = wrow[2 * block];
            uint4 w1 = wrow[2 * block + 1];
            uint words[8] = {{w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w}};
            const device float4* av = reinterpret_cast<const device float4*>(act + block * 32);
            float block_acc = 0.0f;
            for (uint wi = 0; wi < 8; ++wi) {{
                float4 a4 = av[wi];
                uint word = words[wi];
                block_acc += a4.x * {d("word & 0xFFu")};
                block_acc += a4.y * {d("(word >> 8) & 0xFFu")};
                block_acc += a4.z * {d("(word >> 16) & 0xFFu")};
                block_acc += a4.w * {d("(word >> 24) & 0xFFu")};
            }}
            acc += block_acc * act_scale * weight_scale;
        }}
        for (uint off = {lanes_per_row // 2}; off > 0; off >>= 1) {{
            acc += simd_shuffle_xor(acc, off);
        }}
        if (valid && l == 0) {{
            out[row] = acc;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"fp8_gemv_var_{decode.replace('-', '_')}_k{in_features}_l{lanes_per_row}",
        input_names=["act", "act_scales", "weight", "weight_scales", "n_rows"],
        output_names=["out"],
        source=source,
        header=DECODE_HEADER,
    )


def gemv_variant(act, act_scales, weight, weight_scales, decode: str, tg: int):
    n, k = weight.shape
    lanes = _lanes_per_row(k // BLOCK_SIZE)
    n_sg = (n + 32 // lanes - 1) // (32 // lanes)
    return variant_kernel(k, lanes, decode)(
        inputs=[act, act_scales, weight, weight_scales, u32(n)],
        template=[],
        grid=(n_sg * 32, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]


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


def bits(a: mx.array):
    return a.view(mx.uint32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    L = args.layers

    shapes = {"wo_b": (5120, 8192), "wq_b": (32768, 1280)}
    rows = []

    def report(shape, arm, nbytes, best, med, note=""):
        gbps = nbytes / (best * 1e-3) / 1e9
        rows.append({"shape": shape, "arm": arm, "bytes": nbytes, "ms_best": best, "ms_median": med, "gbps": gbps, "note": note})
        print(f"{shape:5s} {arm:22s} best {best:7.4f} ms  median {med:7.4f}  {gbps:6.0f} GB/s {gbps / PEAK_GBPS * 100:5.1f} %  {note}", flush=True)

    for name, (n, k) in shapes.items():
        ws = [fp8_weight(n, k) for _ in range(L)]
        nbytes = ws[0][0].nbytes + ws[0][1].nbytes
        x = mx.random.normal(shape=(k,)).astype(mx.bfloat16)
        mx.eval(x)
        q, qs, qd = quantize_activation_fp8_fused(x)
        mx.eval(q, qs, qd)

        def chain(f):
            def run():
                out = None
                for w, s in ws:
                    y = mx.sum(f(w, s))
                    out = y if out is None else out + y
                return out
            return run

        def sum_arm():
            out = None
            for w, s in ws:
                y = mx.sum(w) + mx.sum(s)
                out = y if out is None else out + y
            return out

        report(name, "sum", nbytes, *timed(sum_arm, L, args.reps))
        report(name, "full (fp8_linear)", nbytes, *timed(chain(lambda w, s: fp8_linear(x, w, s)), L, args.reps))

        def quant_arm():
            out = None
            for _ in range(L):
                a, b, c = quantize_activation_fp8_fused(x)
                y = mx.sum(c)
                out = y if out is None else out + y
            return out

        report(name, "quant only", 0, *timed(quant_arm, L, args.reps))
        shipped = lambda w, s: fp8_gemv_decoded(qd, qs, w, s).astype(mx.bfloat16)  # noqa: E731
        report(name, "gemv (shipped)", nbytes, *timed(chain(shipped), L, args.reps))

        for decode, tg in (("table", 256), ("half", 256), ("half-nonan", 256), ("half", 64), ("half", 128), ("half", 512)):
            w0, s0 = ws[0]
            ref = fp8_gemv_decoded(qd, qs, w0, s0)
            got = gemv_variant(qd, qs, w0, s0, decode, tg)
            mx.eval(ref, got)
            same = bool(mx.all(bits(ref) == bits(got)).item())
            if not same:
                nd = int(mx.sum(bits(ref) != bits(got)).item())
                print(f"{name:5s} variant {decode} tg{tg}: NOT bit-identical ({nd} of {n} outputs differ), not timed", flush=True)
                rows.append({"shape": name, "arm": f"var {decode} tg{tg}", "bit_identical": False, "differing": nd})
                continue
            f = lambda w, s, decode=decode, tg=tg: gemv_variant(qd, qs, w, s, decode, tg).astype(mx.bfloat16)  # noqa: E731
            report(name, f"var {decode} tg{tg}", nbytes, *timed(chain(f), L, args.reps), note="bit-identical")
        del ws

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"layers": L, "reps": args.reps, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
