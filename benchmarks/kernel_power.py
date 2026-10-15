"""E3f (docs/E3F-REAL-KERNELS-RECORD.md): do the model's own quantized kernels, run alone, draw the unexplained decode power?

No model. Distinct random buffers in the model's formats: the trunk's FP8 GEMV weights (`fp8_linear_quantized`, 8 MiB launches, 5.14 GB in all, the
trunk's bytes a token) and the routed experts' 2-bit group-128 weights (`mx.quantized_matmul`, 3 MiB launches, 2.39 GB, the routed experts' bytes a token),
read once per "token" as the real decode reads them: 43 layer groups, each evaluated before the next (the per-layer host synchronization). Conditions, 40 s each,
two passes in opposite order: TOK (both kernels, a layer's trunk then experts, a sync per layer), TOK-80 (TOK padded to 80 ms a token, the decode cadence), TOK-NS (the same, the whole token chained in one evaluation),
FP8 (trunk kernel alone), Q2 (expert kernel alone), idle. The analysis reports each condition's E1b-fit error against `PSTR`; decode through the server and
in-process is +10.9 / +11.5 W. Needs a root `powermetrics` sampling beside it:

    # terminal 1 (Hamed): never the `tasks` sampler
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 700 -o benchmarks/results/energy/e3f-sampler.txt
    # terminal 2 (no sudo)
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/kernel_power.py

    ... benchmarks/kernel_power.py --analyze benchmarks/results/energy/e3f-kernels.jsonl     (re-read a run)
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dram_calibration as dc  # noqa: E402
import power_sources as ps  # noqa: E402

RES = Path(__file__).resolve().parent / "results" / "energy"
COLS, GROUP, LAYERS = 4096, 128, 43
FP8_MIB, Q2_MIB = 8, 3
FP8_PER_LAYER = 15  # 43 x 15 x 8 MiB = 5.04 GiB, the trunk's ~5.14 GB a token
Q2_PER_LAYER = 19   # 43 x 19 x 3 MiB = 2.39 GiB, the routed experts' ~2.39 GB a token
STEPS = ["idle", "TOK", "TOK-80", "TOK-NS", "FP8", "Q2", "BF", "BF-80"]
TOKEN_MS = 80.0  # TOK-80 pads each token to the real decode cadence (about 94 GB/s of weights), the GPU idling in between as it does in decode


def build(mx):
    from cachalot.model.fp8_linear_metal import fp8_linear_quantized, quantize_fp8_activation

    vec = mx.random.normal((COLS,)).astype(mx.bfloat16)
    qx = quantize_fp8_activation(vec)
    mx.eval(vec, qx.values, qx.scales, *([qx.decoded] if qx.decoded is not None else []))
    x2 = vec.reshape(1, COLS)
    r8 = max(32, (FP8_MIB * 2**20 // COLS) // 32 * 32)
    rq = max(GROUP, (Q2_MIB * 2**20 // (COLS // 4 + 4 * COLS // GROUP)) // GROUP * GROUP)
    fp8, q2 = [], []
    for _ in range(LAYERS):
        lf, lq = [], []
        for _ in range(FP8_PER_LAYER):
            w = mx.random.randint(0, 0x7E, (r8, COLS)).astype(mx.uint8)  # finite E4M3 codes only
            sc = mx.full((r8 // 32, COLS // 32), 127, dtype=mx.uint8)
            mx.eval(w, sc)
            lf.append((w, sc))
        for _ in range(Q2_PER_LAYER):
            w, sc, bi = mx.quantize(mx.random.normal((rq, COLS)).astype(mx.bfloat16), group_size=GROUP, bits=2)
            mx.eval(w, sc, bi)
            lq.append((w, sc, bi))
        fp8.append(lf)
        q2.append(lq)
    # bf16 control (docs/E3I-BF16-CONTROL-RECORD.md): dense bf16 matrices with the same bytes per launch and per token as the FP8 and 2-bit
    # buffers above (8 MiB and 3 MiB launches, 15 and 19 a layer), read as GEMVs, so only the weight format differs.
    bf = []
    for _ in range(LAYERS):
        lb = []
        for n_launch, mib in ((FP8_PER_LAYER, FP8_MIB), (Q2_PER_LAYER, Q2_MIB)):
            for _ in range(n_launch):
                w = mx.random.normal((mib * 2**20 // (2 * COLS), COLS)).astype(mx.bfloat16)
                mx.eval(w)
                lb.append(w)
        bf.append(lb)
    f_bf = lambda w: w @ vec  # noqa: E731
    bb = sum(w.nbytes for l in bf for w in l)
    f_fp8 = lambda t: fp8_linear_quantized(qx, *t)  # noqa: E731
    f_q2 = lambda t: mx.quantized_matmul(x2, *t, transpose=True, group_size=GROUP, bits=2)  # noqa: E731
    nb = lambda ls: sum(a.nbytes for l in ls for t in l for a in t)  # noqa: E731
    return fp8, q2, f_fp8, f_q2, nb(fp8), nb(q2), bf, f_bf, bb


def run_step(name, seconds, mx, pool, state):
    fp8, q2, f_fp8, f_q2, b8, bq, bf, f_bf, bb = pool
    if name == "idle":
        time.sleep(seconds)
        return
    end = time.time() + seconds
    while time.time() < end:
        if name in ("BF", "BF-80"):
            tok0 = time.time()
            for i in range(LAYERS):
                mx.eval([f_bf(w) for w in bf[i]])
            state["bytes"] += bb
            if name == "BF-80":
                rest = TOKEN_MS / 1000 - (time.time() - tok0)
                if rest > 0:
                    time.sleep(rest)
            continue
        if name == "TOK-NS":
            mx.eval([f_fp8(t) for l in fp8 for t in l] + [f_q2(t) for l in q2 for t in l])
            state["bytes"] += b8 + bq
            continue
        tok0 = time.time()
        for i in range(LAYERS):  # one token, a synchronization per layer
            if name in ("TOK", "TOK-80", "FP8"):
                outs = [f_fp8(t) for t in fp8[i]]
            else:
                outs = []
            if name in ("TOK", "TOK-80", "Q2"):
                outs += [f_q2(t) for t in q2[i]]
            mx.eval(outs)
        state["bytes"] += (b8 if name in ("TOK", "TOK-80", "FP8") else 0) + (bq if name in ("TOK", "TOK-80", "Q2") else 0)
        if name == "TOK-80":
            rest = TOKEN_MS / 1000 - (time.time() - tok0)
            if rest > 0:
                time.sleep(rest)


def analyze(rows, hdr, skip=5.0):
    import refit_analyze as ra
    steps = {}
    for r in rows:
        steps.setdefault(r["step"], []).append(r)
    out = {}
    print("step            GB/s |  PSTR   fit-err | cpu   gpu  dram  dcs+amcc | fan RPM")
    for name, rs in steps.items():
        if name == "warmup":
            continue
        w = ra.Win(name, rs, hdr, skip)
        if not w.ok:
            continue
        gbs = (rs[-1]["bytes"] - rs[int(skip)]["bytes"]) / (rs[-1]["t"] - rs[int(skip)]["t"]) / 1e9
        fans = [v for k, v in w.smc.items() if k.startswith("F") and k.endswith("Ac")]
        c = w.comp
        print(f"{name:<14}{gbs:6.1f} | {w.pstr:6.1f} {w.err:+8.1f} | {c['cpu']:4.1f} {c['gpu']:5.1f} {c['dram']:5.1f} {c['dcs'] + c['amcc']:7.1f} | {max(fans) if fans else 0:5.0f}")
        out.setdefault(name.split(":", 1)[1], []).append((w.err, gbs, w.pstr))
    print("\nmean per condition:")
    for k, v in out.items():
        print(f"  {k:<8} fit error {st.mean(x[0] for x in v):+5.1f} W  ({', '.join(f'{x[0]:+.1f}' for x in v)})  rate {st.mean(x[1] for x in v):6.1f} GB/s  PSTR {st.mean(x[2] for x in v):5.1f} W")
    print("\nbf16 control: the E1b fit was calibrated on bf16 loads; BF-80 is the format control for TOK-80 (same bytes, same cadence)")
    print("\ndecode for comparison (E3d server / E3e in-process): fit error +10.9 / +11.5 W, about 94 GB/s of weights, PSTR 71-73 W")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-seconds", type=float, default=40.0)
    ap.add_argument("--skip", type=float, default=5.0)
    ap.add_argument("--tag", default="kernels")
    ap.add_argument("--no-check", action="store_true", help="smoke test only: short steps, no sampler check")
    ap.add_argument("--analyze", metavar="JSONL")
    a = ap.parse_args()
    if a.analyze:
        analyze([json.loads(l) for l in open(a.analyze)], json.loads(Path(a.analyze.replace(".jsonl", ".header.json")).read_text()), a.skip)
        return 0
    import mlx.core as mx
    mx.set_cache_limit(0)
    t0 = time.time()
    pool = build(mx)
    print(f"buffers built in {time.time() - t0:.0f} s: fp8 {pool[4] / 2**30:.2f} GiB, q2 {pool[5] / 2**30:.2f} GiB per token, bf16 control {pool[8] / 2**30:.2f} GiB", flush=True)
    probe = ps.IOReport("Energy Model")
    j0 = dc.joules_by_component(probe)
    time.sleep(2.0)
    if dc.joules_by_component(probe).get("dram", 0.0) == j0.get("dram", 0.0) and not a.no_check:
        print("STOP: the DRAM counter did not move in 2 s. Start the root powermetrics sampler first (command in this file's docstring).")
        return 2
    sfx = "-smoke" if a.no_check else ""
    hdr_path = RES / f"e3f-{a.tag}{sfx}.header.json"
    state = {"step": "warmup", "bytes": 0}
    lg = dc.Logger(state, hdr_path)
    lg.start()
    for s in ("TOK", "FP8", "Q2", "BF"):
        run_step(s, 3.0, mx, pool, state)  # compile and settle each kernel
    seconds = 8.0 if a.no_check else a.step_seconds
    plan = [("A", STEPS), ("B", list(reversed(STEPS)))]
    total = 2 * len(STEPS) * seconds
    t1 = time.time()
    for pn, steps in plan:
        for s in steps:
            state["step"] = f"{pn}:{s}"
            print(f"{time.strftime('%H:%M:%S')} step {state['step']}  (~{max(0, total - (time.time() - t1)) / 60:.1f} min left)", flush=True)
            run_step(s, seconds, mx, pool, state)
    lg.stop.set()
    lg.join()
    out = RES / f"e3f-{a.tag}{sfx}.jsonl"
    with open(out, "w") as f:
        for r in lg.rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out} ({len(lg.rows)} rows)")
    analyze(lg.rows, json.loads(hdr_path.read_text()), a.skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
