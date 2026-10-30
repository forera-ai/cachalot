"""
The Neural Engine as a second matmul engine (docs/ANE-MATMUL-RECORD.md, AN-1 to AN-8).

Builds fp16 Core ML programs with DeepSeek's trunk shapes and constant weights, places them with
compute_units=CPU_AND_NE, reads each op's device from MLComputePlan, and times them alone, beside the MLX GPU
stand-in, and as a hand-off from MLX. Needs a venv with coremltools (9.0) and mlx; not the runtime venv:

    S=<scratch>; /opt/homebrew/bin/python3.12 -m venv $S/anevenv && $S/anevenv/bin/pip install coremltools==9.0 mlx==0.32.4
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && $S/anevenv/bin/python benchmarks/ane_matmul_bench.py all

Rows go to benchmarks/results/ane-matmul/<utc>.jsonl. Subcommands: build, plan, alone, handoff, contention, all;
`ane-loop` and `gpu-loop` are the contention cells' worker processes.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "results" / "ane-matmul"
MODELS = OUT / "models"

# name: (M, K, N, kind)
SHAPES = {
    "gemm4096": (512, 4096, 4096, "gemm"),
    "w13_gemm": (512, 5120, 4608, "gemm"),
    "qup_gemm": (512, 1280, 32768, "gemm"),
    "w13_gemv": (1, 5120, 4608, "gemv"),
    "w13_gemv_int8": (1, 5120, 4608, "gemv"),
    "tiny": (1, 5120, 0, "tiny"),
    # the hand-off probe: `tiny` (an add) is placed on the CPU by Core ML, so a small linear is the ANE dispatch cost
    "tiny_ane": (1, 5120, 256, "gemv"),
}


def utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def spread(xs: list[float]) -> dict:
    xs = sorted(xs)
    q = statistics.quantiles(xs, n=4) if len(xs) >= 4 else [xs[0], statistics.median(xs), xs[-1]]
    return {"n": len(xs), "median": statistics.median(xs), "p25": q[0], "p75": q[-1], "min": xs[0], "max": xs[-1]}


class Rows:
    def __init__(self, path: Path):
        OUT.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, **r) -> dict:
        r = {"utc": utc(), "macos": platform.mac_ver()[0], **r}
        with self.path.open("a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        return r


# ---------------------------------------------------------------------------------------------- build
def build(names: list[str] | None = None) -> None:
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    MODELS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, (m, k, n, kind) in SHAPES.items():
        if names and name not in names:
            continue
        path = MODELS / f"{name}.mlpackage"
        if path.exists():
            continue
        if kind == "tiny":
            c = rng.standard_normal((1, k)).astype(np.float16)

            @mb.program(input_specs=[mb.TensorSpec(shape=(m, k), dtype=types.fp16)], opset_version=ct.target.macOS15)
            def prog(x):
                return mb.add(x=x, y=c, name="y")  # noqa: B023 - traced at once, inside this iteration
        else:
            w = (rng.standard_normal((n, k)) * 0.02).astype(np.float16)   # linear: weight is (out, in)

            @mb.program(input_specs=[mb.TensorSpec(shape=(m, k), dtype=types.fp16)], opset_version=ct.target.macOS15)
            def prog(x):
                return mb.linear(x=x, weight=w, name="y")  # noqa: B023 - traced at once, inside this iteration
        model = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                           minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_AND_NE)
        if name.endswith("_int8"):
            cfg = ct.optimize.coreml.OptimizationConfig(
                global_config=ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                                                        weight_threshold=1024))
            model = ct.optimize.coreml.linear_quantize_weights(model, config=cfg)
        model.save(str(path))
        print(f"built {name} {m}x{k}x{n}", flush=True)


def load(name: str, units: str = "CPU_AND_NE"):
    import coremltools as ct

    return ct.models.MLModel(str(MODELS / f"{name}.mlpackage"), compute_units=getattr(ct.ComputeUnit, units))


def make_input(name: str):
    import numpy as np

    m, k, _, _ = SHAPES[name]
    return {"x": (np.random.default_rng(1).standard_normal((m, k)) * 0.1).astype(np.float16)}


# ----------------------------------------------------------------------------------------------- plan
def plan(rows: Rows) -> dict:
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    out = {}
    for name in SHAPES:
        model = load(name)
        compiled = model.get_compiled_model_path()
        cp = MLComputePlan.load_from_path(path=compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
        program = cp.model_structure.program
        ops = []
        for fn in program.functions.values():
            for op in fn.block.operations:
                usage = cp.get_compute_device_usage_for_mlprogram_operation(op)
                dev = type(usage.preferred_compute_device).__name__ if usage is not None else None
                if op.operator_name in ("const",):
                    continue
                ops.append({"op": op.operator_name, "device": dev,
                            "supported": [type(d).__name__ for d in usage.supported_compute_devices] if usage else None})
        out[name] = ops
        rows.write(measure="plan", model=name, ops=ops)
        print(name, [(o["op"], o["device"]) for o in ops], flush=True)
    return out


# ---------------------------------------------------------------------------------------------- timing
def time_model(name: str, seconds: float, units: str = "CPU_AND_NE", start_at: float | None = None) -> list[float]:
    model = load(name, units)
    x = make_input(name)
    for _ in range(5):
        model.predict(x)
    if start_at is not None:
        while time.time() < start_at:
            model.predict(x)
    times = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        t = time.perf_counter()
        model.predict(x)
        times.append(time.perf_counter() - t)
    return times


def rate(name: str, sec: float) -> dict:
    m, k, n, kind = SHAPES[name]
    if kind == "gemm":
        return {"tflops": 2 * m * k * n / sec / 1e12}
    if kind == "gemv":
        bytes_w = k * n * (1 if name.endswith("_int8") else 2)
        return {"weight_gb_s": bytes_w / sec / 1e9}
    return {}


def alone(rows: Rows, seconds: float) -> dict:
    out = {}
    for name in SHAPES:
        for units in ("CPU_AND_NE",) if name != "w13_gemv" else ("CPU_AND_NE", "CPU_ONLY"):
            t = time_model(name, seconds, units)
            s = spread([x * 1e3 for x in t])
            r = {"model": name, "units": units, "ms": s, **rate(name, s["median"] / 1e3)}
            rows.write(measure="alone", **r)
            out[f"{name}:{units}"] = r
            print(f"{name:14s} {units:10s} median {s['median']:.3f} ms ({s['p25']:.3f}-{s['p75']:.3f}) "
                  + " ".join(f"{k}={v:.2f}" for k, v in r.items() if isinstance(v, float)), flush=True)
    return out


def handoff(rows: Rows, n: int = 2000) -> dict:
    import mlx.core as mx
    import numpy as np

    res = {}
    for shape in ((1, 5120), (512, 5120)):
        a = mx.random.normal(shape).astype(mx.float16)
        mx.eval(a)
        to_np, back = [], []
        for _ in range(n):
            t = time.perf_counter()
            h = np.array(a)
            t1 = time.perf_counter()
            b = mx.array(h)
            mx.eval(b)
            back.append(time.perf_counter() - t1)
            to_np.append(t1 - t)
        res[f"mlx_to_numpy_{shape[0]}x{shape[1]}_ms"] = spread([x * 1e3 for x in to_np])
        res[f"numpy_to_mlx_eval_{shape[0]}x{shape[1]}_ms"] = spread([x * 1e3 for x in back])
    t = time_model("tiny", 10.0)
    res["coreml_tiny_cpu_round_trip_ms"] = spread([x * 1e3 for x in t])
    t = time_model("tiny_ane", 10.0)
    res["coreml_tiny_ane_round_trip_ms"] = spread([x * 1e3 for x in t])
    rows.write(measure="handoff", **res)
    for k, v in res.items():
        print(f"{k:34s} median {v['median']:.4f} ms (p75 {v['p75']:.4f})", flush=True)
    return res


# ------------------------------------------------------------------------------------------ contention
GPU_LOOP = r"""
import json, sys, time
import mlx.core as mx
start, dur = float(sys.argv[1]), float(sys.argv[2])
mx.set_cache_limit(0)
A = mx.random.normal((4096, 4096)).astype(mx.float16); B = mx.random.normal((4096, 4096)).astype(mx.float16)
work = 8 * 2 * 4096 ** 3
def step():
    return [A @ B for _ in range(8)]
mx.eval(step())
while time.time() < start:
    mx.eval(step())
vals = []
while time.time() < start + dur:
    t = time.perf_counter(); mx.eval(step()); vals.append(work / (time.perf_counter() - t))
print(json.dumps({"who": "gpu", "rates": vals}))
"""


def contention(rows: Rows, seconds: float, gpu_python: str, warmup: float = 10.0) -> dict:
    order = ["gpu", "ane", "both", "both", "ane", "gpu"]
    cells = []
    for rep, cell in enumerate(order):
        start = time.time() + warmup
        procs = {}
        if cell in ("gpu", "both"):
            procs["gpu"] = subprocess.Popen([gpu_python, "-c", GPU_LOOP, str(start), str(seconds)],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if cell in ("ane", "both"):
            procs["ane"] = subprocess.Popen([sys.executable, __file__, "ane-loop", "--start", str(start),
                                             "--seconds", str(seconds)],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        got = {}
        for who, p in procs.items():
            out, err = p.communicate(timeout=seconds + warmup + 180)
            if p.returncode != 0:
                sys.exit(f"{who} failed: {err[-500:]}")
            d = json.loads(out.strip().splitlines()[-1])
            got[who] = statistics.median(d["rates"])
        r = {"cell": cell, "rep": rep, "gpu_tflops": got.get("gpu", 0) / 1e12 if "gpu" in got else None,
             "ane_tflops": got.get("ane") / 1e12 if "ane" in got else None}
        rows.write(measure="contention", **r)
        cells.append(r)
        print(f"{rep} {cell:5s} gpu={r['gpu_tflops']} ane={r['ane_tflops']}", flush=True)
        time.sleep(3)

    def mean(cell, key):
        v = [c[key] for c in cells if c["cell"] == cell and c[key] is not None]
        return statistics.mean(v) if v else None

    s = {"gpu_alone": mean("gpu", "gpu_tflops"), "gpu_beside_ane": mean("both", "gpu_tflops"),
         "ane_alone": mean("ane", "ane_tflops"), "ane_beside_gpu": mean("both", "ane_tflops")}
    s["gpu_slowdown"] = 1 - s["gpu_beside_ane"] / s["gpu_alone"]
    s["ane_slowdown"] = 1 - s["ane_beside_gpu"] / s["ane_alone"]
    s["combined_tflops"] = s["gpu_beside_ane"] + s["ane_beside_gpu"]
    rows.write(measure="contention_summary", **s)
    print(json.dumps(s, indent=1), flush=True)
    return s


def cmd_ane_loop(args) -> None:
    t = time_model("gemm4096", args.seconds, start_at=args.start)
    m, k, n, _ = SHAPES["gemm4096"]
    print(json.dumps({"who": "ane", "rates": [2 * m * k * n / x for x in t]}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "plan", "alone", "handoff", "contention", "all"):
        s = sub.add_parser(c)
        s.add_argument("--seconds", type=float, default=15.0)
        s.add_argument("--gpu-python", default=str(Path.home() / "venvs/deepseek-v41/bin/python"))
    s = sub.add_parser("ane-loop")
    s.add_argument("--start", type=float, required=True)
    s.add_argument("--seconds", type=float, required=True)
    args = ap.parse_args()
    if args.cmd == "ane-loop":
        return cmd_ane_loop(args)
    rows = Rows(OUT / f"{utc()}.jsonl")
    if args.cmd in ("build", "all"):
        build()
    if args.cmd in ("plan", "all"):
        plan(rows)
    if args.cmd in ("alone", "all"):
        alone(rows, args.seconds)
    if args.cmd in ("handoff", "all"):
        handoff(rows)
    if args.cmd in ("contention", "all"):
        contention(rows, 20.0, args.gpu_python)
    print(f"rows: {rows.path}")


if __name__ == "__main__":
    main()
