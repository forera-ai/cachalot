"""
Bounded CUDA microbenchmark suite (plan section 6, phase N1): planning, statistics, the runner and the torch cells.

Layering, so the Mac can test everything except the GPU calls:

* `plan()` turns a profile and `Limits` into a list of `Cell`s. Pure, no torch. This is what `--dry-run` prints.
* `run()` executes the cells under a hard wall-clock `Deadline` through a `bench` object that has one method,
  `execute(cell, deadline, limits) -> CellResult`. Tests pass a scripted bench; the GPU run passes `TorchBench`.
* `TorchBench` imports torch lazily and implements the cells with CUDA events (device time) and a host monotonic clock
  (end-to-end time). Nothing in this module imports torch at import time.

Hard ceilings (`HARD_*`) cannot be lifted from the command line: an unbounded benchmark is worse than none.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import manifest as mf

HARD_MAX_SECONDS = 900
HARD_MAX_ITERS = 1000
HARD_MAX_ALLOC_MIB = 4096
MIN_TIMED_ITERS = 30          # plan N1.7: at least 30 timed iterations for a baseline cell
P99_MIN_N = 200               # p99 only when n >= 200 (benchmarks/run_manifest.percentiles rule)
RESOLUTION_FRACTION = 0.10    # timer overhead above 10 % of a cell's median: below the instrument's resolution
PCIE_SANITY_GB_S = 70.0       # above PCIe 5.0 x16 theoretical (~63 GB/s) a host<->device rate is a bug or a cache hit
HBM_SANITY_GB_S = 4500.0      # above any shipping device-memory rate
MIB = 1024 * 1024

# Record sizes from docs/CUDA-RESEARCH-PLAN.md section 3 (bytes, exactly): DeepSeek 9.95 MB, GLM 14.16 MB, MiniMax 21.1 MiB.
RECORD_SIZES = {"deepseek_9.95MB": 9_950_000, "glm_14.16MB": 14_160_000, "minimax_21.1MiB": int(21.1 * MIB)}
READ_3MIB = 3 * MIB

# Expert-like projection shape, DERIVED (not read from the model config): 9.95 MB at 2.25 bit/weight is 35.4 M weights,
# which is 3 projections of 4096 x 2880. Verify against the config before any cross-claim; fp16 weights here, so the bytes
# read per launch are 7x the 2-bit record's.
EXPERT_D, EXPERT_F = 4096, 2880


@dataclass(frozen=True)
class Limits:
    max_seconds: float = 60.0
    max_iters: int = 30
    max_alloc_mib: int = 1024
    warmup: int = 3
    seed: int = 20261018

    def validated(self) -> Limits:
        if not (0 < self.max_seconds <= HARD_MAX_SECONDS):
            raise ValueError(f"max_seconds must be in (0, {HARD_MAX_SECONDS}]")
        if not (1 <= self.max_iters <= HARD_MAX_ITERS):
            raise ValueError(f"max_iters must be in [1, {HARD_MAX_ITERS}]")
        if not (16 <= self.max_alloc_mib <= HARD_MAX_ALLOC_MIB):
            raise ValueError(f"max_alloc_mib must be in [16, {HARD_MAX_ALLOC_MIB}]")
        if not (0 <= self.warmup <= 50):
            raise ValueError("warmup must be in [0, 50]")
        return self


PROFILE_DEFAULTS = {
    "smoke": Limits(max_seconds=45.0, max_iters=12, max_alloc_mib=1024, warmup=2),
    "baseline": Limits(max_seconds=300.0, max_iters=30, max_alloc_mib=2048, warmup=5),
    "s2": Limits(max_seconds=150.0, max_iters=30, max_alloc_mib=2048, warmup=5),
}


@dataclass(frozen=True)
class Cell:
    id: str
    kind: str                       # copy | launch | device_bw | matmul | overlap | batchcopy
    params: dict
    alloc_mib: float
    est_s: float
    control: bool = False


@dataclass
class CellResult:
    arms: dict[str, list[float]]            # arm -> per-iteration ms
    timer: str                               # cuda_events | host_monotonic | both
    work_bytes: int | None = None            # bytes moved per iteration (bandwidth = work_bytes / time)
    validation: dict = field(default_factory=lambda: {"ok": True, "detail": "not checked"})
    derived: dict[str, tuple[float, str, str]] = field(default_factory=dict)   # name -> (value, unit, note)
    truncated: bool = False
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------------------------- statistics

def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear interpolation between closest ranks; q in [0, 100]."""
    if not sorted_vals:
        raise ValueError("no samples")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def summarize(samples: list[float], seed: int = 0, boot: int = 400) -> dict:
    """n, p50, p95, (p99 only if n >= 200), min, max and a seeded bootstrap 95 % interval of the median."""
    vals = sorted(samples)
    out = {"n": len(vals), "p50": percentile(vals, 50), "p95": percentile(vals, 95),
           "min": vals[0], "max": vals[-1]}
    if len(vals) >= P99_MIN_N:
        out["p99"] = percentile(vals, 99)
    if len(vals) >= 5:
        rng = random.Random(seed)
        meds = sorted(percentile(sorted(rng.choices(vals, k=len(vals))), 50) for _ in range(boot))
        out.update(ci_low=percentile(meds, 2.5), ci_high=percentile(meds, 97.5), ci_method="bootstrap-median-400")
    return out


def below_resolution(p50_ms: float, timer_overhead_ms: float) -> bool:
    return p50_ms <= 0 or timer_overhead_ms > RESOLUTION_FRACTION * p50_ms


def dtype_support(compute_capability: tuple[int, int]) -> dict[str, bool]:
    """Features by what the device reports, never by its name (plan section 10). Only fp16 kernels are run."""
    return {"fp16": compute_capability >= (7, 0), "bf16": compute_capability >= (8, 0),
            "fp8": compute_capability >= (8, 9)}


def arch_supported(arch_list: list[str], compute_capability: tuple[int, int]) -> bool:
    """True if the installed torch carries kernels the device can run: an exact `sm_XY` image, or a PTX `compute_XY`
    image for a capability not above the device's (PTX is forward compatible, never backward). A cu128 wheel lists no
    `sm_70`, so it cannot run on a V100; setup stops on this instead of failing in the first cell (plan risk table)."""
    major, minor = compute_capability
    cc = major * 10 + minor
    for entry in arch_list:
        kind, _, num = entry.partition("_")
        if not num.isdigit():
            continue
        if kind == "sm" and int(num) == cc:
            return True
        if kind == "compute" and int(num) <= cc:
            return True
    return False


def gbps(nbytes: float, ms: float) -> float:
    return nbytes / 1e9 / (ms / 1e3)


def sanity_problem(kind: str, arm_name: str, bw_gb_s: float, peak_gb_s: float | None) -> str | None:
    if kind in ("copy", "batchcopy") and bw_gb_s > PCIE_SANITY_GB_S:
        return f"{bw_gb_s:.1f} GB/s exceeds the PCIe sanity bound {PCIE_SANITY_GB_S} (cache hit or timing bug)"
    if kind == "device_bw" and bw_gb_s > HBM_SANITY_GB_S:
        return f"{bw_gb_s:.1f} GB/s exceeds the device-memory sanity bound {HBM_SANITY_GB_S}"
    if kind == "matmul" and peak_gb_s and bw_gb_s > 1.2 * peak_gb_s:
        return f"{bw_gb_s:.1f} GB/s is above 1.2x the measured device peak {peak_gb_s:.1f} (stale L2 or timing bug)"
    return None


# --------------------------------------------------------------------------------------------------- planning

def _iters(limits: Limits) -> int:
    return limits.max_iters


def _copy_cell(direction: str, mode: str, nbytes: int, limits: Limits, control: bool = False) -> Cell:
    per_iter = nbytes / 3e9 + 60e-6        # pessimistic 3 GB/s so the estimate is an upper-ish bound
    est = (limits.warmup + _iters(limits) + 2) * per_iter * 2
    cid = f"copy.{direction}.{mode}.{nbytes}" + (".control" if control else "")
    return Cell(cid, "copy", {"direction": direction, "mode": mode, "nbytes": nbytes}, 2 * nbytes / MIB, est, control)


def plan(profile: str, limits: Limits) -> tuple[list[Cell], list[dict]]:
    """(cells in execution order, skipped with reasons). Cells over the allocation cap are skipped, never shrunk."""
    limits.validated()
    if profile not in PROFILE_DEFAULTS:
        raise ValueError(f"unknown profile {profile!r}")
    if profile == "s2":
        return _filter_alloc(_s2_cells(limits), limits)
    smoke = profile == "smoke"
    sizes = ([MIB, RECORD_SIZES["deepseek_9.95MB"], 64 * MIB] if smoke else
             [4096, 64 * 1024, MIB, READ_3MIB, *RECORD_SIZES.values(), 64 * MIB, 256 * MIB])
    groups = [1, 8] if smoke else [1, 2, 4, 8, 16]
    ms = [1, 64] if smoke else [1, 16, 128]
    bw_bytes = (128 if smoke else 512) * MIB
    cells: list[Cell] = []
    ctrl_size = RECORD_SIZES["deepseek_9.95MB"]
    cells.append(_copy_cell("h2d", "pinned", ctrl_size, limits, control=True))      # control cell at the start
    for n in sizes:
        for mode in ("pageable", "pinned"):
            for direction in ("h2d", "d2h"):
                cells.append(_copy_cell(direction, mode, n, limits))
    cells.append(Cell("device_bw.d2d", "device_bw", {"nbytes": bw_bytes}, 2 * bw_bytes / MIB + 8, 0.5 + 2e-3 * _iters(limits)))
    for name, est in (("tiny_launch", 0.3), ("sync_idle", 0.1), ("event_pair", 0.1), ("router_chain_40", 0.4)):
        cells.append(Cell(f"launch.{name}", "launch", {"name": name}, 1.0, est))
    wbytes = EXPERT_D * EXPERT_F * 2
    for m in ms:
        for g in groups:
            for mode in (("loop", "bmm") if g > 1 else ("loop",)):
                alloc = (g * wbytes + 64 * MIB) / MIB + 4
                cells.append(Cell(f"matmul.m{m}.g{g}.{mode}", "matmul",
                                  {"m": m, "d": EXPERT_D, "f": EXPERT_F, "g": g, "mode": mode}, alloc,
                                  0.4 + (limits.warmup + _iters(limits)) * (g * wbytes / 5e11 + 40e-6 * g) * 3))
    ov = 64 * MIB
    cells.append(Cell("overlap.copy_vs_compute", "overlap", {"nbytes": ov, "m": 128, "d": EXPERT_D, "f": EXPERT_F},
                      (ov * 2 + 2 * EXPERT_D * EXPERT_F * 2) / MIB + 16, 2.0 + 0.05 * _iters(limits)))
    cells.append(_copy_cell("h2d", "pinned", ctrl_size, limits, control=True))      # and at the end: drift bound
    return _filter_alloc(cells, limits)


def _filter_alloc(cells: list[Cell], limits: Limits) -> tuple[list[Cell], list[dict]]:
    kept, skipped = [], []
    for c in cells:
        if c.alloc_mib > limits.max_alloc_mib:
            skipped.append({"cell": c.id, "reason": f"needs {c.alloc_mib:.0f} MiB, cap {limits.max_alloc_mib} MiB"})
        else:
            kept.append(c)
    return kept, skipped


S2_OVERLAP_COPY_BYTES = 64 * MIB          # the S1 overlap stream, so S2-A is comparable
S2_D2D_BYTES = 256 * MIB                  # larger than any L2 cache, so the interferer evicts it
S2_LARGE_WEIGHTS = 8                      # 8 x 23.6 MB = 189 MB, above an L2 of about 96 MB (vendor figure, derived input)
S2_BATCH_K = (1, 2, 4, 8)


def _s2_cells(limits: Limits) -> list[Cell]:
    """Session S2 (docs/CUDA-S2-EXPERIMENT-RECORD.md): overlap variants that separate an L2 effect from copy-engine
    interference, and batched expert copies. Cell ids are the record's S2-A..S2-F."""
    ctrl = RECORD_SIZES["deepseek_9.95MB"]
    wbytes = EXPERT_D * EXPERT_F * 2
    cells = [_copy_cell("h2d", "pinned", ctrl, limits, control=True)]
    for weights, interferer in (("resident", "h2d"), ("large", "h2d"), ("resident", "d2d"), ("large", "d2d")):
        nw = S2_LARGE_WEIGHTS if weights == "large" else 1
        ibytes = S2_OVERLAP_COPY_BYTES if interferer == "h2d" else S2_D2D_BYTES
        alloc = (ibytes * (2 if interferer == "d2d" else 1) + ibytes + (nw + 1) * wbytes) / MIB + 16
        cells.append(Cell(f"overlap.{weights}.{interferer}", "overlap",
                          {"nbytes": ibytes, "m": 128, "d": EXPERT_D, "f": EXPERT_F, "nweights": nw, "interferer": interferer},
                          alloc, 3.0 + 0.08 * _iters(limits)))
    for k in S2_BATCH_K:
        for mode in (("single",) if k == 1 else ("single", "loop")):
            cells.append(Cell(f"batchcopy.k{k}.{mode}", "batchcopy", {"k": k, "mode": mode, "nbytes": ctrl},
                              2 * k * ctrl / MIB + 4,
                              (limits.warmup + _iters(limits) + 2) * (k * ctrl / 3e9 + 60e-6 * k) * 2))
    cells.append(_copy_cell("h2d", "pinned", ctrl, limits, control=True))
    return cells


def describe_plan(profile: str, limits: Limits) -> dict:
    cells, skipped = plan(profile, limits)
    est = sum(c.est_s for c in cells)
    return {
        "profile": profile, "limits": limits.__dict__, "cells": len(cells),
        "estimated_s_upper_bound": round(est, 1), "wall_cap_s": limits.max_seconds,
        "gpu_hours_cap": round(limits.max_seconds / 3600, 4),
        "peak_alloc_mib": round(max((c.alloc_mib for c in cells), default=0), 1),
        "skipped": skipped, "cell_ids": [c.id for c in cells],
        "note": "estimate is a conservative upper bound; the wall-clock cap is what actually bounds the run",
    }


# ---------------------------------------------------------------------------------------------------- runner

class Deadline:
    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic):
        self.clock, self.t0, self.seconds = clock, clock(), seconds

    def elapsed(self) -> float:
        return self.clock() - self.t0

    def remaining(self) -> float:
        return self.seconds - self.elapsed()

    def expired(self) -> bool:
        return self.remaining() <= 0


def _rows_for(run_id: str, cell: Cell, res: CellResult, timer_overhead_ms: float, peak: float | None,
              seed: int, sanity_log: list[str]) -> tuple[list[dict], float | None]:
    rows: list[dict] = []
    best_bw: float | None = None
    src = {"cuda_events": "cuda_events", "host_monotonic": "host_monotonic", "both": "cuda_events+host_monotonic"}[res.timer]
    for arm, samples in res.arms.items():
        name = f"{cell.id}|{arm}"
        if not samples:
            rows.append(mf.row(run_id, name, "time_ms", None, "ms", "unavailable", source=src,
                               note="no samples (cap reached before the cell produced data)"))
            continue
        s = summarize(samples, seed=seed)
        kw = dict(source=src, n=s["n"], p50=s["p50"], p95=s["p95"], ci_low=s.get("ci_low"),
                  ci_high=s.get("ci_high"), ci_method=s.get("ci_method"), overhead_ms=timer_overhead_ms)
        if below_resolution(s["p50"], timer_overhead_ms):
            rows.append(mf.row(run_id, name, "time_ms", None, "ms", "unavailable", source=src,
                               note=f"below instrument resolution: timer overhead {timer_overhead_ms:.4f} ms vs p50 {s['p50']:.4f} ms"))
            continue
        notes = []
        if res.truncated:
            notes.append("truncated by the wall-clock cap")
        if s["n"] < MIN_TIMED_ITERS:
            notes.append(f"n={s['n']} is below the {MIN_TIMED_ITERS}-iteration baseline minimum (smoke-grade)")
        note = "; ".join(notes) or None
        rows.append(mf.row(run_id, name, "time_ms", s["p50"], "ms", "measured", note=note, **kw))
        if res.work_bytes and arm in ("events", "main", "host"):
            bw = gbps(res.work_bytes, s["p50"])
            problem = sanity_problem(cell.kind, arm, bw, peak)
            note = None
            if problem:
                sanity_log.append(f"{cell.id}|{arm}: {problem}")
                note = f"SANITY FAIL: {problem}"
            rows.append(mf.row(run_id, name, "bandwidth_gb_s", bw, "GB/s", "derived", source="work_bytes / p50 time",
                               n=s["n"], note=note))
            if peak:
                rows.append(mf.row(run_id, name, "share_of_measured_peak", bw / peak, "fraction", "derived",
                                   source="bandwidth / measured d2d peak (R4)"))
            if arm in ("events", "main") and (best_bw is None or bw > best_bw):
                best_bw = bw
    for dname, (val, unit, dnote) in res.derived.items():
        rows.append(mf.row(run_id, cell.id, dname, val, unit, "derived", source="cell", note=dnote))
    return rows, best_bw


def run(profile: str, limits: Limits, bench, *, run_id: str, machine_id: str, note: str = "",
        code: dict | None = None, clock: Callable[[], float] = time.monotonic) -> tuple[dict, list[dict], dict]:
    """Execute the plan. Returns (run manifest, result rows, raw samples). Never raises on a failed cell."""
    limits.validated()
    cells, skipped = plan(profile, limits)
    dl = Deadline(limits.max_seconds, clock)
    overhead = bench.timer_overhead_ms()
    rows: list[dict] = []
    raw: dict[str, dict] = {}
    gates = {"copy_checksum": True, "kernel_reference": True, "sanity": True}
    failures: list[dict] = []
    sanity_log: list[str] = []
    peak: float | None = None
    control_p50: list[float] = []
    status = "complete"
    for cell in cells:
        if dl.expired():
            status = "truncated"
            skipped.append({"cell": cell.id, "reason": "wall-clock cap reached"})
            continue
        try:
            res = bench.execute(cell, dl, limits)
        except Exception as exc:                      # a failed cell is recorded, never hidden, never fatal
            failures.append({"cell": cell.id, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            rows.append(mf.row(run_id, cell.id, "time_ms", None, "ms", "unavailable", source="runner",
                               note=f"cell failed: {type(exc).__name__}"))
            continue
        if res.truncated:
            status = "truncated"
        if not res.validation.get("ok", False):
            gates["copy_checksum" if cell.kind in ("copy", "overlap", "device_bw", "batchcopy") else "kernel_reference"] = False
            rows.append(mf.row(run_id, cell.id, "validation_failed", 1, "bool", "measured", source="validation",
                               note=str(res.validation.get("detail", ""))[:200]))
            raw[cell.id] = {"validation": res.validation, "arms": {}}
            continue                                  # a failed gate produces no timing row (plan 7.6)
        cell_rows, best = _rows_for(run_id, cell, res, overhead, peak, limits.seed, sanity_log)
        rows += cell_rows
        if cell.kind == "device_bw" and best:
            peak = best
            rows.append(mf.row(run_id, cell.id, "measured_device_peak_gb_s", best, "GB/s", "derived",
                               source="max d2d copy bandwidth (2 x bytes / p50)"))
        if cell.control:
            s = [x for x in res.arms.get("events", res.arms.get("main", []))]
            if s:
                control_p50.append(summarize(s, seed=limits.seed)["p50"])
        raw[cell.id + ("@end" if cell.control and cell.id in raw else "")] = {
            "arms": res.arms, "timer": res.timer, "work_bytes": res.work_bytes, "validation": res.validation,
            "truncated": res.truncated, "notes": res.notes}
    if len(control_p50) >= 2:
        drift = abs(control_p50[-1] / control_p50[0] - 1.0)
        rows.append(mf.row(run_id, "control.copy.h2d.pinned", "control_drift", drift, "fraction", "derived",
                           source="end p50 / start p50 - 1", note="bounds drift within the session (plan N1.7)"))
    if sanity_log:
        gates["sanity"] = False
    manifest = {
        "schema": mf.SCHEMA_NAME, "schema_version": mf.SCHEMA_VERSION, "platform": mf.PLATFORM, "kind": "run",
        "created_utc": mf.utc_now(), "machine_manifest_id": machine_id, "run_id": run_id,
        "instrument": {"name": "benchmarks/cuda/suite.py", "version": "N1-scaffold-1"},
        "code": code or {}, "mode": "timing",
        "workload": {"id": f"N1-micro-{profile}", "class": "microbenchmark (no W1-W7 mapping)",
                     "seed": limits.seed, "compute_dtype": "float16", "stored_dtype": "float16 / uint8 bytes",
                     "expert_like_shape": {"d": EXPERT_D, "f": EXPERT_F, "derived": True},
                     "claims": "no model, no inference; microbenchmarks only"},
        "protocol": {"warmup": limits.warmup, "timed_iters_cap": limits.max_iters, "ordering": "single pass, fixed",
                     "control_cells": "h2d pinned at the DeepSeek record size, first and last",
                     "timer_source": "cuda events (device) and host monotonic (end-to-end)",
                     "timer_overhead_ms": overhead, "min_iters_for_baseline": MIN_TIMED_ITERS},
        "limits": {**limits.__dict__, "hard_max_seconds": HARD_MAX_SECONDS, "hard_max_alloc_mib": HARD_MAX_ALLOC_MIB,
                   "elapsed_s": round(dl.elapsed(), 3)},
        "gates": {**gates, "sanity_violations": sanity_log, "failed_cells": failures},
        "status": status, "skipped": skipped, "operator_note": note,
    }
    return mf.seal(manifest), rows, raw


# ------------------------------------------------------------------------------------------------ torch cells

class _CpuEvent:
    """Stand-in for torch.cuda.Event in the CPU rehearsal."""

    def record(self, stream=None):
        self.t = time.perf_counter()

    def synchronize(self):
        pass

    def elapsed_time(self, other) -> float:
        return (other.t - self.t) * 1e3


class TorchBench:
    """The GPU cells. Constructed only on a machine where torch sees CUDA; imports torch lazily."""

    def __init__(self, device_index: int = 0, rehearsal_cpu: bool = False):
        """`rehearsal_cpu` runs the same cell code on the CPU with a perf_counter event shim. It exists so that the Mac
        tests exercise every cell before any metered GPU time; its numbers are never a CUDA measurement and the command
        line does not expose it."""
        import torch
        self.torch = torch
        self.rehearsal = rehearsal_cpu
        if rehearsal_cpu:
            self.dev = torch.device("cpu")
            self.cc = (0, 0)
            return
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        self.dev = torch.device("cuda", device_index)
        torch.cuda.set_device(self.dev)
        self.cc = torch.cuda.get_device_capability(self.dev)

    # -- helpers
    def _events(self):
        if self.rehearsal:
            return _CpuEvent(), _CpuEvent()
        t = self.torch
        return t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)

    def _sync(self) -> None:
        if not self.rehearsal:
            self.torch.cuda.synchronize()

    def _pin(self, x):
        return x.pin_memory() if not self.rehearsal else x

    def _stream(self):
        return None if self.rehearsal else self.torch.cuda.Stream(device=self.dev)

    def _on(self, stream):
        import contextlib
        return contextlib.nullcontext() if stream is None else self.torch.cuda.stream(stream)

    def _current_stream(self):
        return None if self.rehearsal else self.torch.cuda.current_stream(self.dev)

    def timer_overhead_ms(self) -> float:
        vals = []
        for _ in range(60):
            a, b = self._events()
            a.record()
            b.record()
            b.synchronize()
            vals.append(a.elapsed_time(b))
        self._sync()
        return percentile(sorted(vals[10:]), 50)

    def _timed(self, fn, warmup: int, iters: int, dl: Deadline, stream=None) -> tuple[list[float], list[float], bool]:
        """(event ms, host ms, truncated). The deadline is checked before every timed iteration."""
        for _ in range(warmup):
            fn()
        self._sync()
        ev_ms, host_ms, trunc = [], [], False
        for _ in range(iters):
            if dl.expired():
                trunc = True
                break
            a, b = self._events()
            h0 = time.perf_counter_ns()
            a.record(stream)
            fn()
            b.record(stream)
            b.synchronize()
            host_ms.append((time.perf_counter_ns() - h0) / 1e6)
            ev_ms.append(a.elapsed_time(b))
        return ev_ms, host_ms, trunc

    def _pattern(self, nbytes: int, seed: int, pinned: bool):
        t = self.torch
        g = t.Generator().manual_seed(seed)
        x = t.randint(0, 256, (nbytes,), dtype=t.uint8, generator=g)
        return self._pin(x) if pinned else x

    # -- dispatch
    def execute(self, cell: Cell, dl: Deadline, limits: Limits) -> CellResult:
        fn = {"copy": self._copy, "launch": self._launch, "device_bw": self._device_bw, "matmul": self._matmul,
              "overlap": self._overlap, "batchcopy": self._batchcopy}[cell.kind]
        try:
            return fn(cell, dl, limits)
        finally:
            self._sync()
            if not self.rehearsal:
                self.torch.cuda.empty_cache()

    def _copy(self, cell: Cell, dl: Deadline, limits: Limits) -> CellResult:
        t, p = self.torch, cell.params
        n, pinned, h2d = p["nbytes"], p["mode"] == "pinned", p["direction"] == "h2d"
        host = self._pattern(n, limits.seed, pinned)
        dev = t.empty(n, dtype=t.uint8, device=self.dev)
        if not h2d:
            dev.copy_(host)
        back = self._pin(t.empty(n, dtype=t.uint8)) if pinned else t.empty(n, dtype=t.uint8)
        src, dst = (host, dev) if h2d else (dev, back)
        # validation: the bytes must survive the trip, checked outside the timed region
        dst.copy_(src)
        self._sync()
        check = dev.cpu() if h2d else back
        ok = bool(t.equal(check, host))
        ev, hm, trunc = self._timed(lambda: dst.copy_(src, non_blocking=pinned), limits.warmup, limits.max_iters, dl)
        return CellResult({"events": ev, "host": hm}, "both", n,
                          {"ok": ok, "detail": "round-trip bytes equal" if ok else "byte mismatch after copy"},
                          truncated=trunc)

    def _launch(self, cell: Cell, dl: Deadline, limits: Limits) -> CellResult:
        t, name = self.torch, cell.params["name"]
        x = t.zeros(1, device=self.dev)
        out_arm: dict[str, list[float]] = {"host": []}
        derived: dict = {}
        trunc = False

        def sample(fn, label):
            nonlocal trunc
            vals = []
            for _ in range(limits.warmup):
                fn()
            self._sync()
            for _ in range(limits.max_iters):
                if dl.expired():
                    trunc = True
                    break
                h0 = time.perf_counter_ns()
                fn()
                vals.append((time.perf_counter_ns() - h0) / 1e6)
            out_arm["host"] = vals
            return vals

        if name == "tiny_launch":
            k = 200

            def batch():
                for _ in range(k):
                    x.add_(1)
                self._sync()
            vals = sample(batch, name)
            if vals:
                derived["per_launch_us"] = (percentile(sorted(vals), 50) / k * 1e3, "us",
                                            "framework-level launch (python + torch dispatch), 200 launches then one sync")
        elif name == "sync_idle":
            sample(self._sync, name)
        elif name == "event_pair":
            def pair():
                a, b = self._events()
                a.record()
                b.record()
                b.synchronize()
            sample(pair, name)
        else:                                    # router_chain_40: 40 x (kernel, tiny D2H of its result, host wait)
            def chain():
                for _ in range(40):
                    x.add_(1)
                    _ = x.item()
            vals = sample(chain, name)
            if vals:
                derived["per_layer_sync_us"] = (percentile(sorted(vals), 50) / 40 * 1e3, "us",
                                                "kernel + 4-byte D2H + host wait, per layer; CUDA-Q3/H4 input")
        return CellResult(out_arm, "host_monotonic", None, {"ok": True, "detail": "no output to check"},
                          derived, trunc)

    def _device_bw(self, cell: Cell, dl: Deadline, limits: Limits) -> CellResult:
        t, n = self.torch, cell.params["nbytes"]
        t.manual_seed(limits.seed)
        a = t.randint(0, 256, (n,), dtype=t.uint8, device=self.dev)
        b = t.empty_like(a)
        ev, hm, trunc = self._timed(lambda: b.copy_(a), limits.warmup, limits.max_iters, dl)
        ok = bool(t.equal(a, b))
        return CellResult({"events": ev, "host": hm}, "both", 2 * n,
                          {"ok": ok, "detail": "d2d copy bytes equal" if ok else "d2d mismatch"}, truncated=trunc,
                          notes=["buffer >> L2, so the rate is not a cache rate; bytes = read + write"])

    def _matmul(self, cell: Cell, dl: Deadline, limits: Limits) -> CellResult:
        t, p = self.torch, cell.params
        m, d, f, g, mode = p["m"], p["d"], p["f"], p["g"], p["mode"]
        t.manual_seed(limits.seed)
        x = (t.randn(m, d, device=self.dev) * 0.1).half()
        ws = [(t.randn(d, f, device=self.dev) * 0.05).half() for _ in range(g)]
        stack = t.stack(ws) if mode == "bmm" else None
        xs = x.unsqueeze(0).expand(g, m, d).contiguous() if mode == "bmm" else None
        flush = t.empty(64 * MIB, dtype=t.uint8, device=self.dev)

        def work():
            flush.zero_()                       # evict L2 so the weights come from memory (plan section 9)
            if mode == "bmm":
                return t.bmm(xs, stack)
            return [x @ w for w in ws]

        def timed():
            if mode == "bmm":
                t.bmm(xs, stack)
            else:
                for w in ws:
                    x @ w

        # reference check on expert 0 (float32 on CPU), tolerance recorded
        y = (t.bmm(xs, stack)[0] if mode == "bmm" else x @ ws[0]).float().cpu()
        ref = x.float().cpu() @ ws[0].float().cpu()
        err = float((y - ref).abs().max())
        tol = 0.02 * max(1.0, float(ref.abs().max()))
        ok = math.isfinite(err) and err <= tol
        ev, hm, trunc = [], [], False
        for _ in range(limits.warmup):
            work()
        self._sync()
        for _ in range(limits.max_iters):
            if dl.expired():
                trunc = True
                break
            flush.zero_()
            self._sync()
            a, b = self._events()
            h0 = time.perf_counter_ns()
            a.record()
            timed()
            b.record()
            b.synchronize()
            hm.append((time.perf_counter_ns() - h0) / 1e6)
            ev.append(a.elapsed_time(b))
        wbytes = g * d * f * 2
        flops = 2 * m * d * f * g
        derived = {"tflops": (flops / 1e12 / (percentile(sorted(ev), 50) / 1e3), "TFLOP/s",
                              "2*m*d*f*g / p50 time; fp16 inputs")} if ev else {}
        return CellResult({"events": ev, "host": hm}, "both", wbytes,
                          {"ok": ok, "detail": f"max abs err {err:.4g} vs float32 CPU reference, tol {tol:.4g}"},
                          derived, trunc, [f"fp16 compute; bf16/fp8 gated off (cc {self.cc[0]}.{self.cc[1]})"])

    def _overlap(self, cell: Cell, dl: Deadline, limits: Limits) -> CellResult:
        """Compute timed alone and while an interferer runs on a second stream. Params `nweights` (1 = the S1 cell: one
        weight matrix reused 8 times; n = n distinct matrices, each used once) and `interferer` ('h2d' pinned stream, S1;
        'd2d' device copy that evicts L2 without PCIe) separate a cache effect from copy-engine interference (S2)."""
        t, p = self.torch, cell.params
        n, m, d, f = p["nbytes"], p["m"], p["d"], p["f"]
        nw, interferer = p.get("nweights", 1), p.get("interferer", "h2d")
        t.manual_seed(limits.seed)
        x = (t.randn(m, d, device=self.dev) * 0.1).half()
        ws = [(t.randn(d, f, device=self.dev) * 0.05).half() for _ in range(nw)]
        copy_stream = self._stream()
        if interferer == "h2d":
            host = self._pattern(n, limits.seed, True)
            src, dev = host, t.empty(n, dtype=t.uint8, device=self.dev)
        else:
            src = t.randint(0, 256, (n,), dtype=t.uint8, device=self.dev)
            dev = t.empty_like(src)
            self._sync()                      # src is built on the default stream; the copy stream must not read it early

        def compute():
            if nw == 1:
                for _ in range(8):
                    x @ ws[0]
            else:
                for w in ws:
                    x @ w

        def copy_once():
            with self._on(copy_stream):
                dev.copy_(src, non_blocking=True)

        # validation of the copied bytes (outside timing)
        copy_once()
        self._sync()
        ok = bool(t.equal(dev.cpu(), src.cpu()))
        c_alone, _, trunc = self._timed(compute, limits.warmup, limits.max_iters, dl)
        cp_alone, _, trunc2 = self._timed(copy_once, limits.warmup, limits.max_iters, dl, stream=copy_stream)
        reps = 1
        if c_alone and cp_alone:
            reps = min(8, max(1, math.ceil(percentile(sorted(c_alone), 50) / max(percentile(sorted(cp_alone), 50), 1e-6)) + 1))
        cur = self._current_stream()
        both: list[float] = []
        trunc3 = False
        for _ in range(limits.max_iters):
            if dl.expired():
                trunc3 = True
                break
            self._sync()
            a, b = self._events()
            for _ in range(reps):
                copy_once()                       # interferer traffic covers the whole compute window (reps sized above)
            a.record(cur)
            compute()
            b.record(cur)
            b.synchronize()
            both.append(a.elapsed_time(b))
        self._sync()
        derived = {}
        if c_alone and both:
            slow = percentile(sorted(both), 50) / percentile(sorted(c_alone), 50) - 1.0
            derived["compute_slowdown_with_copy"] = (slow, "fraction", f"p50 compute with {reps} concurrent {interferer} copies / alone - 1; CUDA-H7 input; {nw} weight matrices")
        derived["copies_per_compute_window"] = (float(reps), "count", "sized so interferer traffic spans the compute window")
        if not self.rehearsal:
            l2 = getattr(self.torch.cuda.get_device_properties(self.dev), "L2_cache_size", None)
            if l2:
                derived["l2_cache_bytes"] = (float(l2), "bytes", "device property reported by the framework")
        return CellResult({"compute_alone": c_alone, "copy_alone": cp_alone, "compute_with_copy": both}, "cuda_events",
                          None, {"ok": ok, "detail": "copied bytes equal" if ok else "byte mismatch"}, derived,
                          trunc or trunc2 or trunc3)

    def _batchcopy(self, cell: Cell, dl: Deadline, limits: Limits) -> CellResult:
        """k expert records as one pinned H2D transfer ('single') or as k queued transfers ('loop'), one event pair around
        all of them. Tests the fixed-cost-per-transfer explanation of S1 finding 1."""
        t, p = self.torch, cell.params
        k, mode, n = p["k"], p["mode"], p["nbytes"]
        host = self._pattern(k * n, limits.seed, True)
        dev = t.empty(k * n, dtype=t.uint8, device=self.dev)

        def go():
            if mode == "single":
                dev.copy_(host, non_blocking=True)
            else:
                for i in range(k):
                    dev[i * n:(i + 1) * n].copy_(host[i * n:(i + 1) * n], non_blocking=True)

        go()
        self._sync()
        ok = bool(t.equal(dev.cpu(), host))
        ev, hm, trunc = self._timed(go, limits.warmup, limits.max_iters, dl)
        derived = {}
        if ev:
            derived["per_expert_ms"] = (percentile(sorted(ev), 50) / k, "ms", f"p50 event time / k={k} experts of {n} bytes ({mode})")
        return CellResult({"events": ev, "host": hm}, "both", k * n,
                          {"ok": ok, "detail": "round-trip bytes equal" if ok else "byte mismatch after batched copy"},
                          derived, trunc)
