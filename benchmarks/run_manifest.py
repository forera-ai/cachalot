"""
Run manifest and efficiency scorecard (track L0 of docs/RESEARCH-DIRECTION.md).

A *manifest* records what a result was measured on, so that a later session can re-run it from the file and the
repository alone (charter section 5, L0): commit and dirty flag, runtime version, model and bank paths, hardware and
OS, the expert budget, the GPU working-set sysctl, the workload's identifier and content hash, every `CACHALOT_*` and
`MLX_*` variable, and the machine's state at the start (memory pressure, swap, system wired memory, every process's
GPU allocation, the screensaver, other runtime processes). Fields that cannot be read are recorded as None, never
guessed.

A *scorecard row* is one measured result tied to a manifest id. Its fields come from a fixed schema (charter section
7); each value carries a tag, `measured`, `derived` or `estimated`; a field that a run did not measure is absent,
never zero. Rows are JSON lines, appended to a file under `benchmarks/results/` (Hamed's choice, 2026-10-06).

Usage from an instrument:

    import run_manifest as rm
    man = rm.collect(workload="W4:csharp-body", workload_text=body_json, budget_gib=48, instrument="quality_blind_ab")
    mid = rm.write_manifest(out_dir, man)
    rm.append_rows(out_dir / "scorecard.jsonl", [rm.row(mid, "exact", {"ms_token": (138.3, "measured")})])

Command line (prints a manifest of the current machine, no workload):

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/run_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GiB = 1024 ** 3
SCHEMA_VERSION = 1
TAGS = ("measured", "derived", "estimated")
WORKLOAD_CLASSES = ("W1", "W2", "W3", "W4", "W5", "W6", "W7")

# field -> unit. Grouped as in the charter's section 7. Add a field here before a row may carry it.
SCORECARD_FIELDS: dict[str, str] = {
    # performance
    "ttft_s": "s", "prefill_tok_s": "tok/s", "decode_tok_s": "tok/s", "ms_token": "ms",
    "ms_token_p50": "ms", "ms_token_p95": "ms", "ms_token_p99": "ms", "tokens": "count",
    # memory
    "resident_model_gib": "GiB", "expert_cache_gib": "GiB", "kv_gib": "GiB", "mlx_active_gib": "GiB",
    "mlx_peak_gib": "GiB", "mlx_cache_gib": "GiB", "gpu_alloc_gib": "GiB", "system_wired_gib": "GiB",
    "swap_used_gib": "GiB", "pressure_level": "level",
    # storage
    "bytes_per_token": "B", "reads_per_token": "count", "mean_read_bytes": "B", "read_gb_s": "GB/s",
    "hit_rate": "fraction", "misses_per_token": "count", "dropped_per_token": "count",
    "exposed_stall_ms": "ms", "hidden_io_ms": "ms", "wasted_spec_bytes_per_token": "B", "prefetch_precision": "fraction",
    # compute
    "cpu_ms_token": "ms", "gpu_busy_ms_token": "ms", "sync_wait_ms_token": "ms",
    # energy (measured only with Hamed's powermetrics file; memory and drive power are estimates on this platform)
    "watts_cpu": "W", "watts_gpu": "W", "joules_per_token": "J", "tokens_per_joule": "tok/J",
    # quality
    "flawed": "count", "graded": "count", "dnll_nats": "nats", "kl_mean": "nats", "kl_max": "nats", "top1": "fraction",
}


def _run(*cmd: str) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def _sysctl(name: str) -> str | None:
    out = _run("sysctl", "-n", name)
    return out if out else None


def _int(text: str | None) -> int | None:
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def parse_swap_used_gib(text: str | None) -> float | None:
    """`vm.swapusage` -> GiB used ("total = 2048.00M  used = 1536.25M ..." -> 1.5)."""
    m = re.search(r"used = ([\d.]+)([MG])", text or "")
    if not m:
        return None
    return float(m.group(1)) / (1 if m.group(2) == "G" else 1024)


def runtime_processes(ps_text: str, own_pid: int) -> list[str]:
    """Command lines of other Cachalot runtimes (the one-runtime rule) from `ps -Ao pid=,command=` output."""
    found = []
    for line in ps_text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) == own_pid:
            continue
        cmd = parts[1]
        if os.path.basename(cmd.split(None, 1)[0]) in ("zsh", "bash", "sh", "-zsh", "pgrep", "grep"):
            continue  # a shell whose command text names the runtime is not a runtime (serve.sh's pgrep pitfall)
        if "cachalot.cli" in cmd or ("deepseek-v41/bin/python" in cmd and "run_manifest" not in cmd):
            found.append(cmd[:160])
    return found


def git_state() -> dict:
    sha = _run("git", "-C", str(REPO), "rev-parse", "--short", "HEAD")
    status = _run("git", "-C", str(REPO), "status", "--porcelain")
    return {"sha": sha, "dirty": bool(status), "dirty_files": status.splitlines()[:20] if status else []}


def runtime_version() -> str | None:
    m = re.search(r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(), re.M)
    return m.group(1) if m else None


def gpu_alloc_gib() -> float | None:
    """Every process's GPU allocation (the driver's "Alloc system memory"), via the GLM governor's reader."""
    try:
        sys.path.insert(0, str(REPO / "src"))
        from cachalot.glm.model import gpu_allocated
    except Exception:  # noqa: BLE001 - a manifest never fails on an optional field
        return None
    value = gpu_allocated()
    return round(value / GiB, 2) if value >= 0 else None


def content_hash(text: str | bytes | None) -> str | None:
    if text is None:
        return None
    data = text.encode() if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()[:16]


SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def environment(env: dict[str, str]) -> dict[str, str]:
    """The run's knobs. A credential (`CACHALOT_API_KEY` and any name ending in a secret suffix) is recorded as set,
    never by value: manifests are meant to be shared."""
    return {k: ("<redacted>" if k.endswith(SECRET_SUFFIXES) else v) for k, v in sorted(env.items())
            if k.startswith(("CACHALOT_", "MLX_", "TF_", "PYTHONPATH"))}


def hermes_desktop_running(ps_text: str) -> bool:
    """A Hermes app bundle in one process's command line (not "Hermes" in one line and ".app/" in another)."""
    return any("Hermes" in line and ".app/" in line for line in ps_text.splitlines())


def machine_state() -> dict:
    wired_pages, page = _int(_sysctl("vm.page_wired_count")), _int(_sysctl("hw.pagesize"))
    ps = _run("ps", "-Ao", "pid=,command=") or ""
    return {
        "iogpu_wired_limit_mb": _int(_sysctl("iogpu.wired_limit_mb")),
        "memorystatus_level": _int(_sysctl("kern.memorystatus_level")),
        "swap_used_gib": parse_swap_used_gib(_sysctl("vm.swapusage")),
        "system_wired_gib": round(wired_pages * page / GiB, 2) if wired_pages and page else None,
        "gpu_alloc_gib": gpu_alloc_gib(),
        "screensaver_running": "Flurry.appex" in ps or "ScreenSaverEngine" in ps,
        "hermes_desktop_running": hermes_desktop_running(ps),
        "other_runtimes": runtime_processes(ps, os.getpid()),
        "x10pro_mounted": Path("/Volumes/X10Pro").is_dir(),
        "uptime": _run("uptime"),
    }


def hardware() -> dict:
    mem = _int(_sysctl("hw.memsize"))
    return {
        "model": _sysctl("hw.model"),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "memory_gib": round(mem / GiB) if mem else None,
        "os": platform.mac_ver()[0] or platform.platform(),
        "kernel": platform.release(),
    }


def collect(*, workload: str, workload_text: str | bytes | None = None, instrument: str = "",
            budget_gib: float | None = None, context_tokens: int | None = None, cache_state: str | None = None,
            params: dict | None = None, extra: dict | None = None, env: dict[str, str] | None = None) -> dict:
    """The manifest of a run that is about to start. `workload` is "W<n>:<name>" (charter section 7.1)."""
    cls = workload.split(":", 1)[0]
    if cls not in WORKLOAD_CLASSES:
        raise ValueError(f"workload {workload!r} must start with a class {WORKLOAD_CLASSES}")
    env = dict(os.environ if env is None else env)
    return {
        "schema": SCHEMA_VERSION,
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "instrument": instrument,
        "git": git_state(),
        "version": runtime_version(),
        "model_path": env.get("CACHALOT_MODEL_PATH"),
        "expert_bank": env.get("CACHALOT_EXPERT_BANK"),
        "budget_gib": budget_gib,
        "context_tokens": context_tokens,
        "cache_state": cache_state,
        "workload": {"id": workload, "class": cls, "hash": content_hash(workload_text)},
        "params": params or {},
        "env": environment(env),
        "hardware": hardware(),
        "machine": machine_state(),
        "extra": extra or {},
    }


def manifest_id(manifest: dict) -> str:
    """Stable id: the hash of the manifest's content (sorted keys)."""
    return content_hash(json.dumps(manifest, sort_keys=True, default=str))


def write_manifest(out_dir: Path, manifest: dict) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    mid = manifest_id(manifest)
    (out_dir / f"manifest-{mid}.json").write_text(json.dumps(dict(manifest, id=mid), indent=1, default=str))
    return mid


def row(manifest_id_: str, arm: str, values: dict[str, tuple[float, str]], *, note: str = "") -> dict:
    """One scorecard row. `values` maps a schema field to (value, tag); unknown fields and tags are refused,
    and None values are dropped (absent, never zero)."""
    fields = {}
    for name, (value, tag) in values.items():
        if name not in SCORECARD_FIELDS:
            raise KeyError(f"{name!r} is not a scorecard field; add it to SCORECARD_FIELDS first")
        if tag not in TAGS:
            raise ValueError(f"{name}: tag {tag!r} is not one of {TAGS}")
        if value is None:
            continue
        fields[name] = {"value": value, "unit": SCORECARD_FIELDS[name], "tag": tag}
    out = {"schema": SCHEMA_VERSION, "manifest": manifest_id_, "arm": arm, "fields": fields}
    if note:
        out["note"] = note
    return out


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def percentiles(values: list[float], min_n: int = 200) -> dict[str, float]:
    """p50 / p95 / p99 of per-token latencies; empty below `min_n` values (charter section 7: no percentile from a
    short run)."""
    if len(values) < min_n:
        return {}
    s = sorted(values)

    def q(p: float) -> float:
        return s[min(len(s) - 1, round(p * (len(s) - 1)))]

    return {"ms_token_p50": q(0.50), "ms_token_p95": q(0.95), "ms_token_p99": q(0.99)}


if __name__ == "__main__":
    print(json.dumps(collect(workload="W1:none", instrument="run_manifest cli"), indent=1, default=str))
