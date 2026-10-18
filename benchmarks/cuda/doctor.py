"""
CUDA environment probe (`cuda_bench.py doctor`), plan section 6 N0b and section 7.2 (machine manifest).

Collects non-secret machine metadata and nothing else: no full environment dump, no hostname, no network
identifiers, no cloud metadata endpoint, no GPU UUID or serial. Every external command is optional; a missing tool
yields an `unavailable` entry with a reason, never an exception and never a guess.

The parsers (`parse_*`) are pure functions over text so that the Mac test-suite can feed recorded fixtures.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys

from manifest import PLATFORM, SCHEMA_NAME, SCHEMA_VERSION, filter_env, seal, utc_now

# Order matters: parse_nvidia_smi_csv zips this with the CSV columns. No uuid, serial or pci.bus_id on purpose.
SMI_FIELDS = (
    "name", "driver_version", "compute_cap", "memory.total", "memory.free", "pcie.link.gen.current",
    "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max", "persistence_mode",
    "ecc.mode.current", "power.limit", "clocks.sm", "clocks.max.sm", "clocks.mem", "clocks.max.mem",
    "temperature.gpu",
)


def unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


def run_cmd(*cmd: str, timeout: float = 15.0) -> tuple[str | None, str | None]:
    """(stdout, None) or (None, reason). Never raises."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None, f"{cmd[0]} not found"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{cmd[0]} failed: {type(exc).__name__}"
    if proc.returncode != 0:
        return None, f"{cmd[0]} exited {proc.returncode}"
    return proc.stdout, None


def _num(text: str):
    text = text.strip()
    m = re.fullmatch(r"([\d.]+)\s*(MiB|W|MHz|C)?", text)
    if not m:
        return None
    val = float(m.group(1))
    return int(val) if val.is_integer() else val


def parse_nvidia_smi_csv(text: str) -> list[dict]:
    """`nvidia-smi --query-gpu=<SMI_FIELDS> --format=csv,noheader` -> one dict per GPU.

    Numbers lose their unit suffix and the unit is carried in the key (`memory_total_mib`, `power_limit_w`);
    a field reported as `[N/A]` or `[Not Supported]` becomes None (absent, not zero).
    """
    units = {"memory.total": "mib", "memory.free": "mib", "power.limit": "w", "clocks.sm": "mhz",
             "clocks.max.sm": "mhz", "clocks.mem": "mhz", "clocks.max.mem": "mhz", "temperature.gpu": "c"}
    gpus = []
    for line in text.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(SMI_FIELDS):
            continue
        gpu: dict = {}
        for field, cell in zip(SMI_FIELDS, cells, strict=True):
            key = field.replace(".", "_") + (f"_{units[field]}" if field in units else "")
            if cell.startswith("[") or cell == "":
                gpu[key] = None
            elif field in ("name", "driver_version", "persistence_mode", "ecc_mode_current", "compute_cap",
                           "ecc.mode.current"):
                gpu[key] = cell
            else:
                gpu[key] = _num(cell)
        gpus.append(gpu)
    return gpus


def parse_nvidia_smi_header(text: str) -> dict:
    """Driver and the CUDA version the driver supports, from the banner of plain `nvidia-smi`."""
    out = {}
    m = re.search(r"Driver Version:\s*([\d.]+)", text)
    if m:
        out["driver_version"] = m.group(1)
    m = re.search(r"CUDA Version:\s*([\d.]+)", text)
    if m:
        out["driver_cuda_version"] = m.group(1)
    return out


def parse_nvcc(text: str) -> dict:
    m = re.search(r"release\s+([\d.]+),\s*V([\d.]+)", text)
    return {"release": m.group(1), "build": m.group(2)} if m else {}


def parse_lscpu(text: str) -> dict:
    fields = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    out = {"model": fields.get("Model name")}
    for key, src in (("sockets", "Socket(s)"), ("cores_per_socket", "Core(s) per socket"),
                     ("threads_per_core", "Thread(s) per core"), ("logical_cpus", "CPU(s)"),
                     ("numa_nodes", "NUMA node(s)")):
        try:
            out[key] = int(fields[src])
        except (KeyError, ValueError):
            pass
    return {k: v for k, v in out.items() if v is not None}


def parse_meminfo(text: str) -> dict:
    m = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, re.M)
    return {"ram_total_gib": round(int(m.group(1)) / 1024 / 1024, 2)} if m else {}


def _git(*args: str) -> str | None:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out, _ = run_cmd("git", "-C", repo, *args)
    return out.strip() if out is not None else None


def git_state() -> dict:
    sha = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    if sha is None:
        return unavailable("not a git checkout")
    return {"sha": sha, "dirty": bool(status), "dirty_file_count": len(status.splitlines()) if status else 0}


def probe_torch() -> dict:
    """Framework view of the device. Importing torch is attempted lazily and may legitimately fail."""
    try:
        import torch
    except Exception as exc:  # ImportError, or a broken wheel raising anything at import
        return {"installed": False, "reason": f"import failed: {type(exc).__name__}"}
    info = {"installed": True, "version": torch.__version__, "cuda_build": getattr(torch.version, "cuda", None)}
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        info["cuda_available"] = False
        info["cuda_error"] = type(exc).__name__
    if info["cuda_available"]:
        devices = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            devices.append({
                "index": i, "name": p.name, "compute_capability": f"{p.major}.{p.minor}",
                "sm_count": p.multi_processor_count, "vram_total_mib": int(total / 2**20),
                "vram_free_mib": int(free / 2**20),
            })
        info["devices"] = devices
        info["arch_list"] = list(torch.cuda.get_arch_list())
    return info


def probe_host() -> dict:
    host = {"os": platform.system(), "machine": platform.machine(), "kernel": platform.release()}
    if sys.platform.startswith("linux"):
        text, why = run_cmd("lscpu")
        host["cpu"] = parse_lscpu(text) if text else unavailable(why or "lscpu")
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                host.update(parse_meminfo(fh.read()))
        except OSError:
            host["ram"] = unavailable("/proc/meminfo unreadable")
        virt, why = run_cmd("systemd-detect-virt")
        host["virtualization"] = virt.strip() if virt else (why or "unknown")
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)

            def fmt(v):
                return "unlimited" if v == resource.RLIM_INFINITY else int(v)

            host["memlock_limit_bytes"] = {"soft": fmt(soft), "hard": fmt(hard)}
        except (ImportError, OSError, ValueError):
            host["memlock_limit_bytes"] = unavailable("getrlimit failed")
    else:
        host["cpu"] = unavailable(f"not probed on {sys.platform}")
        host["note"] = "development host without an NVIDIA stack"
    host["python"] = platform.python_version()
    return host


def probe_nvidia() -> dict:
    """Driver-level view via nvidia-smi: authoritative for PCIe link, VRAM and clocks."""
    smi, why = run_cmd("nvidia-smi", "--query-gpu=" + ",".join(SMI_FIELDS), "--format=csv,noheader,nounits")
    if smi is None:
        return unavailable(why or "nvidia-smi unavailable")
    gpus = parse_nvidia_smi_csv(smi)
    banner, _ = run_cmd("nvidia-smi")
    out = {"available": True, "gpus": gpus, **parse_nvidia_smi_header(banner or "")}
    nvcc, why = run_cmd("nvcc", "--version")
    out["nvcc"] = parse_nvcc(nvcc) if nvcc else unavailable(why or "nvcc")
    return out


def cuda_ready(report: dict) -> tuple[bool, str]:
    """Is there a usable CUDA device for the framework? (ok, reason)."""
    torch_info = report.get("software", {}).get("torch", {})
    if not torch_info.get("installed"):
        return False, f"torch not usable ({torch_info.get('reason', 'not installed')})"
    if not torch_info.get("cuda_available"):
        return False, "torch reports no CUDA device (no NVIDIA GPU, no driver, or a CPU-only torch build)"
    if not torch_info.get("devices"):
        return False, "torch reports CUDA available but lists no device"
    return True, "ok"


def collect(note: str | None = None, region: str | None = None, session_cap_s: int | None = None) -> dict:
    """Machine manifest (unsealed fields are filled, then sealed with its content id)."""
    nvidia = probe_nvidia()
    torch_info = probe_torch()
    gpu0 = (nvidia.get("gpus") or [{}])[0] if nvidia.get("available") else {}
    devs = torch_info.get("devices") or []
    gpu = {
        "available": bool(devs or gpu0),
        "driver_view": gpu0 or unavailable(nvidia.get("reason", "no GPU reported by the driver")),
        "framework_view": devs[0] if devs else unavailable("no device visible to the framework"),
        "count": len(nvidia.get("gpus", [])) if nvidia.get("available") else 0,
        "gpu_identity": "not collected (no UUID, serial or bus id by design)",
    }
    host = probe_host()
    doc = {
        "schema": SCHEMA_NAME, "schema_version": SCHEMA_VERSION, "platform": PLATFORM, "kind": "machine",
        "created_utc": utc_now(),
        "instance_class": os.environ.get("CACHALOT_CUDA_INSTANCE_CLASS", "unspecified"),
        "region": region or "unspecified",
        "code": {"git": git_state()},
        "gpu": gpu,
        "software": {
            "driver": {k: nvidia[k] for k in ("driver_version", "driver_cuda_version") if k in nvidia}
            or unavailable("no driver"),
            "nvcc": nvidia.get("nvcc", unavailable("nvidia-smi unavailable")),
            "torch": torch_info,
        },
        "host": host,
        "interconnect": ({k: gpu0.get(k) for k in ("pcie_link_gen_current", "pcie_link_gen_max",
                                                    "pcie_link_width_current", "pcie_link_width_max")}
                         if gpu0 else unavailable("no GPU reported by the driver")),
        "environment": filter_env(),
        "profiler": {
            "cuda_events": "available" if torch_info.get("cuda_available") else "unavailable",
            "nsight_systems": "not probed", "nsight_compute": "not probed",
        },
        "power": ({"nvml_board_power": "not probed in N1; nvidia-smi power.limit recorded only"}
                  if gpu0 else unavailable("no GPU")),
        "provenance": {"operator_note": note or "", "session_cap_s": session_cap_s,
                       "tool": "benchmarks/cuda/doctor.py"},
    }
    return seal(doc)
