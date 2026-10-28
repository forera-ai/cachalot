"""
Follow-up O (docs/CUDA-PIN-FLUSH-RECORD.md, cell o3): build and call `cache_touch.c`, the CPU-side cache operations on a pinned buffer.

`CacheTouch.load()` compiles the C file with the system compiler into a private directory (inside `work_dir` when given, so a `/tmp`
mounted noexec does not stop it) and opens it with ctypes. Every way it can fail (no compiler, a compile error, a library that does not
load) is returned as a reason string, never raised: the cell turns it into `unavailable` rows. `available(mode)` says whether a mode can run on this CPU (CPUID bits from the helper itself).
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "cache_touch.c"
F_CLFLUSH, F_CLFLUSHOPT, F_CLWB, F_SSE2 = 1, 2, 4, 8
FLUSH_KIND = {"clflush": 0, "clflushopt": 1, "clwb": 2}
NEEDS = {"memcpy": 0, "nt": F_SSE2, "clflush": F_CLFLUSH, "clflushopt": F_CLFLUSHOPT, "clwb": F_CLWB}


def compiler() -> str | None:
    for name in ("cc", "gcc", "clang"):
        path = shutil.which(name)
        if path:
            return path
    return None


class CacheTouch:
    def __init__(self, lib: ctypes.CDLL, features: int, cc: str, so: Path):
        self.lib, self.features, self.cc, self.so = lib, features, cc, so

    @classmethod
    def load(cls, source: Path = SOURCE, work_dir: str | None = None) -> tuple[CacheTouch | None, str | None]:
        cc = compiler()
        if cc is None:
            return None, "no C compiler (cc, gcc or clang) on PATH"
        out = Path(tempfile.mkdtemp(prefix="cache_touch_", dir=work_dir)) / "cache_touch.so"
        try:
            p = subprocess.run([cc, "-O2", "-shared", "-fPIC", "-o", str(out), str(source)], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"compile failed: {type(exc).__name__}: {exc}"[:200]
        if p.returncode != 0:
            return None, f"compile failed ({p.returncode}): {(p.stderr or p.stdout).strip()[:180]}"
        try:
            lib = ctypes.CDLL(str(out))
        except OSError as exc:
            return None, f"load failed: {exc}"[:200]
        lib.ct_features.restype = ctypes.c_int
        lib.ct_flush.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.ct_flush.restype = ctypes.c_int
        lib.ct_stream_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.ct_stream_copy.restype = ctypes.c_int
        lib.ct_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.ct_copy.restype = None
        return cls(lib, int(lib.ct_features()), cc, out), None

    def available(self, mode: str) -> str | None:
        """None if `mode` can run here, otherwise the reason."""
        need = NEEDS[mode]
        if need and not self.features & need:
            return f"CPUID does not report {mode} (features {self.features:#x})"
        return None

    def run(self, mode: str, dst: int, src: int, n: int) -> None:
        """Apply `mode` to the n bytes at address `dst` (`src` is the new bytes for memcpy and nt). Raises on a helper error code."""
        if mode == "memcpy":
            self.lib.ct_copy(dst, src, n)
            return
        r = self.lib.ct_stream_copy(dst, src, n) if mode == "nt" else self.lib.ct_flush(dst, n, FLUSH_KIND[mode])
        if r != 0:
            raise RuntimeError(f"cache_touch {mode} returned {r}")
