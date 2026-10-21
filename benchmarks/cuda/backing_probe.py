"""
Session B1 instrument (docs/CUDA-BACKING-EXPERIMENT-RECORD.md): what sets the host-to-device rate of a pinned 9.95 MB buffer,
and the S1 copy curve redone with qualified buffers.

N2 stage 1 found that the same card copies a 9.95 MB pinned buffer in 0.182 ms for most buffers and 0.31 to 0.36 ms for a
minority, stable per buffer. This probe separates the explanations without privileges: pinned memory is made two ways,
`cudaHostAlloc` (what the earlier sessions used) and `cudaHostRegister` on anonymous memory this process maps itself, so
transparent-huge-page advice can be applied before the pages are touched and the outcome read back from `/proc/self/smaps`
(`AnonHugePages`, an observation, not a physical address).

Session B2 (docs/CUDA-HUGEPAGE-RETEST-RECORD.md) adds cells `h1` to `h7`. B1's self-mapped buffers used Python's default
`mmap.mmap(-1, n)`, which is MAP_SHARED (shared anonymous memory: huge pages follow `shmem_enabled`, smaps shows
`ShmemPmdMapped`); B2 maps MAP_PRIVATE|MAP_ANONYMOUS (huge pages follow `enabled`, smaps shows `AnonHugePages`), reads smaps
after the pages are made, after registration and after the timed copies, and records the host's THP policy.

Layering, so the Mac can test everything but the GPU calls: the pure helpers (`parse_smaps`, `slow_flags`, `lag1_agreement`,
`run_lengths`, ...) need no torch; `BackingBench` imports torch lazily and runs on a GPU, or on the CPU with an event shim
(`rehearsal_cpu=True`, never on the command line: its numbers are never a CUDA measurement). Cell groups are separate
processes (`--cells b1`, `rest`, `curve`) because the caching host allocator keeps freed pinned blocks and rounds sizes up.

Usage:
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/backing_probe.py plan
GPU host: `run --cells b1|rest|curve|h1|h2|h3|h4|h5|h6|h7 [--execute] [--tag NAME]`.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import mmap
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import manifest as mf  # noqa: E402
import suite  # noqa: E402

RECORD_BYTES = 9_950_000
MIB = 1024 * 1024
HUGE = 2 * MIB
SLOW_MS_9_95MB = 0.28                         # record: "slow" at 9.95 MB means p50 at or above this
CURVE_SIZES = (MIB, 3 * MIB, RECORD_BYTES, 14_160_000, int(21.1 * MIB), 64 * MIB, 256 * MIB)
CURVE_K = (16, 16, 16, 12, 12, 8, 3)          # buffers per size; at most 1.2 GiB pinned at once (memlock limit 2.9 GiB)
SEED = 20261018
HARD_SECONDS = 600
MADV_POPULATE_WRITE = 23                      # Linux 5.14 and later; Python's mmap has no constant for it


# ------------------------------------------------------------------------------------------------ pure helpers

_VMA_HEAD = re.compile(r"^([0-9a-f]+)-([0-9a-f]+)\s+[rwxsp-]{4}\s")


def parse_smaps(text: str, address: int) -> dict | None:
    """The /proc/self/smaps entry whose range contains `address`: {start, end, size_kb, rss_kb, anon_huge_kb, thp_eligible}."""
    out: dict | None = None
    inside = False
    for line in text.splitlines():
        m = _VMA_HEAD.match(line)
        if m:
            lo, hi = int(m.group(1), 16), int(m.group(2), 16)
            inside = lo <= address < hi
            if inside:
                out = {"start": lo, "end": hi}
            continue
        if inside and out is not None and ":" in line:
            key, _, val = line.partition(":")
            v = val.split()
            if not v:
                continue
            if key == "Size":
                out["size_kb"] = int(v[0])
            elif key == "Rss":
                out["rss_kb"] = int(v[0])
            elif key == "AnonHugePages":
                out["anon_huge_kb"] = int(v[0])
            elif key == "THPeligible":
                out["thp_eligible"] = int(v[0])
            elif key == "ShmemPmdMapped":
                out["shmem_pmd_kb"] = int(v[0])
            elif key == "FilePmdMapped":
                out["file_pmd_kb"] = int(v[0])
    return out if out is not None and "size_kb" in out else None


def find_cudart_lib(search_roots: list[str]) -> str | None:
    """libcudart under an `nvidia/cuda_runtime/lib` directory of a site-packages root (the pip CUDA runtime wheel), or None."""
    for root in search_roots:
        for pattern in ("nvidia/cuda_runtime/lib/libcudart.so.*", "torch/lib/libcudart*.so*"):
            hits = sorted(Path(root).glob(pattern))
            if hits:
                return str(hits[0])
    return None


def huge_fraction(entry: dict | None, registered_bytes: int) -> float | None:
    """AnonHugePages of the mapping over the registered range, capped at 1; None when smaps could not be read."""
    if not entry or "anon_huge_kb" not in entry:
        return None
    return min(1.0, entry["anon_huge_kb"] * 1024 / registered_bytes)


def shmem_fraction(entry: dict | None, registered_bytes: int) -> float | None:
    """ShmemPmdMapped (huge pages of a shared mapping) over the registered range, capped at 1; None when not reported."""
    if not entry or "shmem_pmd_kb" not in entry:
        return None
    return min(1.0, entry["shmem_pmd_kb"] * 1024 / registered_bytes)


THP_FILES = ("enabled", "defrag", "shmem_enabled", "khugepaged/defrag")
MEMINFO_KEYS = ("AnonHugePages", "ShmemHugePages", "ShmemPmdMapped", "HugePages_Total", "MemTotal", "MemAvailable")


def host_thp() -> dict:
    """The host's transparent-huge-page policy and counters, read, never set; missing files read as None (the Mac)."""
    out: dict = {}
    for name in THP_FILES:
        try:
            out[name] = Path("/sys/kernel/mm/transparent_hugepage", name).read_text().strip()
        except OSError:
            out[name] = None
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        text = ""
    for key in MEMINFO_KEYS:
        m = re.search(rf"^{key}:\s+(\d+)", text, re.MULTILINE)
        out[key] = int(m.group(1)) if m else None
    return out


def slow_flags(ms: list[float], threshold: float | None = None, ratio: float = 0.8) -> list[bool]:
    """True for a slow buffer. With `threshold`: p50 at or above it (the 9.95 MB rule). Otherwise: rate below `ratio` times
    the fastest buffer's, i.e. time above fastest / ratio (the size-independent rule of B-P11)."""
    if not ms:
        return []
    cut = threshold if threshold is not None else min(ms) / ratio
    return [x >= cut for x in ms]


def share(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def third_mode_share(ms: list[float], lo: float = 0.20, hi: float = 0.28) -> float:
    return sum(1 for x in ms if lo <= x < hi) / len(ms) if ms else 0.0


def order_gap(flags: list[bool]) -> float | None:
    """Mean allocation index of slow buffers minus that of fast buffers; None when either class is empty."""
    slow = [i for i, f in enumerate(flags) if f]
    fast = [i for i, f in enumerate(flags) if not f]
    if not slow or not fast:
        return None
    return sum(slow) / len(slow) - sum(fast) / len(fast)


def lag1_agreement(flags: list[bool]) -> float | None:
    if len(flags) < 2:
        return None
    return sum(1 for a, b in zip(flags[:-1], flags[1:], strict=True) if a == b) / (len(flags) - 1)


def expected_agreement(p: float) -> float:
    return p * p + (1 - p) * (1 - p)


def run_lengths(flags: list[bool], value: bool = True) -> list[int]:
    out, n = [], 0
    for f in flags:
        if f == value:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def gbps(nbytes: float, ms: float) -> float:
    return nbytes / 1e9 / (ms / 1e3)


def p50(xs: list[float]) -> float:
    return statistics.median(xs)


# ------------------------------------------------------------------------------------------------ the bench

class _CpuEvent:
    def record(self, stream=None):
        self.t = time.perf_counter()

    def synchronize(self):
        pass

    def elapsed_time(self, other) -> float:
        return (other.t - self.t) * 1e3


class MmapBuffer:
    """Anonymous memory this process maps, 2 MiB aligned and rounded up to whole 2 MiB extents, optionally advised, touched,
    and registered as pinned. `advice` is 'thp' (MADV_HUGEPAGE), 'nothp' (MADV_NOHUGEPAGE) or 'default' (none)."""

    def __init__(self, bench, nbytes: int, advice: str, seed_tensor, mapping: str = "shared", populate: bool = False,
                 reg_flags: int = 1):
        """`mapping` 'shared' is what B1 used (Python's default MAP_SHARED: shared anonymous memory, whose huge pages follow
        `shmem_enabled`); 'private' is MAP_PRIVATE|MAP_ANONYMOUS (anonymous memory, whose huge pages follow `enabled`).
        `populate` makes the pages with madvise(MADV_POPULATE_WRITE) before the content is written. `reg_flags` is the
        `cudaHostRegister` flag word (1 = portable, what B1 and B2 used; 0 = DeepInfer's call, session K)."""
        torch = bench.torch
        self.bench, self.nbytes, self.advice, self.mapping = bench, nbytes, advice, mapping
        self.registered = ((nbytes + HUGE - 1) // HUGE) * HUGE
        if mapping == "private":
            anon = getattr(mmap, "MAP_ANONYMOUS", getattr(mmap, "MAP_ANON", 0))
            self.map = mmap.mmap(-1, self.registered + HUGE, flags=mmap.MAP_PRIVATE | anon)
        else:
            self.map = mmap.mmap(-1, self.registered + HUGE)
        self._cbuf = ctypes.c_char.from_buffer(self.map)
        base = ctypes.addressof(self._cbuf)
        self.offset = (-base) % HUGE
        self.address = base + self.offset
        self.advice_error: str | None = None
        opt = {"thp": getattr(mmap, "MADV_HUGEPAGE", None), "nothp": getattr(mmap, "MADV_NOHUGEPAGE", None)}.get(advice)
        if advice in ("thp", "nothp"):
            if opt is None:
                self.advice_error = "mmap.madvise constants are not available on this platform"
            else:
                try:
                    self.map.madvise(opt, self.offset, self.registered)
                except OSError as exc:
                    self.advice_error = f"madvise failed: {exc.strerror or exc}"
        self.populate_error: str | None = None
        if populate:
            try:
                self.map.madvise(MADV_POPULATE_WRITE, self.offset, self.registered)
            except (OSError, ValueError) as exc:
                self.populate_error = f"populate failed: {getattr(exc, 'strerror', None) or exc}"
        self.tensor = torch.frombuffer(self.map, dtype=torch.uint8, offset=self.offset, count=self.registered)
        self.smaps_make = self.smaps_entry() if populate else None      # read before the content is written
        self.tensor.copy_(seed_tensor[: self.registered])           # touches every page after the advice
        if not populate:
            self.smaps_make = self.smaps_entry()                        # after the touch, before registration
        self.register_error: str | None = None
        self.is_registered = False
        self.smaps_register = None
        if not bench.rehearsal:
            try:
                bench.register(self.address, self.registered, reg_flags)
                self.is_registered = True
            except Exception as exc:                                   # recorded, never fatal for the session
                self.register_error = f"{type(exc).__name__}: {str(exc)[:120]}"
        self.smaps_register = self.smaps_entry()                        # after registration (equals smaps_make on the rehearsal)

    def smaps_entry(self):
        try:
            text = Path("/proc/self/smaps").read_text()
        except OSError:
            return None
        return parse_smaps(text, self.address)

    def release(self) -> None:
        if self.is_registered:
            with contextlib.suppress(Exception):
                self.bench.unregister(self.address)
            self.is_registered = False
        self.tensor = None
        self._cbuf = None
        with contextlib.suppress(BufferError, ValueError):
            self.map.close()


class BackingBench:
    """The GPU cells. Imports torch lazily. `rehearsal_cpu` runs the same code on the CPU (Mac tests only)."""

    def __init__(self, device_index: int = 0, rehearsal_cpu: bool = False):
        import torch

        self.torch, self.rehearsal = torch, rehearsal_cpu
        if rehearsal_cpu:
            self.dev = torch.device("cpu")
            return
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        self.dev = torch.device("cuda", device_index)
        torch.cuda.set_device(self.dev)
        self._cudart = torch.cuda.cudart()
        self._lib = None
        if not hasattr(self._cudart, "cudaHostRegister"):                 # older torch builds: call the runtime directly
            path = find_cudart_lib(sys.path)
            if path is None:
                raise RuntimeError("torch.cuda.cudart() has no cudaHostRegister and no libcudart was found")
            self._lib = ctypes.CDLL(path)
            self._lib.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            self._lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]

    # -- primitives
    def _events(self):
        if self.rehearsal:
            return _CpuEvent(), _CpuEvent()
        return self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)

    def sync(self) -> None:
        if not self.rehearsal:
            self.torch.cuda.synchronize()

    def register(self, address: int, nbytes: int, flags: int = 1) -> None:
        fn = self._lib.cudaHostRegister if self._lib is not None else self._cudart.cudaHostRegister
        r = int(fn(address, nbytes, flags))                            # flags 1 = cudaHostRegisterPortable, 0 = none
        if r != 0:
            raise RuntimeError(f"cudaHostRegister returned {r}")

    def unregister(self, address: int) -> None:
        fn = self._lib.cudaHostUnregister if self._lib is not None else self._cudart.cudaHostUnregister
        r = int(fn(address))
        if r != 0:
            raise RuntimeError(f"cudaHostUnregister returned {r}")

    def pinned(self, nbytes: int, seed: int):
        g = self.torch.Generator().manual_seed(seed)
        x = self.torch.randint(0, 256, (nbytes,), dtype=self.torch.uint8, generator=g)
        return x if self.rehearsal else x.pin_memory()

    def empty_host_cache(self) -> None:
        fn = getattr(self.torch._C, "_host_emptyCache", None)
        if fn is not None:
            with contextlib.suppress(Exception):
                fn()

    def dev_buffer(self, nbytes: int):
        return self.torch.empty(nbytes, dtype=self.torch.uint8, device=self.dev)

    def time_copy(self, dst, src, iters: int = 20, warm: int = 5, d2h: bool = False) -> float:
        """p50 CUDA-event ms of one copy, the GPU idle between repetitions."""
        a, b = self._events()
        out = []
        for i in range(warm + iters):
            self.sync()
            a.record()
            if d2h:
                src.copy_(dst, non_blocking=True)
            else:
                dst.copy_(src, non_blocking=True)
            b.record()
            b.synchronize()
            if i >= warm:
                out.append(a.elapsed_time(b))
        return p50(out)

    def roundtrip_ok(self, dev, host, n: int) -> bool:
        self.sync()
        return bool(self.torch.equal(dev[:n].cpu(), host[:n]))

    # -- cells (each returns a plain dict; sizes are parameters so the Mac tests can run them tiny)
    def alloc_order(self, k: int = 48, nbytes: int = RECORD_BYTES, deadline: suite.Deadline | None = None) -> dict:
        dev = self.dev_buffer(nbytes)
        keep, ms, ok = [], [], True
        for i in range(k):
            if deadline is not None and deadline.expired():
                break
            buf = self.pinned(nbytes, SEED + i)
            keep.append(buf)
            ms.append(self.time_copy(dev, buf))
            ok &= self.roundtrip_ok(dev, buf, nbytes)
        return {"ms": ms, "roundtrip_ok": ok, "nbytes": nbytes, "kept": len(keep)}

    def mmap_cell(self, advice: str, k: int = 16, nbytes: int = RECORD_BYTES, deadline: suite.Deadline | None = None) -> dict:
        dev = self.dev_buffer(nbytes)
        seed = self.torch.randint(0, 256, (((nbytes + HUGE - 1) // HUGE) * HUGE,), dtype=self.torch.uint8)
        bufs, ms, frac, ok, errors = [], [], [], True, []
        for _ in range(k):
            if deadline is not None and deadline.expired():
                break
            mb = MmapBuffer(self, nbytes, advice, seed)
            bufs.append(mb)
            if mb.advice_error:
                errors.append(mb.advice_error)
            if mb.register_error:
                errors.append(mb.register_error)
                continue
            ms.append(self.time_copy(dev, mb.tensor[:nbytes]))
            ok &= self.roundtrip_ok(dev, mb.tensor, nbytes)
            frac.append(huge_fraction(mb.smaps_entry(), mb.registered))
        for mb in bufs:
            mb.release()
        return {"ms": ms, "huge_fraction": frac, "roundtrip_ok": ok, "errors": sorted(set(errors)), "nbytes": nbytes}

    def private_cell(self, advice: str, mapping: str = "private", populate: bool = False, k: int = 16,
                     nbytes: int = RECORD_BYTES, deadline: suite.Deadline | None = None, reg_flags: int = 1) -> dict:
        """B2 cells H1 to H5: k buffers on a self-made mapping, each timed alone, smaps read after the content is made, after
        registration and after the timed copies (three points per buffer; arrays aligned with `ms`)."""
        dev = self.dev_buffer(nbytes)
        seed = self.torch.randint(0, 256, (((nbytes + HUGE - 1) // HUGE) * HUGE,), dtype=self.torch.uint8)
        bufs, ms, points, ok, errors = [], [], [], True, []
        unavailable = None
        for _ in range(k):
            if deadline is not None and deadline.expired():
                break
            mb = MmapBuffer(self, nbytes, advice, seed, mapping=mapping, populate=populate, reg_flags=reg_flags)
            bufs.append(mb)
            for e in (mb.advice_error, mb.populate_error, mb.register_error):
                if e:
                    errors.append(e)
            if mb.populate_error:
                unavailable = mb.populate_error                         # said, and the cell stops: this is not an H5 buffer
                break
            if mb.register_error:
                continue
            ms.append(self.time_copy(dev, mb.tensor[:nbytes]))
            ok &= self.roundtrip_ok(dev, mb.tensor, nbytes)             # this buffer's own copy, as in the curve fix
            points.append({"make": mb.smaps_make, "register": mb.smaps_register, "copy": mb.smaps_entry()})
        registered = bufs[0].registered if bufs else ((nbytes + HUGE - 1) // HUGE) * HUGE
        for mb in bufs:
            mb.release()
        return {"ms": ms, "smaps": points, "roundtrip_ok": ok, "errors": sorted(set(errors)), "nbytes": nbytes,
                "registered": registered, "mapping": mapping, "advice": advice, "populate": populate, "reg_flags": reg_flags,
                "unavailable": unavailable}

    def prefix_cell(self, small_sizes=(MIB, 3 * MIB), k_small: int = 16, k: int = 48, nbytes: int = RECORD_BYTES,
                    deadline: suite.Deadline | None = None) -> dict:
        """B2 cell H6: smaller pinned buffers first (kept alive), then k buffers of `nbytes`, each timed alone; the byte check
        follows each buffer's own copy."""
        keep, ok, prefix_ms = [], True, {}
        for size in small_sizes:
            dev = self.dev_buffer(size)
            ms_small = []
            for i in range(k_small):
                if deadline is not None and deadline.expired():
                    break
                buf = self.pinned(size, SEED + 500 + i)
                keep.append(buf)
                ms_small.append(self.time_copy(dev, buf))
                ok &= self.roundtrip_ok(dev, buf, size)
            prefix_ms[str(size)] = ms_small
        main = self.alloc_order(k, nbytes, deadline=deadline)
        main["roundtrip_ok"] = bool(main["roundtrip_ok"] and ok)
        main["prefix_ms"] = prefix_ms
        del keep
        return main

    def reroll(self, advice: str, cycles: int = 12, nbytes: int = RECORD_BYTES) -> dict:
        dev = self.dev_buffer(nbytes)
        seed = self.torch.randint(0, 256, (((nbytes + HUGE - 1) // HUGE) * HUGE,), dtype=self.torch.uint8)
        ms, frac, errors = [], [], []
        for _ in range(cycles):
            mb = MmapBuffer(self, nbytes, advice, seed)
            if mb.advice_error or mb.register_error:
                errors.append(mb.advice_error or mb.register_error)
                mb.release()
                continue
            ms.append(self.time_copy(dev, mb.tensor[:nbytes]))
            frac.append(huge_fraction(mb.smaps_entry(), mb.registered))
            mb.release()
        return {"ms": ms, "huge_fraction": frac, "errors": sorted(set(errors)), "nbytes": nbytes}

    def slab_chunks(self, slab_bytes: int = 256 * MIB, chunk_bytes: int = 4 * MIB) -> dict:
        slab = self.pinned(slab_bytes, SEED)
        dev = self.dev_buffer(slab_bytes)
        n = slab_bytes // chunk_bytes
        chunk_ms = [self.time_copy(dev[i * chunk_bytes:(i + 1) * chunk_bytes], slab[i * chunk_bytes:(i + 1) * chunk_bytes])
                    for i in range(n)]
        whole = self.time_copy(dev, slab, iters=10, warm=3)
        return {"chunk_ms": chunk_ms, "whole_ms": whole, "slab_bytes": slab_bytes, "chunk_bytes": chunk_bytes,
                "roundtrip_ok": self.roundtrip_ok(dev, slab, slab_bytes)}

    def curve(self, sizes=CURVE_SIZES, ks=CURVE_K, deadline: suite.Deadline | None = None) -> dict:
        """Per size: K pinned buffers each timed alone, a pinned D2H at the fastest buffer's size, four pageable buffers. A size
        whose allocation fails (the locked-memory limit) is recorded as an error and the next size still runs."""
        out = {}
        for size, k in zip(sizes, ks, strict=True):
            if deadline is not None and deadline.expired():
                break
            try:
                dev = self.dev_buffer(size)
                bufs = [self.pinned(size, SEED + i) for i in range(k)]
                # the device buffer holds the content of the buffer copied last, so the byte check follows each of the first
                # two buffers' own copy (checking after the loop compared two buffers with the last one's content)
                ms, ok = [], True
                for i, b in enumerate(bufs):
                    ms.append(self.time_copy(dev, b))
                    if i < 2:
                        ok &= self.roundtrip_ok(dev, b, size)
                d2h_dst = self.pinned(size, SEED + 99)
                d2h = self.time_copy(dev, d2h_dst, d2h=True)
                pageable = []
                for i in range(min(4, k)):
                    g = self.torch.Generator().manual_seed(SEED + 200 + i)
                    pageable.append(self.time_copy(dev, self.torch.randint(0, 256, (size,), dtype=self.torch.uint8, generator=g)))
                out[str(size)] = {"pinned_ms": ms, "d2h_fastest_ms": d2h, "pageable_ms": pageable, "roundtrip_ok": ok, "k": k,
                                  "fastest_index": int(min(range(k), key=lambda i: ms[i]))}
                del bufs, d2h_dst
            except RuntimeError as exc:
                out[str(size)] = {"error": f"{type(exc).__name__}: {str(exc)[:150]}", "k": k}
            self.empty_host_cache()
        return out


# ------------------------------------------------------------------------------------------------ rows

def _row(run_id, arm, field, value, unit, tag, source, **kw):
    return mf.row(run_id, arm, field, value, unit, tag, source=source, **kw)


def rows_alloc_order(run_id: str, arm: str, res: dict) -> list[dict]:
    ms = res["ms"]
    if not ms:
        return [_row(run_id, arm, "slow_share", None, "fraction", "unavailable", "cell", note="no buffers timed")]
    fl = slow_flags(ms, SLOW_MS_9_95MB)
    fast = [x for x, f in zip(ms, fl, strict=True) if not f]
    slow = [x for x, f in zip(ms, fl, strict=True) if f]
    rows = [_row(run_id, arm, "slow_share", share(fl), "fraction", "measured", "p50 per buffer, slow = at or above 0.28 ms", n=len(ms)),
            _row(run_id, arm, "third_mode_share", third_mode_share(ms), "fraction", "measured", "buffers between 0.20 and 0.28 ms", n=len(ms)),
            _row(run_id, arm, "copy_ms_min", min(ms), "ms", "measured", "cuda events p50 of the fastest buffer"),
            _row(run_id, arm, "copy_ms_max", max(ms), "ms", "measured", "cuda events p50 of the slowest buffer")]
    if fast:
        rows.append(_row(run_id, arm, "fast_median_ms", p50(fast), "ms", "measured", "cuda events", n=len(fast)))
    if slow:
        rows.append(_row(run_id, arm, "slow_median_ms", p50(slow), "ms", "measured", "cuda events", n=len(slow)))
    g = order_gap(fl)
    if g is None:
        rows.append(_row(run_id, arm, "order_gap", None, "index", "unavailable", "cell", note="one class only"))
    else:
        rows.append(_row(run_id, arm, "order_gap", g, "index", "derived", "mean allocation index of slow minus fast buffers"))
    lag = lag1_agreement(fl)
    if lag is not None:
        rows.append(_row(run_id, arm, "lag1_agreement", lag, "fraction", "derived", "consecutive buffers in the same class"))
        rows.append(_row(run_id, arm, "lag1_expected_if_independent", expected_agreement(share(fl)), "fraction", "derived",
                         "p^2 + (1-p)^2"))
    return rows


def rows_mmap_cell(run_id: str, arm: str, res: dict) -> list[dict]:
    rows = rows_alloc_order(run_id, arm, res)
    frac = [f for f in res.get("huge_fraction", []) if f is not None]
    if frac:
        verified = [i for i, f in enumerate(res["huge_fraction"]) if f is not None and f >= 0.9]
        rows.append(_row(run_id, arm, "huge_verified_count", len(verified), "count", "measured",
                         "smaps AnonHugePages at 90 percent or more of the registered range", n=len(res["huge_fraction"])))
        rows.append(_row(run_id, arm, "huge_fraction_median", p50(frac), "fraction", "measured", "smaps AnonHugePages over the registered range"))
        if verified and res["ms"]:
            vs = [res["ms"][i] for i in verified if i < len(res["ms"])]
            rows.append(_row(run_id, arm, "huge_verified_slow_count", sum(1 for x in vs if x >= SLOW_MS_9_95MB), "count", "measured",
                             "slow buffers among the huge-page-verified ones", n=len(vs)))
    else:
        rows.append(_row(run_id, arm, "huge_fraction_median", None, "fraction", "unavailable", "smaps", note="smaps not readable here"))
    for e in res.get("errors", []):
        rows.append(_row(run_id, arm, "advice_or_register_error", None, "text", "unavailable", "cell", note=e[:150]))
    return rows


def rows_private_cell(run_id: str, arm: str, res: dict) -> list[dict]:
    """Rows of a B2 private/shared mapping cell (H1 to H5). `huge_verified_count` is at the post-make read (R-P1)."""
    rows = rows_alloc_order(run_id, arm, res)
    reg = res["registered"]
    pts = res["smaps"]
    for e in res.get("errors", []):
        rows.append(_row(run_id, arm, "advice_or_register_error", None, "text", "unavailable", "cell", note=e[:150]))
    if res.get("unavailable"):
        rows.append(_row(run_id, arm, "populate_unavailable", None, "text", "unavailable", "cell", note=res["unavailable"][:150]))
    anon = {w: [huge_fraction(p.get(w), reg) for p in pts] for w in ("make", "register", "copy")}
    shm = [shmem_fraction(p.get("make"), reg) for p in pts]
    if not pts or all(f is None for f in anon["make"]):
        rows.append(_row(run_id, arm, "huge_fraction_median", None, "fraction", "unavailable", "smaps", note="smaps not readable here"))
        return rows
    names = {"make": "huge_verified_count", "register": "huge_verified_count_post_register", "copy": "huge_verified_count_post_copy"}
    medn = {"make": "huge_fraction_median", "register": "huge_fraction_median_post_register", "copy": "huge_fraction_median_post_copy"}
    for w in ("make", "register", "copy"):
        fl = [f for f in anon[w] if f is not None]
        if not fl:
            continue
        rows.append(_row(run_id, arm, names[w], sum(1 for f in fl if f >= 0.9), "count", "measured",
                         f"smaps AnonHugePages at 90 percent or more of the registered range, read at the {w} point", n=len(fl)))
        rows.append(_row(run_id, arm, medn[w], p50(fl), "fraction", "measured", f"smaps AnonHugePages, {w} point"))
    sh = [f for f in shm if f is not None]
    if sh:
        rows.append(_row(run_id, arm, "shmem_pmd_verified_count", sum(1 for f in sh if f >= 0.9), "count", "measured",
                         "smaps ShmemPmdMapped at 90 percent or more of the registered range, make point", n=len(sh)))
    drops = [a - b for a, b in zip(anon["make"], anon["register"], strict=True)
             if a is not None and b is not None and a >= 0.9]
    if drops:
        rows.append(_row(run_id, arm, "huge_fraction_max_drop_after_register", max(drops), "fraction", "derived",
                         "largest fall of the huge fraction from the make to the register point among verified buffers", n=len(drops)))
    ms = res["ms"]
    verified = [i for i, f in enumerate(anon["make"]) if f is not None and f >= 0.9 and i < len(ms)]
    small = [i for i, f in enumerate(anon["make"]) if f is not None and f < 0.9 and i < len(ms)]
    if verified:
        vs = [ms[i] for i in verified]
        rows.append(_row(run_id, arm, "huge_verified_slow_count", sum(1 for x in vs if x >= SLOW_MS_9_95MB), "count", "measured",
                         "slow buffers among the huge-page-verified ones", n=len(vs)))
        rows.append(_row(run_id, arm, "huge_verified_ms_max", max(vs), "ms", "measured", "slowest huge-page-verified buffer"))
        rows.append(_row(run_id, arm, "huge_verified_ms_min", min(vs), "ms", "measured", "fastest huge-page-verified buffer"))
    if small:
        rows.append(_row(run_id, arm, "small_page_slow_share", share([ms[i] >= SLOW_MS_9_95MB for i in small]), "fraction", "measured",
                         "slow share among the buffers not huge-page-verified", n=len(small)))
    return rows


def rows_prefix(run_id: str, arm: str, res: dict) -> list[dict]:
    rows = rows_alloc_order(run_id, arm, res)
    for size, ms in res.get("prefix_ms", {}).items():
        if ms:
            rows.append(_row(run_id, f"{arm}.{size}", "prefix_h2d_gbps_median", gbps(int(size), p50(ms)), "GB/s", "measured",
                             "median of the smaller buffers allocated first", n=len(ms)))
    return rows


def rows_slab(run_id: str, arm: str, res: dict) -> list[dict]:
    ms, cb = res["chunk_ms"], res["chunk_bytes"]
    fl = slow_flags(ms)
    fast = [x for x, f in zip(ms, fl, strict=True) if not f]
    slow = [x for x, f in zip(ms, fl, strict=True) if f]
    rows = [_row(run_id, arm, "slow_chunk_share", share(fl), "fraction", "measured", "chunks below 0.8 x the fastest chunk's rate", n=len(ms)),
            _row(run_id, arm, "whole_slab_gbps", gbps(res["slab_bytes"], res["whole_ms"]), "GB/s", "measured", "one copy of the slab")]
    runs = run_lengths(fl)
    rows.append(_row(run_id, arm, "slow_run_mean_length", (sum(runs) / len(runs)) if runs else 0.0, "chunks", "derived", "runs of consecutive slow chunks"))
    if fast:
        rows.append(_row(run_id, arm, "fast_chunk_gbps_median", gbps(cb, p50(fast)), "GB/s", "measured", "cuda events", n=len(fast)))
    if slow:
        rows.append(_row(run_id, arm, "slow_chunk_gbps_median", gbps(cb, p50(slow)), "GB/s", "measured", "cuda events", n=len(slow)))
        if fast:
            # time-weighted whole-slab rate implied by the chunk classes
            implied = res["slab_bytes"] / 1e9 / ((sum(ms)) / 1e3)
            rows.append(_row(run_id, arm, "whole_slab_gbps_implied_by_chunks", implied, "GB/s", "derived", "slab bytes over the sum of chunk times"))
    return rows


def rows_curve(run_id: str, arm: str, res: dict) -> list[dict]:
    rows = []
    for size, d in res.items():
        size_b = int(size)
        a = f"{arm}.{size_b}"
        if "error" in d:
            rows.append(_row(run_id, a, "pinned_h2d_gbps_fastest", None, "GB/s", "unavailable", "cell", note=d["error"]))
            continue
        ms = d["pinned_ms"]
        fastest, slowest, med = min(ms), max(ms), p50(ms)
        rows += [_row(run_id, a, "pinned_h2d_gbps_fastest", gbps(size_b, fastest), "GB/s", "measured", "fastest of the K buffers", n=len(ms)),
                 _row(run_id, a, "pinned_h2d_gbps_median", gbps(size_b, med), "GB/s", "measured", "median of the K buffers", n=len(ms)),
                 _row(run_id, a, "pinned_h2d_gbps_slowest", gbps(size_b, slowest), "GB/s", "measured", "slowest of the K buffers", n=len(ms)),
                 _row(run_id, a, "slow_share", share(slow_flags(ms)), "fraction", "measured", "buffers below 0.8 x the fastest buffer's rate", n=len(ms)),
                 _row(run_id, a, "pinned_d2h_gbps_fastest_buffer", gbps(size_b, d["d2h_fastest_ms"]), "GB/s", "measured", "one pinned destination")]
        if d["pageable_ms"]:
            rows.append(_row(run_id, a, "pageable_h2d_gbps_median", gbps(size_b, p50(d["pageable_ms"])), "GB/s", "measured", "median of 4 pageable buffers", n=len(d["pageable_ms"])))
    return rows


# ------------------------------------------------------------------------------------------------ CLI

GROUPS = {"b1": "alloc_order x1 (run the group twice, in two processes)",
          "rest": "thp, nothp, default, reroll(default, thp), slab_chunks (B1: SHARED mappings, see the B1 results section 6)",
          "curve": "pinned/pageable/D2H curve with K buffers per size",
          "h1": "B2 H1: 16 buffers, private mapping, MADV_HUGEPAGE before the touch",
          "h2": "B2 H2: 16 buffers, private mapping, MADV_NOHUGEPAGE",
          "h3": "B2 H3: 16 buffers, private mapping, no advice",
          "h4": "B2 H4: 16 buffers, SHARED mapping (B1's), MADV_HUGEPAGE, with ShmemPmdMapped read",
          "h5": "B2 H5: 16 buffers, private mapping, MADV_HUGEPAGE then MADV_POPULATE_WRITE (unavailable is a valid outcome)",
          "h6": "B2 H6: 16 x 1 MiB and 16 x 3 MiB cudaHostAlloc first, then 48 x 9.95 MB, each timed alone",
          "h7": "B2 H7: 48 x 9.95 MB cudaHostAlloc in a fresh process (the control for H6)"}
B2_GROUPS = ("h1", "h2", "h3", "h4", "h5", "h6", "h7")
# Sizes for the Mac rehearsal tests only: every group runs through the real _run_group, gates included.
SMALL = {"k_alloc": 6, "nbytes": 1 << 16, "k_map": 3, "k_reroll": 3, "slab": (1 << 20, 1 << 16), "curve": ((1 << 16, 1 << 17), (3, 2)),
         "k_small": 2, "small_sizes": (1 << 14, 1 << 15), "k_prefix": 4}


def cmd_plan(args) -> int:
    for g, what in GROUPS.items():
        print(f"{g:6s} {what}")
    print(f"curve sizes {list(CURVE_SIZES)} with K = {list(CURVE_K)}")
    return 0


def _run_group(bench, group: str, run_id: str, dl: suite.Deadline, gates: dict, failures: list, raw: dict,
               small: bool = False) -> list[dict]:
    rows: list[dict] = []
    cfg = SMALL if small else {"k_alloc": 48, "nbytes": RECORD_BYTES, "k_map": 16, "k_reroll": 12, "slab": (256 * MIB, 4 * MIB),
                               "curve": (CURVE_SIZES, CURVE_K), "k_small": 16, "small_sizes": (MIB, 3 * MIB), "k_prefix": 48}

    def guarded(name, fn, row_fn):
        try:
            res = fn()
        except Exception as exc:
            failures.append({"cell": name, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            return
        raw[name] = res
        if "roundtrip_ok" in res and not res["roundtrip_ok"]:
            gates["roundtrip"] = False
        if name == "curve" and any(not d.get("roundtrip_ok", True) for d in res.values()):
            gates["roundtrip"] = False
        if res.get("unavailable"):
            raw.setdefault("unavailable", {})[name] = res["unavailable"]       # a valid outcome, never a failed cell
        if name == "curve" and any("error" in d for d in res.values()):
            failures.append({"cell": "curve", "error": "at least one size could not be allocated: " + "; ".join(
                f"{k}: {d['error']}" for k, d in res.items() if "error" in d)[:300]})
        rows.extend(row_fn(run_id, f"backing.{name}", res))

    nb = cfg["nbytes"]
    if group == "b1":
        guarded("alloc_order", lambda: bench.alloc_order(cfg["k_alloc"], nb, deadline=dl), rows_alloc_order)
    elif group == "rest":
        for advice in ("thp", "nothp", "default"):
            guarded(f"mmap_{advice}", lambda a=advice: bench.mmap_cell(a, cfg["k_map"], nb, deadline=dl), rows_mmap_cell)
            bench.empty_host_cache()
        for advice in ("default", "thp"):
            guarded(f"reroll_{advice}", lambda a=advice: bench.reroll(a, cfg["k_reroll"], nb), rows_mmap_cell)
        guarded("slab_chunks", lambda: bench.slab_chunks(*cfg["slab"]), rows_slab)
    elif group == "curve":
        guarded("curve", lambda: bench.curve(*cfg["curve"], deadline=dl), rows_curve)
    elif group in ("h1", "h2", "h3", "h4", "h5"):
        advice, mapping, populate = {"h1": ("thp", "private", False), "h2": ("nothp", "private", False),
                                     "h3": ("default", "private", False), "h4": ("thp", "shared", False),
                                     "h5": ("thp", "private", True)}[group]
        name = {"h1": "private_thp", "h2": "private_nothp", "h3": "private_default", "h4": "shared_thp", "h5": "private_populate"}[group]
        guarded(name, lambda: bench.private_cell(advice, mapping, populate, cfg["k_map"], nb, deadline=dl), rows_private_cell)
    elif group == "h6":
        guarded("prefix", lambda: bench.prefix_cell(cfg["small_sizes"], cfg["k_small"], cfg["k_prefix"], nb, deadline=dl), rows_prefix)
    elif group == "h7":
        guarded("fresh48", lambda: bench.alloc_order(cfg["k_alloc"], nb, deadline=dl), rows_alloc_order)
    return rows


def cmd_run(args) -> int:
    if not args.execute:
        cmd_plan(args)
        print("\n# DRY RUN: nothing executed. Add --execute on the provisioned GPU host.", file=sys.stderr)
        return 0
    import doctor

    report = doctor.collect(note=args.note, region=args.region, session_cap_s=int(args.max_seconds))
    ok, why = doctor.cuda_ready(report)
    if not ok:
        print(f"SKIP: no CUDA run possible here: {why}", file=sys.stderr)
        return 3
    bench = BackingBench(0)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{stamp}-backing-{args.cells}-{report['manifest_id'][:6]}"
    out = Path(args.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    mf.write_json(out / "machine-manifest.json", report)
    dl = suite.Deadline(args.max_seconds)
    gates = {"roundtrip": True, "sanity": True}
    failures: list[dict] = []
    raw: dict = {}
    raw["host_thp"] = host_thp()
    rows = _run_group(bench, args.cells, run_id, dl, gates, failures, raw)
    for r in rows:
        if r["field"].endswith("gbps") or "gbps" in r["field"]:
            if r.get("value") is not None and r["value"] > suite.PCIE_SANITY_GB_S:
                gates["sanity"] = False
    status = "truncated" if dl.expired() else "complete"
    manifest = {
        "schema": mf.SCHEMA_NAME, "schema_version": mf.SCHEMA_VERSION, "platform": mf.PLATFORM, "kind": "run",
        "created_utc": mf.utc_now(), "machine_manifest_id": report["manifest_id"], "run_id": run_id,
        "instrument": {"name": "benchmarks/cuda/backing_probe.py", "version": "B2-1" if args.cells in B2_GROUPS else "B1-1"}, "code": report["code"], "mode": "timing",
        "workload": {"id": f"{'B2' if args.cells in B2_GROUPS else 'B1'}-{args.cells}", "class": "pinned host buffer H2D rate by backing (no model)",
                     "tag": args.tag, "claims": "host-to-device copy rate of pinned buffers only"},
        "protocol": {"repetitions": "20 timed after 5 warm-up, p50 per buffer, GPU idle between", "timer": "cuda events"},
        "limits": {"max_seconds": args.max_seconds, "elapsed_s": round(dl.elapsed(), 3)},
        "gates": {**gates, "failed_cells": failures}, "status": status,
    }
    mf.write_json(out / "run-manifest.json", mf.seal(manifest))
    mf.write_json(out / "raw-samples.json", {"run": run_id, "samples": raw})
    (out / "rows.jsonl").write_text("")
    mf.append_rows(out / "rows.jsonl", rows)
    print(f"run {run_id}: status {status}, {len(rows)} rows, {dl.elapsed():.1f} s -> {out}")
    bad = [k for k, v in gates.items() if not v] + ([f"{len(failures)} failed cell(s)"] if failures else [])
    if bad:
        print("FAIL: " + "; ".join(bad), file=sys.stderr)
        return 4
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("--cells", choices=tuple(GROUPS), required=True)
    r.add_argument("--execute", action="store_true")
    r.add_argument("--max-seconds", type=float, default=300.0)
    r.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "cuda"))
    r.add_argument("--note", default="")
    r.add_argument("--region")
    r.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
