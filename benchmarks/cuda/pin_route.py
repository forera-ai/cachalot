"""
Session K instrument, part 1 (docs/CUDA-PIN-ROUTE-RECORD.md): the pinning route of a 9.95 MB host buffer (DC-3) and the committed
classification of an instance's copy population (DC-6). The issue-loop synchronisation guard (DC-5) is in `overlap_arena.py`
(profiles `k-smoke`, `k-full`).

Groups, each its own fresh process (the allocation context is one of the things under test):
  k0, k5  classification at the start and at the end of the session: twelve `cudaHostAlloc` buffers of 9.95 MB and three of 64 MiB,
          each timed alone, a 256 MiB device-to-device copy, and a read-only `nvidia-smi` link sampler beside the copies.
  k1      24 buffers: a self-made 2 MiB-aligned private mapping, registered with the portable flag (B1 and B2's route).
  k2      the same mapping, registered with flag 0 (DeepInfer's call).
  k3      24 buffers of ordinary allocator memory (`torch.empty`), registered with flag 0 ("memory that already exists").
  k4      24 buffers through `cudaHostAlloc` (torch `pin_memory`), the route of S1 to N3 (the control), with the link sampler.

The committed rule (record section 3.1): a 9.95 MB buffer is slow if its rate is below 0.75 x the highest p50 rate of the three 64 MiB
buffers; the instance is `fast` with at most 1 slow of 12, `slow` with at least 11, `mixed` otherwise. The pure helpers need no
torch; `PinRouteBench` extends `backing_probe.BackingBench` (which imports torch lazily and has a CPU rehearsal path for the Mac tests,
never a CUDA number).

Follow-up L (docs/CUDA-PIN-ROUTE-FOLLOWUP-RECORD.md, what makes the arena process's pinned copies fast): groups l1 to l4, each its own
fresh process, 24 `cudaHostAlloc` buffers of 9.95 MB timed alone, varying three factors of the arena process one at a time and together:
  l1      all buffers allocated first, then timed (a 9.95 MB destination, default stream)
  l2      interleaved, the destination is slot 0 of a 24 GiB device buffer allocated first
  l3      interleaved, timed on a non-default stream
  l4      all three together (the analogue of the arena's source qualification)

Follow-up M (docs/CUDA-PIN-ORDER-RECORD.md, why is an allocate-then-copy buffer slow, and the huge-page, route and classification answers in the
fast pattern): groups m0 and m9 (the committed classification rule on the batch-first pattern, at the start and the end), m1 (12 buffers, one
copy, 12 more, all timed afterwards), m4 (one copy of another buffer first, then 24 buffers allocated and timed), m6 and m7 (24 private
mappings made before any is timed, huge-page advice and its opposite, registered), m8 (24 ordinary buffers made first, registered with flag 0).

Follow-up N (docs/CUDA-PIN-SETTLE-RECORD.md, is a just-allocated buffer slow for a time, or while its bytes are in the CPU's caches): groups
n1 (a 1,000 ms train of single timed copies after each allocation), n2 (allocate, idle T ms, time; T swept), n3 (allocate, E bytes of
unrelated CPU writes, time) and n4 (buffers aged 500 ms, then untouched, read or rewritten by the CPU, then timed).

Usage:
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/pin_route.py plan
GPU host: `run --cells k0|k1|k2|k3|k4|k5|l1|l2|l3|l4|m0|m1|m4|m6|m7|m8|m9|n1|n2|n3|n4 [--execute] [--tag NAME] [--out-dir DIR]`.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backing_probe as bp  # noqa: E402
import manifest as mf  # noqa: E402
import suite  # noqa: E402

MIB = bp.MIB
RECORD_BYTES = bp.RECORD_BYTES
PLATEAU_BYTES = 64 * MIB
STREAM_BYTES = 256 * MIB
SLOW_FRACTION = 0.75                 # slow = rate below this fraction of the 64 MiB plateau rate (record section 3.1)
K_CLASSIFY, K_PLATEAU, K_ROUTE = 12, 3, 24
CLASS_CODE = {"fast": 0, "mixed": 1, "slow": 2}
STREAM_REL, STREAM_ABS_US = 0.15, 3.0   # "differs significantly": more than 15 % and more than 3 microseconds (DM-4)
LINK_QUERY = "pcie.link.gen.current,pcie.link.width.current"
SEED = bp.SEED

GROUPS = {"k0": "DC-6 start: classify the instance (12 x 9.95 MB, 3 x 64 MiB, device copy, link sampler)",
          "k1": "DC-3: 24 x 9.95 MB, private 2 MiB-aligned mapping, cudaHostRegister portable (flag 1)",
          "k2": "DC-3: the same mapping, cudaHostRegister flag 0 (DeepInfer's call)",
          "k3": "DC-3: 24 x 9.95 MB ordinary allocator memory (torch.empty), cudaHostRegister flag 0",
          "k4": "DC-3 control: 24 x 9.95 MB cudaHostAlloc (torch pin_memory), with the link sampler",
          "k5": "DC-6 end: classify again (the same cell as k0, a fresh process)"}
GROUPS.update({"l1": "follow-up L: 24 pinned buffers allocated first, then timed (small destination, default stream)",
               "l2": "follow-up L: interleaved, destination is slot 0 of a 24 GiB device buffer allocated first",
               "l3": "follow-up L: interleaved, timed on a non-default stream",
               "l4": "follow-up L: all of l1, l2 and l3 together (the arena's source-qualification pattern)"})
GROUPS.update({"m0": "follow-up M: the committed classification rule, batch-first pattern (session start)",
               "m1": "follow-up M: 12 pinned buffers, one H2D copy, 12 more, then time all 24 in allocation order",
               "m4": "follow-up M: one H2D copy of a separate 1 MiB pinned buffer, then allocate 24 buffers, then time them",
               "m6": "follow-up M: 24 private mappings, MADV_HUGEPAGE, registered, all made before any is timed",
               "m7": "follow-up M: the same with MADV_NOHUGEPAGE (control for m6)",
               "m8": "follow-up M: 24 ordinary allocator buffers registered with flag 0, all made before timing",
               "m9": "follow-up M: the committed classification rule, batch-first pattern (session end)"})
GROUPS.update({"n1": "follow-up N: 6 buffers, each allocated then copied back to back for 1,000 ms, every copy timed",
               "n2": "follow-up N: 28 buffers, each allocated, left idle for T ms, then timed (T swept 0 to 500)",
               "n3": "follow-up N: 24 buffers, each allocated, E bytes of unrelated CPU writes, then timed (E 0, 64, 256 MiB)",
               "n4": "follow-up N: 24 buffers allocated first and left 500 ms, then each untouched, read or rewritten by the CPU, then timed"})
ORDER = ("k0", "k1", "k2", "k3", "k4", "k5")
ORDER_N = ("k4", "l1", "n1", "n2", "n3", "n4", "k4")
WAITS_MS = (0, 500, 5, 100, 20, 200, 50, 0, 500, 100, 5, 50, 200, 20, 500, 0, 50, 5, 200, 20, 100, 0, 500, 20, 5, 100, 200, 50)   # four of each T, fixed shuffle
EVICT_BYTES = (0, 64 * MIB, 256 * MIB)
RETOUCH = ("none", "read", "write")
ORDER_M = ("m0", "k4", "l1", "m1", "m4", "m6", "m7", "m8", "m9")
FOLLOWUP_ORDER = ("k4", "k3", "k2", "k1", "l1", "l2", "l3", "l4", "k4")     # reversed routes, the four new cells, the control again
ARENA_BYTES = 24 * 2**30
CONTEXT = {"l1": (True, False, False), "l2": (False, True, False), "l3": (False, False, True), "l4": (True, True, True)}   # batch_first, arena, side stream
SMALL = {"arena_bytes": 1 << 20, "k_classify": 4, "k_plateau": 2, "k_route": 3, "nbytes": 1 << 16, "plateau_bytes": 1 << 18, "stream_bytes": 1 << 18,
         "k_train": 2, "train_ms": 5.0, "waits_ms": (0, 2, 0, 2), "evict_bytes": (0, 1 << 16, 1 << 18), "evict_reps": 2, "age_ms": 2.0}


# ------------------------------------------------------------------------------------------------ pure helpers (no torch)

def classify_buffers(ms_record: list[float], plateau_ms: list[float], nbytes: int = RECORD_BYTES, plateau_bytes: int = PLATEAU_BYTES,
                     fraction: float = SLOW_FRACTION) -> dict:
    """The committed rule. `plateau_gbps` is the highest p50 rate of the plateau buffers; a record buffer is slow below
    `fraction` x that rate. With no plateau buffer nothing can be classified (instance_class 'unavailable')."""
    if not ms_record or not plateau_ms:
        return {"plateau_gbps": None, "slow": [], "slow_count": None, "instance_class": "unavailable"}
    plateau = max(bp.gbps(plateau_bytes, m) for m in plateau_ms)
    slow = [bp.gbps(nbytes, m) < fraction * plateau for m in ms_record]
    return {"plateau_gbps": plateau, "slow": slow, "slow_count": sum(slow), "instance_class": instance_class(slow)}


def instance_class(slow: list[bool]) -> str:
    """`fast` with at most 1 slow of 12, `slow` with at least 11 of 12, `mixed` otherwise (scaled to the number of buffers)."""
    n = len(slow)
    if n == 0:
        return "unavailable"
    s = sum(slow)
    if s <= round(n / 12):
        return "fast"
    if s >= n - round(n / 12):
        return "slow"
    return "mixed"


def label_agreement(relative: list[bool], absolute: list[bool]) -> float | None:
    """Share of buffers for which the committed rule and the absolute 0.28 ms rule give the same label."""
    if not relative or len(relative) != len(absolute):
        return None
    return sum(1 for a, b in zip(relative, absolute, strict=True) if a == b) / len(relative)


def stream_changed(first_gbps: float, second_gbps: float, nbytes: int = STREAM_BYTES, rel: float = STREAM_REL, abs_us: float = STREAM_ABS_US) -> bool:
    """True if the second reading differs from the first by more than `rel` AND by more than `abs_us` microseconds per copy.
    Rates are read plus write (2 x bytes / time), so the time of one copy is 2 x bytes / rate."""
    t1 = 2 * nbytes / (first_gbps * 1e9) * 1e6
    t2 = 2 * nbytes / (second_gbps * 1e9) * 1e6
    return abs(t2 - t1) > abs_us and abs(t2 - t1) / t1 > rel


def parse_link_lines(text: str) -> dict:
    """Summarise the sampler's csv lines ('5, 16' and the like; a header or blank line is skipped)."""
    gens: dict[int, int] = {}
    widths: dict[int, int] = {}
    n = 0
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            continue
        g, w = int(parts[0]), int(parts[1])
        gens[g] = gens.get(g, 0) + 1
        widths[w] = widths.get(w, 0) + 1
        n += 1
    return {"samples": n, "gens": {str(k): v for k, v in sorted(gens.items())}, "widths": {str(k): v for k, v in sorted(widths.items())},
            "min_gen": min(gens) if gens else None, "max_gen": max(gens) if gens else None}


class LinkSampler:
    """A read-only `nvidia-smi` loop in a separate process for the duration of a cell. `unavailable` (with the reason) when the tool
    is missing or does not report the fields; never fatal."""

    def __init__(self, interval_ms: int = 100):
        self.interval_ms, self.proc, self.error = interval_ms, None, None

    def start(self) -> None:
        exe = shutil.which("nvidia-smi")
        if exe is None:
            self.error = "nvidia-smi not found"
            return
        try:
            self.proc = subprocess.Popen([exe, f"--query-gpu={LINK_QUERY}", "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> dict:
        if self.proc is None:
            return {"unavailable": self.error or "not started", "samples": 0}
        self.proc.terminate()
        try:
            out, _ = self.proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            out, _ = self.proc.communicate()
        summary = parse_link_lines(out or "")
        if summary["samples"] == 0:
            summary["unavailable"] = "the sampler returned no readable sample"
        summary["interval_ms"] = self.interval_ms
        return summary


# ------------------------------------------------------------------------------------------------ the bench

class OrdinaryBuffer:
    """Ordinary allocator memory (`torch.empty` on the CPU, not 2 MiB aligned, no advice) registered with `cudaHostRegister`."""

    def __init__(self, bench, nbytes: int, seed_tensor, reg_flags: int):
        self.bench, self.nbytes = bench, nbytes
        self.tensor = bench.torch.empty(nbytes, dtype=bench.torch.uint8)
        self.tensor.copy_(seed_tensor[:nbytes])                     # touches every page
        self.address = self.tensor.data_ptr()
        self.registered = nbytes
        self.smaps_make = self.smaps_entry()
        self.register_error: str | None = None
        self.is_registered = False
        if not bench.rehearsal:
            try:
                bench.register(self.address, nbytes, reg_flags)
                self.is_registered = True
            except Exception as exc:                               # recorded, never fatal for the session
                self.register_error = f"{type(exc).__name__}: {str(exc)[:120]}"
        self.smaps_register = self.smaps_entry()

    def smaps_entry(self):
        try:
            return bp.parse_smaps(Path("/proc/self/smaps").read_text(), self.address)
        except OSError:
            return None

    def release(self) -> None:
        if self.is_registered:
            with contextlib.suppress(Exception):
                self.bench.unregister(self.address)
            self.is_registered = False
        self.tensor = None


class PinRouteBench(bp.BackingBench):
    def ordinary_cell(self, reg_flags: int = 0, k: int = K_ROUTE, nbytes: int = RECORD_BYTES, deadline: suite.Deadline | None = None) -> dict:
        """Group k3: k ordinary buffers, each timed alone, smaps read after the content is made, after registration and after the copies."""
        dev = self.dev_buffer(nbytes)
        seed = self.torch.randint(0, 256, (nbytes,), dtype=self.torch.uint8)
        bufs, ms, points, ok, errors = [], [], [], True, []
        for _ in range(k):
            if deadline is not None and deadline.expired():
                break
            ob = OrdinaryBuffer(self, nbytes, seed, reg_flags)
            bufs.append(ob)
            if ob.register_error:
                errors.append(ob.register_error)
                continue
            ms.append(self.time_copy(dev, ob.tensor))
            ok &= self.roundtrip_ok(dev, ob.tensor, nbytes)
            points.append({"make": ob.smaps_make, "register": ob.smaps_register, "copy": ob.smaps_entry()})
        for ob in bufs:
            ob.release()
        return {"ms": ms, "smaps": points, "roundtrip_ok": ok, "errors": sorted(set(errors)), "nbytes": nbytes, "registered": nbytes,
                "mapping": "ordinary", "advice": "none", "populate": False, "reg_flags": reg_flags, "unavailable": None}

    def time_copy_on(self, dst, src, stream, iters: int = 20, warm: int = 5) -> float:
        """`time_copy` with the copy and its events on `stream` (None: the current stream)."""
        if stream is None:
            return self.time_copy(dst, src, iters=iters, warm=warm)
        a, b = self._events()
        out = []
        for i in range(warm + iters):
            self.sync()
            with self.torch.cuda.stream(stream):
                a.record(stream)
                dst.copy_(src, non_blocking=True)
                b.record(stream)
            b.synchronize()
            if i >= warm:
                out.append(a.elapsed_time(b))
        return bp.p50(out)

    def context_cell(self, batch_first: bool, arena: bool, side_stream: bool, k: int = K_ROUTE, nbytes: int = RECORD_BYTES,
                     arena_bytes: int = ARENA_BYTES, deadline: suite.Deadline | None = None) -> dict:
        """Follow-up L: `k` `cudaHostAlloc` buffers, each timed alone, with the arena process's three differences switched on one at a
        time. `arena` allocates a big device buffer first and copies into its slot 0; `side_stream` times on a non-default stream;
        `batch_first` allocates every pinned buffer before timing any (otherwise each is timed right after it is allocated)."""
        big = self.dev_buffer(arena_bytes) if arena else None
        dev = big[:nbytes] if big is not None else self.dev_buffer(nbytes)
        stream = self.torch.cuda.Stream(device=self.dev) if (side_stream and not self.rehearsal) else None
        keep, ms, ok = [], [], True

        def time_one(buf):
            nonlocal ok
            ms.append(self.time_copy_on(dev, buf, stream))
            ok &= self.roundtrip_ok(dev, buf, nbytes)

        if batch_first:
            keep = [self.pinned(nbytes, SEED + i) for i in range(k)]
            for buf in keep:
                if deadline is not None and deadline.expired():
                    break
                time_one(buf)
        else:
            for i in range(k):
                if deadline is not None and deadline.expired():
                    break
                buf = self.pinned(nbytes, SEED + i)
                keep.append(buf)
                time_one(buf)
        return {"ms": ms, "roundtrip_ok": ok, "nbytes": nbytes, "kept": len(keep),
                "context": {"batch_first": batch_first, "arena": arena, "arena_bytes": arena_bytes if arena else 0, "side_stream": side_stream}}

    def classify_batch_cell(self, k: int = K_CLASSIFY, k_plateau: int = K_PLATEAU, nbytes: int = RECORD_BYTES, plateau_bytes: int = PLATEAU_BYTES,
                            stream_bytes: int = STREAM_BYTES, sampler: LinkSampler | None = None, deadline: suite.Deadline | None = None) -> dict:
        """Groups m0 and m9: the committed rule on the batch-first pattern (every pinned buffer allocated before any is timed)."""
        if sampler is not None:
            sampler.start()
        try:
            recs = [self.pinned(nbytes, SEED + i) for i in range(k)]
            plats = [self.pinned(plateau_bytes, SEED + 700 + i) for i in range(k_plateau)]
            dev, ok, ms, plateau_ms = self.dev_buffer(nbytes), True, [], []
            for buf in recs:
                if deadline is not None and deadline.expired():
                    break
                ms.append(self.time_copy(dev, buf))
                ok &= self.roundtrip_ok(dev, buf, nbytes)
            devp = self.dev_buffer(plateau_bytes)
            for buf in plats:
                if deadline is not None and deadline.expired():
                    break
                plateau_ms.append(self.time_copy(devp, buf))
                ok &= self.roundtrip_ok(devp, buf, plateau_bytes)
            stream = self.stream_gbps(stream_bytes)
        finally:
            link = sampler.stop() if sampler is not None else None
        res = {"ms": ms, "roundtrip_ok": bool(ok), "nbytes": nbytes, "kept": len(recs), "plateau_ms": plateau_ms, "plateau_bytes": plateau_bytes,
               "stream": stream, "pattern": "batch_first"}
        if link is not None:
            res["link"] = link
        return res

    def half_copy_half(self, k_half: int = 12, nbytes: int = RECORD_BYTES, deadline: suite.Deadline | None = None) -> dict:
        """Group m1: k_half buffers, one copy of the first, k_half more, then every buffer timed in allocation order."""
        dev = self.dev_buffer(nbytes)
        bufs = [self.pinned(nbytes, SEED + i) for i in range(k_half)]
        dev.copy_(bufs[0], non_blocking=True)                       # the one copy that runs between the two halves
        self.sync()
        bufs += [self.pinned(nbytes, SEED + k_half + i) for i in range(k_half)]
        ms, ok = [], True
        for buf in bufs:
            if deadline is not None and deadline.expired():
                break
            ms.append(self.time_copy(dev, buf))
            ok &= self.roundtrip_ok(dev, buf, nbytes)
        return {"ms": ms, "roundtrip_ok": ok, "nbytes": nbytes, "kept": len(bufs), "split": k_half}

    def copy_first(self, k: int = K_ROUTE, nbytes: int = RECORD_BYTES, small_bytes: int = MIB, deadline: suite.Deadline | None = None) -> dict:
        """Group m4: one copy of a separate small pinned buffer, then allocate k buffers, then time them."""
        small = self.pinned(small_bytes, SEED + 900)
        dsmall = self.dev_buffer(small_bytes)
        dsmall.copy_(small, non_blocking=True)
        self.sync()
        res = self.context_cell(True, False, False, k=k, nbytes=nbytes, deadline=deadline)
        res["context"]["copy_first_bytes"] = small_bytes
        return res

    def registered_batch_cell(self, kind: str, advice: str, k: int = K_ROUTE, nbytes: int = RECORD_BYTES, reg_flags: int = 1,
                              deadline: suite.Deadline | None = None) -> dict:
        """Groups m6, m7 (kind 'private', the advice) and m8 (kind 'ordinary'): k registered buffers all made before any is timed."""
        dev = self.dev_buffer(nbytes)
        huge = ((nbytes + bp.HUGE - 1) // bp.HUGE) * bp.HUGE
        seed = self.torch.randint(0, 256, (huge,), dtype=self.torch.uint8)
        bufs = []
        for _ in range(k):
            bufs.append(bp.MmapBuffer(self, nbytes, advice, seed, mapping="private", reg_flags=reg_flags) if kind == "private"
                        else OrdinaryBuffer(self, nbytes, seed, reg_flags))
        ms, points, ok, errors = [], [], True, []
        for b in bufs:
            for e in (getattr(b, "advice_error", None), getattr(b, "register_error", None)):
                if e:
                    errors.append(e)
            if b.register_error:
                continue
            if deadline is not None and deadline.expired():
                break
            tensor = b.tensor[:nbytes]
            ms.append(self.time_copy(dev, tensor))
            ok &= self.roundtrip_ok(dev, b.tensor, nbytes)
            points.append({"make": b.smaps_make, "register": b.smaps_register, "copy": b.smaps_entry()})
        registered = bufs[0].registered if bufs else huge
        for b in bufs:
            b.release()
        return {"ms": ms, "smaps": points, "roundtrip_ok": ok, "errors": sorted(set(errors)), "nbytes": nbytes, "registered": registered,
                "mapping": kind, "advice": advice, "populate": False, "reg_flags": reg_flags, "unavailable": None, "pattern": "batch_first"}

    def stream_gbps(self, nbytes: int = STREAM_BYTES, iters: int = 20) -> dict:
        """One device-to-device copy of `nbytes`, p50 of `iters` after 5 warm-up; the rate counts read plus write (2 x bytes / time)."""
        src = self.dev_buffer(nbytes)
        dst = self.dev_buffer(nbytes)
        ms = self.time_copy(dst, src, iters=iters, warm=5)
        return {"ms": ms, "gbps": 2 * nbytes / 1e9 / (ms / 1e3), "nbytes": nbytes}

    def classify_cell(self, k: int = K_CLASSIFY, k_plateau: int = K_PLATEAU, nbytes: int = RECORD_BYTES, plateau_bytes: int = PLATEAU_BYTES,
                      stream_bytes: int = STREAM_BYTES, sampler: LinkSampler | None = None, deadline: suite.Deadline | None = None) -> dict:
        """Groups k0 and k5: the committed canonical copy (record section 3.1) in a fresh process."""
        if sampler is not None:
            sampler.start()
        try:
            main = self.alloc_order(k, nbytes, deadline=deadline)
            dev = self.dev_buffer(plateau_bytes)
            plateau_ms, ok = [], True
            for i in range(k_plateau):
                if deadline is not None and deadline.expired():
                    break
                buf = self.pinned(plateau_bytes, SEED + 700 + i)
                plateau_ms.append(self.time_copy(dev, buf))
                ok &= self.roundtrip_ok(dev, buf, plateau_bytes)
            stream = self.stream_gbps(stream_bytes)
        finally:
            link = sampler.stop() if sampler is not None else None
        res = {"ms": main["ms"], "roundtrip_ok": bool(main["roundtrip_ok"] and ok), "nbytes": nbytes, "kept": main["kept"],
               "plateau_ms": plateau_ms, "plateau_bytes": plateau_bytes, "stream": stream}
        if link is not None:
            res["link"] = link
        return res

    # -- follow-up N (docs/CUDA-PIN-SETTLE-RECORD.md): what happens between a buffer's allocation and its timing
    def one_copy_ms(self, dst, src) -> float:
        """One copy timed alone with events, the GPU synchronised first (a single sample of `time_copy`'s loop)."""
        a, b = self._events()
        self.sync()
        a.record()
        dst.copy_(src, non_blocking=True)
        b.record()
        b.synchronize()
        return a.elapsed_time(b)

    def settle_train(self, k: int = 6, train_ms: float = 1000.0, nbytes: int = RECORD_BYTES, deadline: suite.Deadline | None = None) -> dict:
        """Group n1: each buffer allocated, then copied back to back for train_ms, every copy's ms and host ms since allocation kept."""
        dev = self.dev_buffer(nbytes)
        keep, trains, ok = [], [], True
        for i in range(k):
            if deadline is not None and deadline.expired():
                break
            buf = self.pinned(nbytes, SEED + i)
            t0 = time.perf_counter()
            keep.append(buf)
            ms, at = [], []
            while (time.perf_counter() - t0) * 1e3 < train_ms:
                at.append((time.perf_counter() - t0) * 1e3)
                ms.append(self.one_copy_ms(dev, buf))
            ok &= self.roundtrip_ok(dev, buf, nbytes)
            trains.append({"ms": ms, "t_ms": at})
        return {"trains": trains, "ms": [bp.p50(t["ms"][:25]) for t in trains if t["ms"]], "roundtrip_ok": ok, "nbytes": nbytes,
                "kept": len(keep), "train_ms": train_ms}

    def wait_sweep(self, waits_ms: tuple = WAITS_MS, nbytes: int = RECORD_BYTES, deadline: suite.Deadline | None = None) -> dict:
        """Group n2: each buffer allocated, left idle for T ms (the GPU idle, nothing else running), then timed."""
        dev = self.dev_buffer(nbytes)
        keep, ms, gap, done, ok = [], [], [], [], True
        for i, w in enumerate(waits_ms):
            if deadline is not None and deadline.expired():
                break
            buf = self.pinned(nbytes, SEED + i)
            t0 = time.perf_counter()
            keep.append(buf)
            self.sync()
            time.sleep(w / 1e3)
            gap.append((time.perf_counter() - t0) * 1e3)
            ms.append(self.time_copy(dev, buf))
            ok &= self.roundtrip_ok(dev, buf, nbytes)
            done.append(w)
        return {"ms": ms, "wait_ms": done, "gap_ms": gap, "roundtrip_ok": ok, "nbytes": nbytes, "kept": len(keep)}

    def evict_cell(self, sizes: tuple = EVICT_BYTES, reps: int = 8, nbytes: int = RECORD_BYTES, deadline: suite.Deadline | None = None) -> dict:
        """Group n3: each buffer allocated, E bytes written into one unrelated pageable buffer, then timed at once."""
        dev = self.dev_buffer(nbytes)
        junk = self.torch.empty(max(sizes), dtype=self.torch.uint8)
        keep, ms, ev, fill_ms, ok = [], [], [], [], True
        for i in range(reps * len(sizes)):
            if deadline is not None and deadline.expired():
                break
            e = sizes[i % len(sizes)]
            buf = self.pinned(nbytes, SEED + i)
            keep.append(buf)
            t0 = time.perf_counter()
            if e:
                junk[:e].fill_(i & 0xFF)
            fill_ms.append((time.perf_counter() - t0) * 1e3)
            ms.append(self.time_copy(dev, buf))
            ok &= self.roundtrip_ok(dev, buf, nbytes)
            ev.append(e)
        return {"ms": ms, "evict_bytes": ev, "fill_ms": fill_ms, "roundtrip_ok": ok, "nbytes": nbytes, "kept": len(keep)}

    def retouch_cell(self, k: int = K_ROUTE, age_ms: float = 500.0, nbytes: int = RECORD_BYTES, deadline: suite.Deadline | None = None) -> dict:
        """Group n4: k buffers allocated first and left age_ms, then each untouched, read (sum) or rewritten by the CPU, then timed."""
        dev = self.dev_buffer(nbytes)
        bufs = [self.pinned(nbytes, SEED + i) for i in range(k)]
        self.sync()
        time.sleep(age_ms / 1e3)
        ms, modes, touch_ms, ok = [], [], [], True
        for i, buf in enumerate(bufs):
            if deadline is not None and deadline.expired():
                break
            mode = RETOUCH[i % len(RETOUCH)]
            t0 = time.perf_counter()
            if mode == "read":
                int(buf.sum())
            elif mode == "write":
                g = self.torch.Generator().manual_seed(SEED + 5000 + i)
                buf.copy_(self.torch.randint(0, 256, (nbytes,), dtype=self.torch.uint8, generator=g))
            touch_ms.append((time.perf_counter() - t0) * 1e3)
            ms.append(self.time_copy(dev, buf))
            ok &= self.roundtrip_ok(dev, buf, nbytes)
            modes.append(mode)
        return {"ms": ms, "mode": modes, "touch_ms": touch_ms, "roundtrip_ok": ok, "nbytes": nbytes, "kept": len(bufs), "age_ms": age_ms}

    def alloc_cell(self, k: int = K_ROUTE, nbytes: int = RECORD_BYTES, sampler: LinkSampler | None = None,
                   deadline: suite.Deadline | None = None) -> dict:
        """Group k4: the `cudaHostAlloc` control, optionally beside the link sampler."""
        if sampler is not None:
            sampler.start()
        try:
            res = self.alloc_order(k, nbytes, deadline=deadline)
        finally:
            link = sampler.stop() if sampler is not None else None
        if link is not None:
            res["link"] = link
        return res


# ------------------------------------------------------------------------------------------------ rows

def _row(run_id, arm, field, value, unit, tag, source, **kw):
    return mf.row(run_id, arm, field, value, unit, tag, source=source, **kw)


def rows_link(run_id: str, arm: str, link: dict | None) -> list[dict]:
    if link is None:
        return []
    if link.get("unavailable"):
        return [_row(run_id, arm, "link_gen_min", None, "gen", "unavailable", "nvidia-smi sampler", note=link["unavailable"][:150])]
    rows = [_row(run_id, arm, "link_gen_min", link["min_gen"], "gen", "measured", "nvidia-smi sampler at 10 Hz beside the copies", n=link["samples"]),
            _row(run_id, arm, "link_gen_max", link["max_gen"], "gen", "measured", "nvidia-smi sampler", n=link["samples"])]
    low = sum(v for g, v in link["gens"].items() if int(g) < link["max_gen"])
    rows.append(_row(run_id, arm, "link_samples_below_max_gen", low, "count", "measured", "samples whose generation is below the highest seen",
                     n=link["samples"], note=f"gens {link['gens']} widths {link['widths']}"))
    return rows


def rows_relative(run_id: str, arm: str, ms: list[float], plateau: dict) -> list[dict]:
    """The committed rule's rows for any cell with a plateau (classification cells), next to the absolute 0.28 ms rule."""
    rows = []
    if plateau["instance_class"] == "unavailable":
        return [_row(run_id, arm, "instance_class", None, "text", "unavailable", "committed rule", note="no plateau buffer was timed")]
    absolute = bp.slow_flags(ms, bp.SLOW_MS_9_95MB)
    rows += [_row(run_id, arm, "plateau_gbps", plateau["plateau_gbps"], "GB/s", "measured", "highest p50 rate of the 64 MiB pinned buffers"),
             _row(run_id, arm, "slow_count_relative", plateau["slow_count"], "count", "derived", "rate below 0.75 x plateau (committed rule)", n=len(ms)),
             _row(run_id, arm, "slow_count_absolute", sum(absolute), "count", "derived", "p50 at or above 0.28 ms (the earlier rule)", n=len(ms)),
             _row(run_id, arm, "label_agreement", label_agreement(plateau["slow"], absolute), "fraction", "derived",
                  "buffers given the same label by both rules", n=len(ms)),
             _row(run_id, arm, "instance_class_code", CLASS_CODE[plateau["instance_class"]], "code", "derived",
                  "committed rule: 0 fast, 1 mixed, 2 slow", note=plateau["instance_class"])]
    return rows


def rows_classify(run_id: str, arm: str, res: dict) -> list[dict]:
    rows = bp.rows_alloc_order(run_id, arm, res)
    plateau = classify_buffers(res["ms"], res["plateau_ms"], res["nbytes"], res["plateau_bytes"])
    rows += rows_relative(run_id, arm, res["ms"], plateau)
    s = res["stream"]
    rows.append(_row(run_id, arm, "stream_d2d_gbps", s["gbps"], "GB/s", "measured", "256 MiB device-to-device copy, 2 x bytes / p50 (read plus write)"))
    rows.append(_row(run_id, arm, "stream_d2d_ms", s["ms"], "ms", "measured", "p50 of 20 after 5 warm-up"))
    rows += rows_link(run_id, arm, res.get("link"))
    return rows


def rows_half(run_id: str, arm: str, res: dict) -> list[dict]:
    """Group m1: the slow share of the buffers allocated before the one copy and of those allocated after it."""
    rows = bp.rows_alloc_order(run_id, arm, res)
    ms, split = res["ms"], res["split"]
    for name, part in (("first_half", ms[:split]), ("second_half", ms[split:])):
        if part:
            rows.append(_row(run_id, arm, f"slow_share_{name}", bp.share(bp.slow_flags(part, bp.SLOW_MS_9_95MB)), "fraction", "measured",
                             f"slow share of the {name} (allocated {'before' if name == 'first_half' else 'after'} the one copy)", n=len(part)))
    return rows


def rows_route(run_id: str, arm: str, res: dict) -> list[dict]:
    """Rows of a DC-3 route cell. Registered routes carry smaps points (rows of B2's private cell); the control has none."""
    rows = bp.rows_private_cell(run_id, arm, res) if "smaps" in res else bp.rows_alloc_order(run_id, arm, res)
    if res["ms"]:
        rows.append(_row(run_id, arm, "h2d_gbps_median", bp.gbps(res["nbytes"], bp.p50(res["ms"])), "GB/s", "measured", "median of the buffers' p50 rates", n=len(res["ms"])))
    rows += rows_link(run_id, arm, res.get("link"))
    return rows


def _slow_share_rows(run_id: str, arm: str, ms: list[float], keys: list, name: str, unit_note: str) -> list[dict]:
    rows = []
    for key in dict.fromkeys(keys):
        part = [x for x, k in zip(ms, keys, strict=True) if k == key]
        rows.append(_row(run_id, arm, f"slow_share_{name}_{key}", bp.share(bp.slow_flags(part, bp.SLOW_MS_9_95MB)), "fraction", "measured",
                         f"slow share of the buffers with {name} {key}{unit_note}", n=len(part)))
        rows.append(_row(run_id, arm, f"median_ms_{name}_{key}", bp.p50(part), "ms", "measured", "median of the buffers' p50", n=len(part)))
    return rows


def train_summary(train: dict, run: int = 10, last: int = 100) -> dict:
    """n1: whether the last `last` copies are slow, and the settle time: host ms of the first copy of `run` fast copies in a row."""
    ms, at = train["ms"], train["t_ms"]
    fast = [x <= 0.25 for x in ms]
    settle = None
    for i in range(len(ms) - run + 1):
        if all(fast[i:i + run]):
            settle = at[i]
            break
    tail = ms[-last:]
    return {"copies": len(ms), "first_ms": ms[0] if ms else None, "settle_ms": settle,
            "tail_slow_share": bp.share(bp.slow_flags(tail, bp.SLOW_MS_9_95MB)) if tail else None}


def rows_train(run_id: str, arm: str, res: dict) -> list[dict]:
    rows = bp.rows_alloc_order(run_id, arm, res)
    for i, t in enumerate(res["trains"]):
        s = train_summary(t)
        rows += [_row(run_id, arm, f"train_copies_b{i}", s["copies"], "count", "measured", "copies in the train"),
                 _row(run_id, arm, f"train_first_ms_b{i}", s["first_ms"], "ms", "measured", "first copy after allocation"),
                 _row(run_id, arm, f"train_tail_slow_share_b{i}", s["tail_slow_share"], "fraction", "measured", "slow share of the last 100 copies"),
                 _row(run_id, arm, f"train_settle_ms_b{i}", s["settle_ms"], "ms", "measured" if s["settle_ms"] is not None else "unavailable",
                      "host ms since allocation of the first of 10 fast copies in a row",
                      note=None if s["settle_ms"] is not None else "no 10 fast copies in a row within the train")]
    return rows


def rows_wait(run_id: str, arm: str, res: dict) -> list[dict]:
    return bp.rows_alloc_order(run_id, arm, res) + _slow_share_rows(run_id, arm, res["ms"], res["wait_ms"], "wait", " ms")


def rows_evict(run_id: str, arm: str, res: dict) -> list[dict]:
    keys = [e // 1024 for e in res["evict_bytes"]]
    rows = bp.rows_alloc_order(run_id, arm, res) + _slow_share_rows(run_id, arm, res["ms"], keys, "evict", " KiB")
    for key in dict.fromkeys(keys):
        part = [f for f, k in zip(res["fill_ms"], keys, strict=True) if k == key]
        rows.append(_row(run_id, arm, f"fill_ms_evict_{key}", bp.p50(part), "ms", "measured", "host ms of the unrelated write", n=len(part)))
    return rows


def rows_retouch(run_id: str, arm: str, res: dict) -> list[dict]:
    return bp.rows_alloc_order(run_id, arm, res) + _slow_share_rows(run_id, arm, res["ms"], res["mode"], "touch", "")


# ------------------------------------------------------------------------------------------------ CLI

def cmd_plan(args) -> int:
    for g, what in GROUPS.items():
        print(f"{g:3s} {what}")
    print(f"follow-up order {list(FOLLOWUP_ORDER)}")
    print(f"order {list(ORDER)}; record bytes {RECORD_BYTES}, plateau {PLATEAU_BYTES}, stream {STREAM_BYTES}; slow below {SLOW_FRACTION} x plateau")
    return 0


def run_group(bench: PinRouteBench, group: str, run_id: str, dl: suite.Deadline, gates: dict, failures: list, raw: dict,
              small: bool = False, sampler_factory=LinkSampler) -> list[dict]:
    cfg = SMALL if small else {"k_classify": K_CLASSIFY, "k_plateau": K_PLATEAU, "k_route": K_ROUTE, "nbytes": RECORD_BYTES,
                               "plateau_bytes": PLATEAU_BYTES, "stream_bytes": STREAM_BYTES, "arena_bytes": ARENA_BYTES}
    rows: list[dict] = []

    def guarded(name, fn, row_fn):
        try:
            res = fn()
        except Exception as exc:
            failures.append({"cell": name, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            return
        raw[name] = res
        if not res.get("roundtrip_ok", True):
            gates["roundtrip"] = False
        rows.extend(row_fn(run_id, f"pinroute.{name}", res))

    nb = cfg["nbytes"]
    if group in ("k0", "k5"):
        guarded("classify_start" if group == "k0" else "classify_end",
                lambda: bench.classify_cell(cfg["k_classify"], cfg["k_plateau"], nb, cfg["plateau_bytes"], cfg["stream_bytes"],
                                            sampler=sampler_factory(), deadline=dl), rows_classify)
    elif group in ("k1", "k2"):
        flags = 1 if group == "k1" else 0
        guarded("register_portable" if group == "k1" else "register_flag0",
                lambda: bench.private_cell("default", "private", False, cfg["k_route"], nb, deadline=dl, reg_flags=flags), rows_route)
    elif group == "k3":
        guarded("ordinary_flag0", lambda: bench.ordinary_cell(0, cfg["k_route"], nb, deadline=dl), rows_route)
    elif group == "k4":
        guarded("alloc_control", lambda: bench.alloc_cell(cfg["k_route"], nb, sampler=sampler_factory(), deadline=dl), rows_route)
    elif group in ("m0", "m9"):
        guarded("classify_batch_start" if group == "m0" else "classify_batch_end",
                lambda: bench.classify_batch_cell(cfg["k_classify"], cfg["k_plateau"], nb, cfg["plateau_bytes"], cfg["stream_bytes"],
                                                  sampler=sampler_factory(), deadline=dl), rows_classify)
    elif group == "m1":
        guarded("half_copy_half", lambda: bench.half_copy_half(cfg["k_route"] // 2, nb, deadline=dl), rows_half)
    elif group == "m4":
        guarded("copy_first", lambda: bench.copy_first(cfg["k_route"], nb, small_bytes=min(MIB, nb), deadline=dl), rows_route)
    elif group in ("m6", "m7"):
        guarded("private_thp_batch" if group == "m6" else "private_nothp_batch",
                lambda: bench.registered_batch_cell("private", "thp" if group == "m6" else "nothp", cfg["k_route"], nb, 1, deadline=dl), rows_route)
    elif group == "m8":
        guarded("ordinary_batch", lambda: bench.registered_batch_cell("ordinary", "none", cfg["k_route"], nb, 0, deadline=dl), rows_route)
    elif group == "n1":
        guarded("settle_train", lambda: bench.settle_train(cfg.get("k_train", 6), cfg.get("train_ms", 1000.0), nb, deadline=dl), rows_train)
    elif group == "n2":
        guarded("wait_sweep", lambda: bench.wait_sweep(cfg.get("waits_ms", WAITS_MS), nb, deadline=dl), rows_wait)
    elif group == "n3":
        guarded("evict", lambda: bench.evict_cell(cfg.get("evict_bytes", EVICT_BYTES), cfg.get("evict_reps", 8), nb, deadline=dl), rows_evict)
    elif group == "n4":
        guarded("retouch", lambda: bench.retouch_cell(cfg["k_route"], cfg.get("age_ms", 500.0), nb, deadline=dl), rows_retouch)
    elif group in CONTEXT:
        bf, ar, st = CONTEXT[group]
        guarded({"l1": "batch_first", "l2": "arena_dest", "l3": "side_stream", "l4": "arena_full"}[group],
                lambda: bench.context_cell(bf, ar, st, cfg["k_route"], nb, cfg.get("arena_bytes", ARENA_BYTES), deadline=dl), rows_route)
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
    bench = PinRouteBench(0)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{stamp}-pinroute-{args.cells}-{report['manifest_id'][:6]}"
    out = Path(args.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    mf.write_json(out / "machine-manifest.json", report)
    dl = suite.Deadline(args.max_seconds)
    gates = {"roundtrip": True, "sanity": True}
    failures: list[dict] = []
    raw: dict = {"host_thp": bp.host_thp()}
    rows = run_group(bench, args.cells, run_id, dl, gates, failures, raw)
    for r in rows:
        if "gbps" in r["field"] and r.get("value") is not None and r["field"] != "stream_d2d_gbps" and r["value"] > suite.PCIE_SANITY_GB_S:
            gates["sanity"] = False
    status = "truncated" if dl.expired() else "complete"
    manifest = {
        "schema": mf.SCHEMA_NAME, "schema_version": mf.SCHEMA_VERSION, "platform": mf.PLATFORM, "kind": "run",
        "created_utc": mf.utc_now(), "machine_manifest_id": report["manifest_id"], "run_id": run_id,
        "instrument": {"name": "benchmarks/cuda/pin_route.py", "version": "K-1"}, "code": report["code"], "mode": "timing",
        "workload": {"id": f"K-{args.cells}", "class": "pinned host buffer H2D rate by pinning route and the instance's copy class (no model)",
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
