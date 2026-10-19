"""
CUDA lane N3 stage A harness (docs/CUDA-N3-EXPERIMENT-RECORD.md): layer-ahead prefetch of recorded predicted experts against
demand fetch on the N2 stage 1 slot arena, with a bandwidth-bound stand-in compute, real pinned host-to-device copies on a copy
stream, event dependencies, a per-layer host synchronization and a per-token critical-path reconstruction. Standalone: it imports
only `replay_arena`, `manifest`, `suite` (all in `benchmarks/cuda/`), numpy and torch.

Layering, so the Mac can test everything except the GPU calls:

* `ArenaPF` / `PfBook`: the bookkeeping (LRU arena slots plus prefetch insertion, pending and covered accounting), written from
  the record's section 3 and **independent of `reference_pass_pf`** (an OrderedDict model) and of
  `prefetch_coverage.coverage`; their event hashes and counts are compared (gate (a)). `expect` (Mac) writes the reference hashes
  next to the trace's sha256; the host refuses to time a cell whose own hash differs.
* `StandIn`: the stand-in compute (a reduction over W bytes of resident weights, two kernels per layer split 40 / 60 so the late
  prefetch arm has a window to wait on), calibrated so the pair alone takes C ms (gate (c)).
* `OverlapArena`: `TorchArena` plus the compute and copy streams and the event pool; `rehearsal_cpu=True` runs the same code on
  the CPU with an event shim (never on the command line: its numbers are never a CUDA measurement).
* `run_cell`: the timed replay of one arm. The bookkeeping order is fixed by the record: for every decode row the predicted set
  is inserted, then the row's real experts are requested; the prefetch for row r is issued at the boundary before row r-1's
  compute (arm `P0`: queued behind that boundary's demand copies; arm `P40`: after 40 % of row r-1's compute, through an event
  the copy stream waits on).

Claims overlap mechanics only: exposed copy time against compute window, copy-lane saturation and interference, for this card and
this trace. The compute is a stand-in, not the model.

Usage (Mac, no GPU):
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/overlap_arena.py expect \
        --trace benchmarks/results/routing-trace-deepseek-long/trace.npz --out benchmarks/cuda/n3-expected.json
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/overlap_arena.py plan
GPU host: `run --trace T --expected benchmarks/cuda/n3-expected.json --cells smoke|full [--execute]`.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import manifest as mf  # noqa: E402
import replay_arena as ra  # noqa: E402
import suite  # noqa: E402

ARMS = ("N", "D", "P0", "P40", "O")
ARM_MODE = {"D": "none", "P0": "pred", "P40": "pred", "O": "oracle"}      # the bookkeeping each arm uses
SPLIT = 0.4                                                               # first kernel's share of a layer's compute
DECODE_TOKENS = 400
CALIBRATION_TOLERANCE = 0.05
DRIFT_LIMIT = 0.03
CLOSURE_TOLERANCE = 0.05
CLOSURE_MIN_SHARE = 0.9                                                   # N3-P10: at least 90 % of tokens close
CLOSURE_REJECT_SHARE = 0.8                                                # below this the cell's decomposition is rejected
WEIGHT_CAP_BYTES = 6 * 2**30
HARD_CELL_SECONDS = 900


# ------------------------------------------------------------------------------------------------ bookkeeping

class ArenaPF(ra.ArenaLRU):
    """`ArenaLRU` plus prefetch insertion: a predicted key not resident is loaded as most recently used (evicting from the LRU end
    like a demand miss); a resident one is left alone (no recency refresh)."""

    def prefetch(self, key: tuple[int, int]):
        """Returns (loaded, slot, ops)."""
        lru = self.lru
        if key in lru:
            return False, lru[key], ()
        ops = []
        while len(lru) >= self.slots:
            victim, _ = self._evict()
            ops.append((0, victim[0], victim[1]))
        slot = self.free.pop()
        lru[key] = slot
        ops.append((1, key[0], key[1]))
        return True, slot, tuple(ops)


class PfBook:
    """The record's bookkeeping with counters. `mode` is 'none' (demand only), 'pred' (recorded predicted sets) or 'oracle' (the
    next row's real experts). Per row: `prefetch` (the row's predicted set), then `demand` (its real experts), then `end_row`."""

    def __init__(self, slots: int, n_layers: int, mode: str):
        if mode not in ("none", "pred", "oracle"):
            raise ValueError(f"unknown mode {mode!r}")
        self.arena, self.mode = ArenaPF(slots, n_layers), mode
        self.pending: set[tuple[int, int]] = set()
        self.hash = hashlib.sha256()
        self.chain = 0
        self.demand_misses = self.loads = self.covered = 0
        self.token_counts: dict[int, dict[str, int]] = {}

    def _tok(self, tok: int) -> dict[str, int]:
        return self.token_counts.setdefault(tok, {"demand": 0, "loads": 0, "covered": 0, "requests": 0})

    def prefill(self, layer: int, ordered_keys):
        hits, miss, ops, sets = self.arena.prefill_layer(layer, ordered_keys)
        ra._digest_feed(self.hash, ("P", layer, hits, miss, ops))
        return sets

    def prefetch(self, tok: int, layer: int, experts) -> list[tuple[int, tuple[int, int]]]:
        """Insert the keys (layer, e) for e in `experts`; returns [(slot, key)] of the loads to copy."""
        out = []
        for e in experts:
            key = (layer, int(e))
            loaded, slot, ops = self.arena.prefetch(key)
            ra._digest_feed(self.hash, ("F", tok, layer, key[1], loaded, ops))
            if loaded:
                out.append((slot, key))
                self.pending.add(key)
                self.loads += 1
                self._tok(tok)["loads"] += 1
        return out

    def demand(self, tok: int, layer: int, experts) -> list[tuple[int, tuple[int, int]]]:
        out = []
        t = self._tok(tok)
        for e in experts:
            key = (layer, int(e))
            was_pending = key in self.pending
            hit, slot, ops = self.arena.decode_request(key)
            t["requests"] += 1
            covered = bool(hit and was_pending)
            ra._digest_feed(self.hash, ("D", tok, layer, key[1], hit, covered, ops))
            if covered:
                self.covered += 1
                t["covered"] += 1
                self.pending.discard(key)
            if not hit:
                out.append((slot, key))
                self.demand_misses += 1
                t["demand"] += 1
                self.chain = (self.chain * 1000003 + ((tok * 64 + layer) * 1024 + key[1]) * 8192 + slot) & ((1 << 61) - 1)
        return out

    def end_row(self, layer: int) -> None:
        self.pending = {k for k in self.pending if k[0] != layer}

    def summary(self) -> dict:
        n = max(1, len(self.token_counts))
        return {"hash": self.hash.hexdigest(), "chain": self.chain, "tokens": len(self.token_counts),
                "demand_misses": self.demand_misses, "prefetch_loads": self.loads, "covered": self.covered,
                "wasted": self.loads - self.covered, "demand_per_token": self.demand_misses / n,
                "loads_per_token": self.loads / n, "covered_per_token": self.covered / n,
                "wasted_per_token": (self.loads - self.covered) / n,
                "token_demand": [self.token_counts[t]["demand"] for t in sorted(self.token_counts)]}


def predicted_lookup(arrays) -> dict[tuple[int, int], list[int]]:
    if "pred_target" not in arrays:
        return {}
    return {(int(p), int(t)): e.tolist() for t, p, e in zip(arrays["pred_target"], arrays["pred_position"], arrays["pred_experts"], strict=True)}


def decode_plan(arrays, max_decode_tokens: int | None):
    """The walk with one entry per item: ('P', layer, ordered_keys) or ('D', token, layer, experts, position)."""
    pos_rows = np.nonzero(arrays["phase"] == ra.PHASE_DECODE)[0]
    positions = [int(arrays["position"][i]) for i in pos_rows]
    out, di = [], 0
    for item in ra.walk(arrays, max_decode_tokens):
        if item[0] == "P":
            out.append(item)
        else:
            out.append(("D", item[1], item[2], item[3], positions[di]))
            di += 1
    return out


def row_prefetch_set(mode: str, pred, plan, i: int):
    """The experts the arm prefetches for decode item `i` of the plan (None when it prefetches nothing for that row)."""
    if mode == "none":
        return None
    item = plan[i]
    if mode == "oracle":
        return item[3]
    return pred.get((item[4], item[2]))


def bookkeeping_pass_pf(arrays, slots: int, n_layers: int, mode: str, max_decode_tokens: int | None) -> dict:
    book = PfBook(slots, n_layers, mode)
    pred = predicted_lookup(arrays) if mode == "pred" else {}
    plan = decode_plan(arrays, max_decode_tokens)
    for i, item in enumerate(plan):
        if item[0] == "P":
            book.prefill(item[1], item[2])
            continue
        _, tok, layer, experts, _pos = item
        ahead = row_prefetch_set(mode, pred, plan, i)
        if ahead:
            book.prefetch(tok, layer, ahead)
        book.demand(tok, layer, experts)
        book.end_row(layer)
    book.arena.check_invariants()
    return book.summary()


def reference_pass_pf(arrays, slots: int, n_layers: int, mode: str, max_decode_tokens: int | None) -> dict:
    """The same record semantics as a plain OrderedDict model (no `ArenaLRU`, no `PfBook`), logged in the same event format.
    Prefill uses `ArenaLRU.prefill_layer` (already compared with the simulator's `Store` in N2 stage 1)."""
    pre = ra.ArenaLRU(slots, n_layers)
    lru: OrderedDict[tuple[int, int], int] = OrderedDict()
    free = list(range(slots - 1, -1, -1))
    h = hashlib.sha256()
    pending: set[tuple[int, int]] = set()
    pred = predicted_lookup(arrays) if mode == "pred" else {}
    plan = decode_plan(arrays, max_decode_tokens)
    counts = {"demand": 0, "loads": 0, "covered": 0}
    toks: set[int] = set()

    def load(key):
        ops = []
        while len(lru) >= slots:
            victim = next(iter(lru))
            free.append(lru.pop(victim))
            ops.append((0, victim[0], victim[1]))
        lru[key] = free.pop()
        ops.append((1, key[0], key[1]))
        return tuple(ops)

    synced = False
    for i, item in enumerate(plan):
        if item[0] == "P":
            if synced:                                  # a prefill after decode rows continues from the decode state
                pre.lru, pre.free = OrderedDict(lru), list(free)
            hits, miss, ops, _sets = pre.prefill_layer(item[1], item[2])
            ra._digest_feed(h, ("P", item[1], hits, miss, ops))
            synced = False
            continue
        if not synced:                                  # the decode phase continues from the arena the prefill left
            lru, free = OrderedDict(pre.lru), list(pre.free)
            synced = True
        _, tok, layer, experts, _pos = item
        toks.add(tok)
        ahead = row_prefetch_set(mode, pred, plan, i)
        for e in ahead or []:
            key = (layer, int(e))
            loaded = key not in lru
            ops = load(key) if loaded else ()
            ra._digest_feed(h, ("F", tok, layer, key[1], loaded, ops))
            if loaded:
                pending.add(key)
                counts["loads"] += 1
        for e in experts:
            key = (layer, int(e))
            hit = key in lru
            covered = hit and key in pending
            if hit:
                lru.move_to_end(key)
                ops = ()
            else:
                ops = load(key)
                counts["demand"] += 1
            ra._digest_feed(h, ("D", tok, layer, key[1], hit, covered, ops))
            if covered:
                counts["covered"] += 1
                pending.discard(key)
        pending = {k for k in pending if k[0] != layer}
    n = max(1, len(toks))
    return {"hash": h.hexdigest(), "tokens": len(toks), "demand_misses": counts["demand"], "prefetch_loads": counts["loads"],
            "covered": counts["covered"], "wasted": counts["loads"] - counts["covered"], "demand_per_token": counts["demand"] / n,
            "loads_per_token": counts["loads"] / n}


# ------------------------------------------------------------------------------------------------ cells

def cell_defs(profile: str) -> list[dict]:
    """The registered cells (record section 3). `smoke` is a short GPU smoke of the stream code; `rehearsal` is for the Mac tests."""
    if profile == "smoke":
        return [{"id": "p17", "percent": 17, "tokens": 40, "c_values": (1.0,), "arms": ("N", "D", "P0", "O", "N")}]
    if profile == "full":
        return [{"id": "p17", "percent": 17, "tokens": DECODE_TOKENS, "c_values": (0.3, 1.0, 3.0),
                 "arms": ("N", "D", "P0", "P40", "O", "N")},
                {"id": "p8", "percent": 8, "tokens": DECODE_TOKENS, "c_values": (1.0,), "arms": ("N", "D", "P0", "N")}]
    if profile == "rehearsal":
        return [{"id": "p17", "percent": 17, "slots": 40, "tokens": 6, "c_values": (0.05,), "arms": ("N", "D", "P0", "P40", "O", "N")}]
    raise ValueError(f"unknown profile {profile!r}")


def cells_of(group: dict) -> list[dict]:
    """One entry per (C, arm) in run order; the first and last `N` of a C are its start and end controls."""
    out = []
    for c in group["c_values"]:
        for pos, arm in enumerate(group["arms"]):
            role = "N0" if (arm == "N" and pos == 0) else ("N1" if arm == "N" else arm)
            out.append({"id": f"{group['id']}.C{c}.{role}", "group": group["id"], "percent": group["percent"], "tokens": group["tokens"],
                        "c": c, "arm": arm, "role": role})
    return out


# ------------------------------------------------------------------------------------------------ the stand-in compute

class _CpuEvent:
    def record(self, stream=None):
        self.t = time.perf_counter()

    def synchronize(self):
        pass

    def elapsed_time(self, other) -> float:
        return (other.t - self.t) * 1e3


class StandIn:
    """Two reduction kernels per layer over resident weights (first `SPLIT` of the elements, then the rest)."""

    def __init__(self, torch, device, rehearsal: bool):
        self.t, self.device, self.rehearsal = torch, device, rehearsal
        self.w = None
        self.n = 0
        self.cut = 0
        self.events = (_CpuEvent(), _CpuEvent()) if rehearsal else (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))

    def set_bytes(self, nbytes: int) -> None:
        self.w = None
        self.n = max(2, int(nbytes) // 2)
        self.w = self.t.ones(self.n, dtype=self.t.float16, device=self.device)
        self.cut = max(1, int(self.n * SPLIT))

    def part1(self):
        return self.w[: self.cut].sum()

    def part2(self):
        return self.w[self.cut:].sum()

    def nbytes(self) -> int:
        return self.n * 2

    def measure(self, iters: int = 30, warm: int = 5) -> float:
        out = []
        a, b = self.events
        for i in range(iters + warm):
            if not self.rehearsal:
                self.t.cuda.synchronize()
            a.record()
            self.part1()
            self.part2()
            b.record()
            b.synchronize()
            if i >= warm:
                out.append(a.elapsed_time(b))
        return sorted(out)[len(out) // 2]

    def calibrate(self, target_ms: float, start_bytes: int, cap_bytes: int = WEIGHT_CAP_BYTES, tolerance: float = CALIBRATION_TOLERANCE,
                  rounds: int = 5) -> dict:
        nbytes = start_bytes
        ms = None
        for _ in range(rounds):
            self.set_bytes(nbytes)
            ms = self.measure()
            if abs(ms / target_ms - 1.0) <= tolerance:
                break
            nbytes = min(cap_bytes, max(4, int(nbytes * target_ms / max(ms, 1e-9))))
        return calibration_result(target_ms, ms, self.nbytes(), tolerance, self.nbytes() >= cap_bytes)


def calibration_result(target_ms: float, measured_ms: float, nbytes: int, tolerance: float, capped: bool = False) -> dict:
    """Gate (c): the stand-in pair alone must take C within `tolerance`."""
    err = measured_ms / target_ms - 1.0
    return {"target_ms": target_ms, "measured_ms": measured_ms, "bytes": nbytes, "error": err, "tolerance": tolerance,
            "ok": abs(err) <= tolerance, "capped": capped}


# ------------------------------------------------------------------------------------------------ the arena with streams

class OverlapArena(ra.TorchArena):
    """`TorchArena` plus a compute stream (the current one) and the per-layer event pool. Copies go on `self.stream`."""

    EVENT_KINDS = ("pre", "start", "mid", "end", "dem_a", "dem_b", "pf_a", "pf_b")

    def __init__(self, slots: int, nbytes: int, seed: int = 20261018, device_index: int = 0, rehearsal_cpu: bool = False,
                 n_layers: int = 40):
        super().__init__(slots, nbytes, seed, device_index, rehearsal_cpu)
        torch = self.t
        self.n_layers = n_layers
        self.compute_stream = None if rehearsal_cpu else torch.cuda.current_stream(self.dev_name)
        mk = (lambda: _CpuEvent()) if rehearsal_cpu else (lambda: torch.cuda.Event(enable_timing=True))
        self.pool = [{k: [mk() for _ in range(n_layers)] for k in self.EVENT_KINDS} for _ in range(2)]
        self.tok_start = [mk(), mk()]
        self.standin = StandIn(torch, "cpu" if rehearsal_cpu else self.dev_name, rehearsal_cpu)
        self.issued_copies = 0

    # events and streams
    def rec_compute(self, ev) -> None:
        if self.rehearsal:
            ev.record()
        else:
            ev.record(self.compute_stream)

    def rec_copy(self, ev) -> None:
        if self.rehearsal:
            ev.record()
        else:
            ev.record(self.stream)

    def compute_wait(self, ev) -> None:
        if not self.rehearsal:
            self.compute_stream.wait_event(ev)

    def copy_wait(self, ev) -> None:
        if not self.rehearsal and self.stream is not None:
            self.stream.wait_event(ev)

    def enqueue(self, loads: list[tuple[int, tuple[int, int]]]) -> None:
        """Queue real copies on the copy stream; no event, no wait."""
        ctx = contextlib.nullcontext() if self.stream is None else self.t.cuda.stream(self.stream)
        with ctx:
            for slot, key in loads:
                self._slot(slot).copy_(self.sources[ra.source_index(key)], non_blocking=not self.rehearsal)
        self.copies += len(loads)
        self.issued_copies += len(loads)

    def compute_part(self, which: int) -> None:
        (self.standin.part1 if which == 1 else self.standin.part2)()

    def sync_compute(self, ev) -> None:
        ev.synchronize()

    def elapsed(self, a, b) -> float:
        return a.elapsed_time(b)


# ------------------------------------------------------------------------------------------------ the timed replay

def _span(arena, a, b) -> float:
    return arena.elapsed(a, b)


def token_record(arena, parity: int, n: int, had_demand: list[bool], had_pf: list[bool]) -> dict:
    """Read one token's events (all complete): critical-path segments and lane times, ms."""
    ev = arena.pool[parity]
    ts = arena.tok_start[parity]
    busy = stall = gap = dem = pf = 0.0
    prev_end = ts
    for i in range(n):
        gap += _span(arena, prev_end, ev["pre"][i])
        stall += _span(arena, ev["pre"][i], ev["start"][i])
        busy += _span(arena, ev["start"][i], ev["end"][i])
        prev_end = ev["end"][i]
        if had_demand[i]:
            dem += _span(arena, ev["dem_a"][i], ev["dem_b"][i])
        if had_pf[i]:
            pf += _span(arena, ev["pf_a"][i], ev["pf_b"][i])
    return {"busy_ms": busy, "stall_ms": stall, "gap_ms": gap, "span_ms": busy + stall + gap, "dem_busy_ms": dem, "pf_busy_ms": pf}


def run_cell(arrays, slots: int, n_layers: int, arena: OverlapArena, arm: str, *, max_decode_tokens: int | None = None,
             readback_every: int = ra.READBACK_EVERY, deadline: suite.Deadline | None = None, seed: int = 0) -> dict:
    """The timed replay of one arm. `N`: stand-in plus per-layer synchronization only. Otherwise the bookkeeping and copies of the
    arm (record section 3). Prefill admissions populate the arena untimed. Token 0 is a warm-up and is excluded from the
    timing statistics (not from the counts)."""
    mode = ARM_MODE.get(arm)
    pred = predicted_lookup(arrays) if mode == "pred" else {}
    plan = decode_plan(arrays, max_decode_tokens)
    book = PfBook(slots, n_layers, mode or "none") if arm != "N" else None
    tokens: list[dict] = []
    readbacks = readback_fail = 0
    next_check = readback_every
    truncated = False
    # group the decode items by token, keeping prefill items as separators
    cur_tok = None
    pending_records: tuple | None = None                  # (parity, n, had_demand, had_pf, wall_ms, host_ms, tok) of the previous token
    parity = 0

    def flush(rec) -> None:
        par, n, hd, hp, wall, host, tok = rec
        r = token_record(arena, par, n, hd, hp)
        r.update(wall_ms=wall, host_ms=host, tok=tok)
        tokens.append(r)

    def prefetch_for(idx: int) -> list:
        """Bookkeeping of the prefetch for plan item `idx` (a D row); loads to copy."""
        ahead = row_prefetch_set(mode, pred, plan, idx) if mode else None
        if not ahead:
            return []
        it = plan[idx]
        return book.prefetch(it[1], it[2], ahead)

    items = list(range(len(plan)))
    k = 0
    started_pf = set()                                    # D items whose prefetch was already done at the previous boundary
    while k < len(items):
        item = plan[k]
        if item[0] == "P":
            if book is not None:
                sets = book.prefill(item[1], item[2])
                arena.populate(sets)
            k += 1
            continue
        tok = item[1]
        if deadline is not None and tok != cur_tok and deadline.expired():
            truncated = True
            break
        # one token: consecutive D items with this token number
        j = k
        while j < len(plan) and plan[j][0] == "D" and plan[j][1] == tok:
            j += 1
        rows = list(range(k, j))
        n = len(rows)
        if n > arena.n_layers:
            raise ValueError("a token has more layers than the event pool")
        ev = arena.pool[parity]
        had_demand, had_pf = [False] * n, [False] * n
        host_s = 0.0
        # no device synchronize here: a prefetch still in flight from the previous token's last boundary (arm `O` only) belongs to this
        # token's first layer and must be waited for by its compute, not hidden outside the timed span
        w0 = time.perf_counter()
        arena.rec_compute(arena.tok_start[parity])
        for pos, idx in enumerate(rows):
            h0 = time.perf_counter()
            _, _t, layer, experts, _p = plan[idx]
            demand_loads: list = []
            pf_loads: list = []
            if book is not None:
                if idx not in started_pf:                 # the first row after a prefill (or of the run) has no earlier boundary
                    demand_loads = prefetch_for(idx)      # needed by this row now: the row's compute waits for them like a miss
                demand_loads = demand_loads + book.demand(tok, layer, experts)
                book.end_row(layer)
                nxt = idx + 1
                if nxt < len(plan) and plan[nxt][0] == "D":
                    pf_loads = prefetch_for(nxt)
                    started_pf.add(nxt)
                # the batch event is always recorded and waited for: it completes after every copy queued before it, so a row whose
                # keys were all prefetched still waits for them (a stall) and never runs ahead of their arrival
                arena.rec_copy(ev["dem_a"][pos])
                arena.enqueue(demand_loads)
                arena.rec_copy(ev["dem_b"][pos])
                had_demand[pos] = True
            if pf_loads and arm in ("P0", "O"):
                arena.rec_copy(ev["pf_a"][pos])
                arena.enqueue(pf_loads)
                arena.rec_copy(ev["pf_b"][pos])
                had_pf[pos] = True
            arena.rec_compute(ev["pre"][pos])
            if had_demand[pos]:
                arena.compute_wait(ev["dem_b"][pos])
            arena.rec_compute(ev["start"][pos])
            arena.compute_part(1)
            arena.rec_compute(ev["mid"][pos])
            arena.compute_part(2)
            arena.rec_compute(ev["end"][pos])
            if pf_loads and arm == "P40":
                arena.copy_wait(ev["mid"][pos])           # the event is recorded already: the copy stream waits for it
                arena.rec_copy(ev["pf_a"][pos])
                arena.enqueue(pf_loads)
                arena.rec_copy(ev["pf_b"][pos])
                had_pf[pos] = True
            host_s += time.perf_counter() - h0
            arena.sync_compute(ev["end"][pos])            # the router synchronization
        wall = (time.perf_counter() - w0) * 1e3
        rec = (parity, n, had_demand, had_pf, wall, host_s * 1e3, tok)
        if pending_records is not None:
            flush(pending_records)
        pending_records = rec
        parity ^= 1
        cur_tok = tok
        # gate (b): outside the timed span, between tokens
        if book is not None and arena.copies >= next_check:
            next_check += readback_every
            key, slot = next(reversed(book.arena.lru.items()))
            readbacks += 1
            if not arena.verify(slot, key):
                readback_fail += 1
        k = j
    arena.sync()
    if pending_records is not None:
        flush(pending_records)
    if book is not None:
        rng = np.random.default_rng(seed)
        resident = list(book.arena.lru.items())
        sample = [resident[i] for i in rng.choice(len(resident), size=min(48, len(resident)), replace=False)] if resident else []
        for key, slot in sample:
            readbacks += 1
            if not arena.verify(slot, key):
                readback_fail += 1
        book.arena.check_invariants()
    return {"arm": arm, "tokens": tokens, "truncated": truncated, "readbacks": readbacks, "readback_failures": readback_fail,
            "book": book.summary() if book is not None else None, "memory": arena.memory(), "nbytes": arena.nbytes,
            "issued_copies": arena.issued_copies}


# ------------------------------------------------------------------------------------------------ analysis

def closure(tokens: list[dict]) -> dict:
    """N3-P10: per token, |wall - GPU span| / wall where span = lead + stalls + busy; the share of tokens within tolerance."""
    use = tokens[1:] if len(tokens) > 1 else tokens
    errs = [abs(t["wall_ms"] - t["span_ms"]) / t["wall_ms"] for t in use if t["wall_ms"] > 0]
    share = (sum(1 for e in errs if e <= CLOSURE_TOLERANCE) / len(errs)) if errs else 0.0
    return {"share": share, "median_error": sorted(errs)[len(errs) // 2] if errs else None, "n": len(errs),
            "ok": share >= CLOSURE_REJECT_SHARE, "meets_prediction": share >= CLOSURE_MIN_SHARE}


def p50(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2]


def exposed_and_hidden(walls: dict[str, float]) -> dict:
    """`walls` maps role to p50 token wall ms ('N0', 'N1', 'D', 'P0', 'P40', 'O'). E_X = wall_X - wall_N, H_X = 1 - E_X / E_D."""
    n_walls = [walls[r] for r in ("N0", "N1") if r in walls]
    if not n_walls:
        return {}
    floor = sum(n_walls) / len(n_walls)
    out = {"floor_ms": floor}
    if len(n_walls) == 2:
        out["drift"] = walls["N1"] / walls["N0"] - 1.0
    for arm in ("D", "P0", "P40", "O"):
        if arm in walls:
            out[f"E_{arm}"] = walls[arm] - floor
    if "E_D" in out and out["E_D"] > 0:
        for arm in ("P0", "P40", "O"):
            if f"E_{arm}" in out:
                out[f"H_{arm}"] = 1.0 - out[f"E_{arm}"] / out["E_D"]
    return out


def cell_rows(run_id: str, cell: dict, res: dict, hash_ok: bool | None, calib: dict, source_ms: list[float] | None) -> list[dict]:
    arm = f"n3.{cell['id']}"
    toks = res["tokens"][1:] if len(res["tokens"]) > 1 else res["tokens"]
    rows: list[dict] = []
    src = "cuda_events+host_monotonic"
    if toks:
        for field, key, tag, note in (
                ("wall_ms_token", "wall_ms", "measured", "host wall time of a token, first event to last synchronization"),
                ("compute_busy_ms_token", "busy_ms", "measured", "sum of the stand-in event durations (compute lane resource time)"),
                ("compute_stall_ms_token", "stall_ms", "measured", "compute stream wait for copies, event pair before and after the wait"),
                ("host_gap_ms_token", "gap_ms", "measured", "GPU timeline between a layer's end and the next layer's pre event, plus the lead"),
                ("copy_busy_demand_ms_token", "dem_busy_ms", "measured", "copy lane resource time of demand batches (not a wall time)"),
                ("copy_busy_prefetch_ms_token", "pf_busy_ms", "measured", "copy lane resource time of prefetch batches (not a wall time)"),
                ("host_issue_ms_token", "host_ms", "measured", "host time in bookkeeping, copy issue and launch")):
            s = suite.summarize([t[key] for t in toks], seed=1)
            rows.append(mf.row(run_id, arm, field, s["p50"], "ms", tag, source=src, n=s["n"], p50=s["p50"], p95=s["p95"],
                               ci_low=s.get("ci_low"), ci_high=s.get("ci_high"), ci_method=s.get("ci_method"), note=note))
    cl = closure(res["tokens"])
    rows.append(mf.row(run_id, arm, "closure_share", cl["share"], "fraction", "derived", source="wall against GPU span, tolerance 5 %",
                       n=cl["n"], note=f"median error {cl['median_error']}"))
    b = res.get("book")
    if b:
        for field, key in (("demand_misses_per_token", "demand_per_token"), ("prefetch_loads_per_token", "loads_per_token"),
                           ("covered_per_token", "covered_per_token"), ("wasted_per_token", "wasted_per_token")):
            rows.append(mf.row(run_id, arm, field, b[key], "count", "measured", source="replay bookkeeping", n=b["tokens"]))
        if hash_ok is not None:
            rows.append(mf.row(run_id, arm, "event_hash_equals_expected", 1 if hash_ok else 0, "bool", "measured", source="gate (a)"))
    rows.append(mf.row(run_id, arm, "readback_failures", res["readback_failures"], "count", "measured", source="gate (b)", n=res["readbacks"]))
    rows.append(mf.row(run_id, arm, "calibration_error", calib["error"], "fraction", "measured", source="gate (c): stand-in pair alone against C",
                       note=f"{calib['bytes']} weight bytes, capped={calib['capped']}"))
    if source_ms:
        qs = sorted(source_ms)
        rows.append(mf.row(run_id, arm, "source_copy_ms_median", qs[len(qs) // 2], "ms", "measured", source="source qualification, cuda events"))
        rows.append(mf.row(run_id, arm, "source_copy_ms_min", qs[0], "ms", "measured", source="source qualification, cuda events"))
        rows.append(mf.row(run_id, arm, "source_copy_ms_max", qs[-1], "ms", "measured", source="source qualification, cuda events"))
        rows.append(mf.row(run_id, arm, "source_records_slow_share", sum(1 for x in qs if x >= 1.5 * qs[0]) / len(qs), "fraction",
                           "derived", source="records at or above 1.5 x the fastest record's copy time"))
    return rows


def group_rows(run_id: str, group_id: str, c: float, walls: dict[str, float], busy: dict[str, float], c_per_copy: float | None) -> list[dict]:
    """Derived rows across arms of one (group, C): exposed time, hidden share, drift, compute interference, and the record's `c`."""
    eh = exposed_and_hidden(walls)
    arm = f"n3.{group_id}.C{c}"
    rows = []
    for key, val in eh.items():
        unit = "ms" if key.startswith(("E_", "floor")) else "fraction"
        rows.append(mf.row(run_id, arm, key, val, unit, "derived", source="p50 token wall times across arms, N controls bracketing"))
    if "N0" in busy and "N1" in busy:
        nb = (busy["N0"] + busy["N1"]) / 2
        for a in ("D", "P0", "P40", "O"):
            if a in busy and nb > 0:
                rows.append(mf.row(run_id, arm, f"compute_busy_ratio_{a}", busy[a] / nb - 1.0, "fraction", "derived",
                                   source="compute-lane busy p50 against the N controls (interference)"))
    if c_per_copy is not None:
        rows.append(mf.row(run_id, arm, "per_copy_ms_c", c_per_copy, "ms", "measured", source="median of the 64 qualified source records"))
    return rows


# ------------------------------------------------------------------------------------------------ CLI

def file_sha(path) -> str:
    return ra.file_sha256(path)


def cmd_plan(args) -> int:
    for prof in ("smoke", "full"):
        for g in cell_defs(prof):
            slots = ra.slots_for(g["percent"])
            cs = cells_of(g)
            print(f"{prof:5s} {g['id']:4s} {g['percent']:2d} % slots={slots:5d} arena={slots * ra.RECORD_BYTES / 2**30:5.1f} GiB "
                  f"tokens={g['tokens']} C={list(g['c_values'])} cells={len(cs)}: " + " ".join(c["id"].split('.', 1)[1] for c in cs))
    return 0


def cmd_expect(args) -> int:
    trace = ra.load_arrays(args.trace)
    with np.load(args.trace) as data:
        full = {k: data[k] for k in data.files if k != "segments"}
    out = {"trace_sha256": file_sha(args.trace), "record_bytes": ra.RECORD_BYTES, "n_layers": ra.trace_layer_count(trace), "cells": {}}
    seen = set()
    for prof in ("smoke", "full"):
        for g in cell_defs(prof):
            nl = out["n_layers"]
            slots = ra.slots_for(g["percent"])
            for mode in sorted({ARM_MODE[a] for a in g["arms"] if a in ARM_MODE}):
                key = f"{g['percent']}.{g['tokens']}.{mode}"
                if key in seen:
                    continue
                seen.add(key)
                ref = reference_pass_pf(full, slots, nl, mode, g["tokens"])
                mine = bookkeeping_pass_pf(full, slots, nl, mode, g["tokens"])
                ok = ref["hash"] == mine["hash"] and ref["demand_misses"] == mine["demand_misses"] and ref["prefetch_loads"] == mine["prefetch_loads"]
                out["cells"][key] = {"slots": slots, "hash": ref["hash"], "demand_per_token": ref["demand_per_token"],
                                     "loads_per_token": ref["loads_per_token"], "tokens": ref["tokens"], "arena_equals_reference": ok}
                print(f"{key:14s} slots={slots:5d} demand/token={ref['demand_per_token']:7.2f} loads/token={ref['loads_per_token']:7.2f} "
                      f"hash {ref['hash'][:12]} equal={ok}")
                if not ok:
                    print("FAIL: the arena bookkeeping differs from the independent reference: nothing downstream counts", file=sys.stderr)
                    return 4
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


def run_group(arrays_full, group: dict, expected: dict | None, run_id: str, dl: suite.Deadline, gates: dict, failures: list, raw: dict,
              *, rehearsal: bool = False, weight_start: int | None = None, calib_tolerance: float = CALIBRATION_TOLERANCE,
              nbytes: int = ra.RECORD_BYTES, readback_every: int = ra.READBACK_EVERY, drift_limit: float = DRIFT_LIMIT) -> tuple[list[dict], str]:
    """Run every cell of a group in order; returns (rows, status). Used by the CLI and, with `rehearsal=True`, by the Mac tests."""
    trace = {k: arrays_full[k] for k in ("phase", "layer", "position", "experts")}
    nl = ra.trace_layer_count(trace)
    slots = group.get("slots") or ra.slots_for(group["percent"])
    rows: list[dict] = []
    status = "complete"
    arena = OverlapArena(slots, nbytes, rehearsal_cpu=rehearsal, n_layers=nl)
    source_ms = arena.qualify_sources()
    c_per_copy = sorted(source_ms)[len(source_ms) // 2]
    for c in group["c_values"]:
        cs = [x for x in cells_of(group) if x["c"] == c]
        start_bytes = weight_start or (256 * 2**20)
        calib = arena.standin.calibrate(c, start_bytes, tolerance=calib_tolerance)
        raw.setdefault("calibration", {})[f"{group['id']}.C{c}"] = calib
        gates["calibration"] &= calib["ok"]
        if not calib["ok"]:
            failures.append({"cell": f"{group['id']}.C{c}", "error": f"stand-in calibration off by {calib['error']:.3f}"})
            continue
        walls: dict[str, float] = {}
        busy: dict[str, float] = {}
        for cell in cs:
            if dl.expired():
                status = "truncated"
                break
            try:
                mode = ARM_MODE.get(cell["arm"])
                hash_ok = None
                if mode is not None:
                    first = bookkeeping_pass_pf(arrays_full, slots, nl, mode, cell["tokens"])
                    if expected is not None:
                        exp = expected["cells"][f"{group['percent']}.{cell['tokens']}.{mode}"]
                        hash_ok = first["hash"] == exp["hash"]
                    else:
                        hash_ok = first["hash"] == reference_pass_pf(arrays_full, slots, nl, mode, cell["tokens"])["hash"]
                    gates["event_hash"] &= hash_ok
                    if not hash_ok:
                        failures.append({"cell": cell["id"], "error": "event hash differs from the Mac reference"})
                        continue
                cap = suite.Deadline(min(HARD_CELL_SECONDS, max(1.0, dl.remaining())))
                res = run_cell(arrays_full, slots, nl, arena, cell["arm"], max_decode_tokens=cell["tokens"], deadline=cap,
                               readback_every=readback_every)
                if mode is not None:
                    again = bookkeeping_pass_pf(arrays_full, slots, nl, mode, cell["tokens"])
                    gates["determinism"] &= again["hash"] == first["hash"] and (res["truncated"] or res["book"]["chain"] == first["chain"])
                    if not res["truncated"]:
                        gates["counts"] &= (res["book"]["demand_misses"] == first["demand_misses"]
                                            and res["book"]["prefetch_loads"] == first["prefetch_loads"])
                gates["readback"] &= res["readback_failures"] == 0
                cl = closure(res["tokens"])
                gates["closure"] &= cl["ok"]
                if res["truncated"]:
                    status = "truncated"
                rows += cell_rows(run_id, cell, res, hash_ok, calib, source_ms if cell["role"] == "N0" else None)
                toks = res["tokens"][1:] if len(res["tokens"]) > 1 else res["tokens"]
                if toks:
                    walls[cell["role"]] = p50([t["wall_ms"] for t in toks])
                    busy[cell["role"]] = p50([t["busy_ms"] for t in toks])
                raw[cell["id"]] = {"wall_ms": [t["wall_ms"] for t in res["tokens"]], "busy_ms": [t["busy_ms"] for t in res["tokens"]],
                                   "stall_ms": [t["stall_ms"] for t in res["tokens"]], "span_ms": [t["span_ms"] for t in res["tokens"]],
                                   "closure": cl}
            except Exception as exc:                                  # a failed cell is recorded, never hidden
                failures.append({"cell": cell["id"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        eh = exposed_and_hidden(walls)
        if "drift" in eh:
            gates["drift"] &= abs(eh["drift"]) <= drift_limit
            if abs(eh["drift"]) > drift_limit:
                failures.append({"cell": f"{group['id']}.C{c}", "error": f"N control drift {eh['drift']:.3f} above {drift_limit}"})
        rows += group_rows(run_id, group["id"], c, walls, busy, c_per_copy)
        if status == "truncated":
            break
    return rows, status


def cmd_run(args) -> int:
    groups = cell_defs(args.cells)
    if not args.execute:
        cmd_plan(args)
        print("\n# DRY RUN: nothing executed. Add --execute on the provisioned GPU host.", file=sys.stderr)
        return 0
    import doctor

    expected = json.loads(Path(args.expected).read_text())
    if file_sha(args.trace) != expected["trace_sha256"]:
        print("STOP: the trace file differs from the one the expected hashes were made from", file=sys.stderr)
        return 5
    report = doctor.collect(note=args.note, region=args.region, session_cap_s=int(args.max_seconds))
    ok, why = doctor.cuda_ready(report)
    if not ok:
        print(f"SKIP: no CUDA run possible here: {why}", file=sys.stderr)
        return 3
    with np.load(args.trace) as data:
        full = {k: data[k] for k in data.files if k != "segments"}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{stamp}-n3a-{args.cells}-{report['manifest_id'][:6]}"
    out = Path(args.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    mf.write_json(out / "machine-manifest.json", report)
    dl = suite.Deadline(args.max_seconds)
    gates = {"event_hash": True, "determinism": True, "counts": True, "readback": True, "calibration": True, "drift": True,
             "closure": True, "sanity": True}
    failures: list[dict] = []
    raw: dict = {}
    rows: list[dict] = []
    status = "complete"
    for g in groups:
        try:
            r, st = run_group(full, g, expected, run_id, dl, gates, failures, raw)
            rows += r
            if st == "truncated":
                status = "truncated"
        except Exception as exc:
            failures.append({"cell": g["id"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        import torch

        torch.cuda.empty_cache()
        if status == "truncated":
            break
    for r in rows:
        if r["field"] == "source_copy_ms_min" and r.get("value") is not None and r["value"] > 0:
            if ra.RECORD_BYTES / 1e9 / (r["value"] / 1e3) > suite.PCIE_SANITY_GB_S:
                gates["sanity"] = False
    manifest = {
        "schema": mf.SCHEMA_NAME, "schema_version": mf.SCHEMA_VERSION, "platform": mf.PLATFORM, "kind": "run",
        "created_utc": mf.utc_now(), "machine_manifest_id": report["manifest_id"], "run_id": run_id,
        "instrument": {"name": "benchmarks/cuda/overlap_arena.py", "version": "N3A-1"}, "code": report["code"], "mode": "timing",
        "workload": {"id": f"N3-stageA-{args.cells}", "class": "N3 layer-ahead prefetch with a stand-in compute and real copies (no model)",
                     "trace_sha256": expected["trace_sha256"], "record_bytes": ra.RECORD_BYTES, "source_records": ra.SOURCE_RECORDS,
                     "policy": "lru", "claims": "overlap mechanics only: exposed copy time against compute window, copy-lane saturation, interference"},
        "protocol": {"ordering": "per C: N start, D, P0, P40, O, N end; 400 decode tokens after the same prefill; token 0 a warm-up"},
        "limits": {"max_seconds": args.max_seconds, "hard_cell_seconds": HARD_CELL_SECONDS, "elapsed_s": round(dl.elapsed(), 3)},
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
    e = sub.add_parser("expect", help="Mac only: reference event hashes from the independent bookkeeping model")
    e.add_argument("--trace", required=True)
    e.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--trace", required=True)
    r.add_argument("--expected", required=True)
    r.add_argument("--cells", choices=("smoke", "full"), required=True)
    r.add_argument("--execute", action="store_true")
    r.add_argument("--max-seconds", type=float, default=1500.0)
    r.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "cuda"))
    r.add_argument("--note", default="")
    r.add_argument("--region")
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "expect": cmd_expect, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
