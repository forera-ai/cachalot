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

GATE_ARMS = {"G00": 0.0, "G10": 0.10, "G20": 0.20}                       # N3 stage C: P0L with a router-weight gate (G00 is P0L by construction)
ARMS = ("N", "D", "P0", "P0L", "P40", "O", "L2") + tuple(GATE_ARMS)
PRED_ARMS = ("P0", "P0L", "P40", "L2") + tuple(GATE_ARMS)
LEVELS = (0, 1, 2, 3, 4, 5, 6)                                            # N3 stage B: see the record, section 3; 6 = L3 plus an event after every copy (row timing)
COPY_EVENT_POOL = 768                                                     # per-copy events a token may use (two tokens in flight)
ARM_MODE = {"D": "none", "P0": "pred", "P0L": "pred", "P40": "pred", "O": "oracle",
            "L2": "pred~2",                                                       # N3 stage D: the lead-2 recorded set, issued two rows ahead
            **{a: f"pred@{t:.2f}" for a, t in GATE_ARMS.items()}}                  # the bookkeeping each arm uses ('pred@t': gated)


def mode_parts(mode: str) -> tuple[str, float | None]:
    """'pred@0.20' -> ('pred', 0.2); 'pred~2' -> ('pred', None); any other mode -> (mode, None)."""
    base, _, gate = mode.partition("~")[0].partition("@")
    return base, (float(gate) if gate else None)


def mode_lead(mode: str) -> int:
    """'pred~2' -> 2: the lead of the recorded predicted set the arm uses (and the number of rows it is issued ahead); default 1."""
    return int(mode.partition("~")[2]) if "~" in mode else 1
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

_MASK61 = (1 << 61) - 1


def chain2_step(chain: int, kind: int, tok: int, layer: int, expert: int, slot: int) -> int:
    """A rolling integer over every demand miss (kind 0) and every prefetch load (kind 1): the cheap stand-in for the event hash in
    the lean timed loop. The same formula is used by `PfBook`, `LeanBook` and `reference_pass_pf`, so equality is a real test."""
    return (chain * 1000003 + (((tok * 64 + layer) * 1024 + expert) * 8192 + slot) * 2 + kind) & _MASK61


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

    def __init__(self, slots: int, n_layers: int, mode: str, digest: bool = True):
        if mode_parts(mode)[0] not in ("none", "pred", "oracle"):
            raise ValueError(f"unknown mode {mode!r}")
        self.arena, self.mode, self.digest = ArenaPF(slots, n_layers), mode, digest
        self.pending: set[tuple[int, int]] = set()
        self.hash = hashlib.sha256()
        self.chain = 0
        self.chain2 = 0
        self.demand_misses = self.loads = self.covered = 0
        self.token_counts: dict[int, dict[str, int]] = {}

    def _tok(self, tok: int) -> dict[str, int]:
        return self.token_counts.setdefault(tok, {"demand": 0, "loads": 0, "covered": 0, "requests": 0})

    def prefill(self, layer: int, ordered_keys):
        hits, miss, ops, sets = self.arena.prefill_layer(layer, ordered_keys)
        if self.digest:
            ra._digest_feed(self.hash, ("P", layer, hits, miss, ops))
        return sets

    def prefetch(self, tok: int, layer: int, experts) -> list[tuple[int, tuple[int, int]]]:
        """Insert the keys (layer, e) for e in `experts`; returns [(slot, key)] of the loads to copy."""
        out = []
        for e in experts:
            key = (layer, int(e))
            loaded, slot, ops = self.arena.prefetch(key)
            if self.digest:
                ra._digest_feed(self.hash, ("F", tok, layer, key[1], loaded, ops))
            if loaded:
                self.chain2 = chain2_step(self.chain2, 1, tok, layer, key[1], slot)
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
            if self.digest:
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
                self.chain2 = chain2_step(self.chain2, 0, tok, layer, key[1], slot)
        return out

    def end_row(self, layer: int) -> None:
        self.pending = {k for k in self.pending if k[0] != layer}

    def summary(self) -> dict:
        n = max(1, len(self.token_counts))
        return {"hash": self.hash.hexdigest(), "chain": self.chain, "chain2": self.chain2, "tokens": len(self.token_counts),
                "demand_misses": self.demand_misses, "prefetch_loads": self.loads, "covered": self.covered,
                "wasted": self.loads - self.covered, "demand_per_token": self.demand_misses / n,
                "loads_per_token": self.loads / n, "covered_per_token": self.covered / n,
                "wasted_per_token": (self.loads - self.covered) / n,
                "token_demand": [self.token_counts[t]["demand"] for t in sorted(self.token_counts)]}


class LeanBook:
    """The lean timed-loop bookkeeping of stage B (`L2` and above): the same LRU decisions and slot assignments as `PfBook`, with no
    digest, no per-event tuples and no per-token dictionaries; a rolling integer `chain2` stands in for the event hash."""

    digest = False

    def __init__(self, slots: int, n_layers: int, mode: str):
        if mode_parts(mode)[0] not in ("none", "pred", "oracle"):
            raise ValueError(f"unknown mode {mode!r}")
        self.arena, self.mode = ArenaPF(slots, n_layers), mode
        self.pending: set[tuple[int, int]] = set()
        self.chain2 = 0
        self.demand_misses = self.loads = self.covered = self.tokens = 0
        self._last_tok = None

    def prefill(self, layer: int, ordered_keys):
        return self.arena.prefill_layer(layer, ordered_keys)[3]

    def prefetch(self, tok: int, layer: int, experts) -> list[tuple[int, tuple[int, int]]]:
        lru, free, slots, pending = self.arena.lru, self.arena.free, self.arena.slots, self.pending
        out, c = [], self.chain2
        for e in experts:
            key = (layer, e)
            if key in lru:
                continue
            while len(lru) >= slots:
                free.append(lru.pop(next(iter(lru))))
            slot = free.pop()
            lru[key] = slot
            pending.add(key)
            out.append((slot, key))
            c = (c * 1000003 + (((tok * 64 + layer) * 1024 + e) * 8192 + slot) * 2 + 1) & _MASK61
        self.chain2 = c
        self.loads += len(out)
        return out

    def demand(self, tok: int, layer: int, experts) -> list[tuple[int, tuple[int, int]]]:
        if tok != self._last_tok:
            self._last_tok = tok
            self.tokens += 1
        lru, free, slots, pending = self.arena.lru, self.arena.free, self.arena.slots, self.pending
        out, c = [], self.chain2
        for e in experts:
            key = (layer, e)
            if key in lru:
                if key in pending:
                    self.covered += 1
                    pending.discard(key)
                lru.move_to_end(key)
                continue
            while len(lru) >= slots:
                free.append(lru.pop(next(iter(lru))))
            slot = free.pop()
            lru[key] = slot
            out.append((slot, key))
            c = (c * 1000003 + (((tok * 64 + layer) * 1024 + e) * 8192 + slot) * 2) & _MASK61
        self.chain2 = c
        self.demand_misses += len(out)
        return out

    def end_row(self, layer: int) -> None:
        if self.pending:
            self.pending = {k for k in self.pending if k[0] != layer}

    def summary(self) -> dict:
        n = max(1, self.tokens)
        return {"hash": None, "chain": None, "chain2": self.chain2, "tokens": self.tokens, "demand_misses": self.demand_misses,
                "prefetch_loads": self.loads, "covered": self.covered, "wasted": self.loads - self.covered,
                "demand_per_token": self.demand_misses / n, "loads_per_token": self.loads / n, "covered_per_token": self.covered / n,
                "wasted_per_token": (self.loads - self.covered) / n}


def predicted_lookup(arrays, mode: str = "pred") -> dict[tuple[int, int], list[int]]:
    """The recorded predicted sets; for a gated mode ('pred@t') only the experts whose recorded router weight is at least t."""
    if "pred_target" not in arrays:
        return {}
    gate = mode_parts(mode)[1]
    if "pred_source" in arrays:                       # a trace recorded with several leads keeps only the records of the arm's lead
        keep = np.nonzero(arrays["pred_target"].astype(np.int64) - arrays["pred_source"].astype(np.int64) == mode_lead(mode))[0]
    else:
        keep = np.arange(len(arrays["pred_target"]))
    if gate is None:
        return {(int(arrays["pred_position"][i]), int(arrays["pred_target"][i])): arrays["pred_experts"][i].tolist() for i in keep}
    return {(int(arrays["pred_position"][i]), int(arrays["pred_target"][i])):
            [int(x) for x, w in zip(arrays["pred_experts"][i], arrays["pred_weights"][i], strict=True) if w >= gate] for i in keep}


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


def issue_ahead_ok(plan, i: int, lead: int) -> bool:
    """Stage D: the row `lead` ahead exists and every item up to it is a decode row of the same token (a prefill or the next token ends it)."""
    t = i + lead
    return t < len(plan) and all(plan[x][0] == "D" and plan[x][1] == plan[i][1] for x in range(i + 1, t + 1))


def bookkeeping_pass_pf(arrays, slots: int, n_layers: int, mode: str, max_decode_tokens: int | None) -> dict:
    book = PfBook(slots, n_layers, mode)
    pred = predicted_lookup(arrays, mode) if mode_parts(mode)[0] == "pred" else {}
    plan = decode_plan(arrays, max_decode_tokens)
    lead = mode_lead(mode)
    started: set[int] = set()
    for i, item in enumerate(plan):
        if item[0] == "P":
            book.prefill(item[1], item[2])
            continue
        _, tok, layer, experts, _pos = item
        if lead == 1 or i not in started:                # lead 1: the set of this row, right before its demand (stage A/B order)
            ahead = row_prefetch_set(mode, pred, plan, i)
            if ahead:
                book.prefetch(tok, layer, ahead)
        book.demand(tok, layer, experts)
        book.end_row(layer)
        if lead > 1 and issue_ahead_ok(plan, i, lead):   # stage D: the set of the row `lead` ahead, right after this row's demand
            ahead = row_prefetch_set(mode, pred, plan, i + lead)
            if ahead:
                book.prefetch(plan[i + lead][1], plan[i + lead][2], ahead)
            started.add(i + lead)
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
    pred = predicted_lookup(arrays, mode) if mode_parts(mode)[0] == "pred" else {}
    plan = decode_plan(arrays, max_decode_tokens)
    counts = {"demand": 0, "loads": 0, "covered": 0}
    chain2 = 0
    toks: set[int] = set()

    def load(key):
        ops = []
        while len(lru) >= slots:
            victim = next(iter(lru))
            free.append(lru.pop(victim))
            ops.append((0, victim[0], victim[1]))
        slot = free.pop()
        lru[key] = slot
        ops.append((1, key[0], key[1]))
        return tuple(ops), slot

    lead = mode_lead(mode)
    started: set[int] = set()

    def fetch(ahead, tok, layer, chain):
        for e in ahead or []:
            key = (layer, int(e))
            loaded = key not in lru
            ops, slot = load(key) if loaded else ((), -1)
            ra._digest_feed(h, ("F", tok, layer, key[1], loaded, ops))
            if loaded:
                pending.add(key)
                counts["loads"] += 1
                chain = chain2_step(chain, 1, tok, layer, key[1], slot)
        return chain

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
        if lead == 1 or i not in started:
            chain2 = fetch(row_prefetch_set(mode, pred, plan, i), tok, layer, chain2)
        for e in experts:
            key = (layer, int(e))
            hit = key in lru
            covered = hit and key in pending
            if hit:
                lru.move_to_end(key)
                ops = ()
            else:
                ops, slot = load(key)
                counts["demand"] += 1
                chain2 = chain2_step(chain2, 0, tok, layer, key[1], slot)
            ra._digest_feed(h, ("D", tok, layer, key[1], hit, covered, ops))
            if covered:
                counts["covered"] += 1
                pending.discard(key)
        pending = {k for k in pending if k[0] != layer}
        if lead > 1 and issue_ahead_ok(plan, i, lead):
            tgt = plan[i + lead]
            chain2 = fetch(row_prefetch_set(mode, pred, plan, i + lead), tgt[1], tgt[2], chain2)
            started.add(i + lead)
    n = max(1, len(toks))
    return {"hash": h.hexdigest(), "chain2": chain2, "tokens": len(toks), "demand_misses": counts["demand"], "prefetch_loads": counts["loads"],
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
    if profile == "b-smoke":
        return [{"id": "p17", "percent": 17, "tokens": 40, "plan": [
            {"level": 0, "c": 1.0, "arms": ("N", "D", "O", "P0L", "N")}, {"level": 3, "c": 1.0, "arms": ("N", "O", "N")},
            {"level": 4, "c": 1.0, "arms": ("N", "O", "N")}, {"level": 5, "c": 1.0, "arms": ("N", "O", "N")}]}]
    if profile == "b-full":
        return [{"id": "p17", "percent": 17, "tokens": DECODE_TOKENS, "plan": [
            {"level": 0, "c": 1.0, "arms": ("N", "D", "O", "P0", "P0L", "P40", "N")},
            {"level": 1, "c": 1.0, "arms": ("N", "O", "P0L", "N")}, {"level": 2, "c": 1.0, "arms": ("N", "O", "P0L", "N")},
            {"level": 3, "c": 1.0, "arms": ("N", "O", "P0L", "N")},
            {"level": 4, "c": 1.0, "arms": ("N", "D", "O", "P0", "P0L", "N")},
            {"level": 5, "c": 1.0, "arms": ("N", "D", "O", "P0L", "N")},
            {"level": 0, "c": 0.3, "arms": ("N", "D", "O", "P0", "P0L", "P40", "N")},
            {"level": 4, "c": 0.3, "arms": ("N", "D", "O", "P0", "P0L", "N")}]}]
    if profile == "b-rehearsal":
        return [{"id": "p17", "percent": 17, "slots": 40, "tokens": 6, "plan": [
            {"level": lv, "c": 0.05, "arms": ("N", "D", "O", "P0", "P0L", "P40", "N") if lv == 0 else ("N", "D", "O", "P0", "P0L", "N")}
            for lv in range(6)]}]
    if profile == "c-smoke":
        return [{"id": "p17", "percent": 17, "tokens": 40, "plan": [
            {"level": 5, "c": 1.0, "arms": ("N", "D", "P0L", "G20", "O", "N")}, {"level": 4, "c": 0.3, "arms": ("N", "D", "G10", "N")}]}]
    if profile == "c-full":                              # record N3C section 3; C = 0.3 at L4 and C = 1.0 at L5 to match stage B's baselines
        return [{"id": "p17", "percent": 17, "tokens": DECODE_TOKENS, "plan": [
                    {"level": 5, "c": 1.0, "arms": ("N", "D", "P0L", "G10", "G20", "O", "N")},
                    {"level": 4, "c": 0.3, "arms": ("N", "D", "P0L", "G10", "G20", "O", "N")}]},
                {"id": "p8", "percent": 8, "tokens": DECODE_TOKENS, "plan": [
                    {"level": 5, "c": 1.0, "arms": ("N", "D", "P0L", "G20", "N")}]}]
    if profile == "d-smoke":
        return [{"id": "p17", "percent": 17, "tokens": 40, "plan": [
            {"level": 5, "c": 1.0, "arms": ("N", "D", "P0L", "L2", "O", "N")}, {"level": 4, "c": 0.3, "arms": ("N", "D", "L2", "N")}]}]
    if profile == "d-full":                              # record N3D section 3
        arms6 = ("N", "D", "P0L", "L2", "O", "N")
        return [{"id": "p17", "percent": 17, "tokens": DECODE_TOKENS, "plan": [
                    {"level": 5, "c": 1.0, "arms": arms6}, {"level": 4, "c": 0.6, "arms": arms6}, {"level": 4, "c": 0.3, "arms": arms6}]},
                {"id": "p8", "percent": 8, "tokens": DECODE_TOKENS, "plan": [
                    {"level": 4, "c": 0.3, "arms": ("N", "D", "P0L", "L2", "N")}]}]
    if profile == "e-smoke":
        return [{"id": "p17", "percent": 17, "tokens": 40, "plan": [
            {"level": 5, "c": 1.0, "arms": ("N", "D", "P0L", "N")}, {"level": 4, "c": 0.3, "arms": ("N", "D", "P0L", "N")}]}]
    if profile == "e-full":                              # the bridge session (record CUDA-BRIDGE): run once per trace, order old, new, new, old
        arms4 = ("N", "D", "P0L", "N")
        return [{"id": "p17", "percent": 17, "tokens": DECODE_TOKENS, "plan": [
            {"level": 5, "c": 1.0, "arms": arms4}, {"level": 4, "c": 0.6, "arms": arms4}, {"level": 4, "c": 0.3, "arms": arms4}]}]
    if profile == "e-rehearsal":
        return [{"id": "p17", "percent": 17, "slots": 40, "tokens": 6, "plan": [
            {"level": 0, "c": 0.05, "arms": ("N", "D", "P0L", "N")}, {"level": 4, "c": 0.05, "arms": ("N", "D", "P0L", "N")}]}]
    if profile == "f-smoke":                             # row timing (record CUDA-ROW-TIMING): a short GPU smoke of both new levels
        return [{"id": "p17", "percent": 17, "tokens": 40, "plan": [
            {"level": 6, "c": 1.0, "arms": ("N", "D", "P0L", "N")}, {"level": 3, "c": 0.3, "arms": ("N", "D", "P0L", "N"), "rows": True}]}]
    if profile == "f-full":                              # record CUDA-ROW-TIMING section 3 with the section 7 erratum: 24 cells a run
        arms4 = ("N", "D", "P0L", "N")
        return [{"id": "p17", "percent": 17, "tokens": DECODE_TOKENS, "plan": [
            {"level": 3, "c": 1.0, "arms": arms4, "rows": True}, {"level": 3, "c": 0.3, "arms": arms4, "rows": True},
            {"level": 6, "c": 1.0, "arms": arms4}, {"level": 6, "c": 0.3, "arms": arms4},
            {"level": 4, "c": 1.0, "arms": arms4}, {"level": 4, "c": 0.3, "arms": arms4}]}]
    if profile == "f-rehearsal":
        return [{"id": "p17", "percent": 17, "slots": 40, "tokens": 6, "plan": [
            {"level": 3, "c": 0.05, "arms": ("N", "D", "P0L", "N"), "rows": True}, {"level": 6, "c": 0.05, "arms": ("N", "D", "P0L", "N")},
            {"level": 4, "c": 0.05, "arms": ("N", "D", "P0L", "N")}]}]
    if profile == "d-rehearsal":
        return [{"id": "p17", "percent": 17, "slots": 40, "tokens": 6, "plan": [
            {"level": 0, "c": 0.05, "arms": ("N", "D", "P0L", "L2", "O", "N")}, {"level": 4, "c": 0.05, "arms": ("N", "D", "P0L", "L2", "N")}]}]
    if profile == "c-rehearsal":
        return [{"id": "p17", "percent": 17, "slots": 40, "tokens": 6, "plan": [
            {"level": 0, "c": 0.05, "arms": ("N", "D", "P0L", "G00", "G10", "G20", "O", "N")},      # L5 needs a CUDA graph: not on the CPU
            {"level": 4, "c": 0.05, "arms": ("N", "P0L", "G10", "G20", "N")}]}]
    raise ValueError(f"unknown profile {profile!r}")


def cells_of(group: dict) -> list[dict]:
    """One entry per (C, arm) in run order; the first and last `N` of a C are its start and end controls."""
    out = []
    if "plan" in group:                                    # stage B: levels
        for entry in group["plan"]:
            for pos, arm in enumerate(entry["arms"]):
                role = "N0" if (arm == "N" and pos == 0) else ("N1" if arm == "N" else arm)
                out.append({"id": f"{group['id']}.L{entry['level']}.C{entry['c']}.{role}", "group": group["id"], "percent": group["percent"],
                            "tokens": group["tokens"], "c": entry["c"], "arm": arm, "role": role, "level": entry["level"],
                            "rows": bool(entry.get("rows", False))})
        return out
    for c in group["c_values"]:
        for pos, arm in enumerate(group["arms"]):
            role = "N0" if (arm == "N" and pos == 0) else ("N1" if arm == "N" else arm)
            out.append({"id": f"{group['id']}.C{c}.{role}", "group": group["id"], "percent": group["percent"], "tokens": group["tokens"],
                        "c": c, "arm": arm, "role": role, "level": 0})
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
        self.graph = None                                  # a captured graph points at the old weights
        self.n = max(2, int(nbytes) // 2)
        self.w = self.t.ones(self.n, dtype=self.t.float16, device=self.device)
        self.cut = max(1, int(self.n * SPLIT))

    def part1(self):
        return self.w[: self.cut].sum()

    def part2(self):
        return self.w[self.cut:].sum()

    def nbytes(self) -> int:
        return self.n * 2

    graph = None
    graph_error: str | None = None

    def build_graph(self) -> bool:
        """Capture the layer (both kernels) as one CUDA graph; False (with `graph_error`) when it cannot be done (always on the CPU)."""
        self.graph = None
        if self.rehearsal:
            self.graph_error = "no CUDA graphs on the CPU rehearsal"
            return False
        try:
            torch = self.t
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.part1()
                    self.part2()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.part1()
                self.part2()
            self.graph, self.graph_error = g, None
            return True
        except Exception as exc:                                   # said, never fatal
            self.graph, self.graph_error = None, f"{type(exc).__name__}: {str(exc)[:150]}"
            return False

    def run_layer(self) -> None:
        if self.graph is not None:
            self.graph.replay()
        else:
            self.part1()
            self.part2()

    def measure_graph(self, iters: int = 30, warm: int = 5) -> float:
        out = []
        a, b = self.events
        for i in range(iters + warm):
            self.t.cuda.synchronize()
            a.record()
            self.graph.replay()
            b.record()
            b.synchronize()
            if i >= warm:
                out.append(a.elapsed_time(b))
        return sorted(out)[len(out) // 2]

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
        self.views = None
        self.use_graph = False
        self.cpool = None                                  # per-copy events, built on first use (level 6)
        self.cev_on = False
        self.cev_next = [0, 0]
        self._mk_event = mk

    def enable_copy_events(self, on: bool) -> None:
        """Level 6: one event after every enqueued copy. The pool is built once, on first use."""
        self.cev_on = on
        if on and self.cpool is None:
            self.cpool = [[self._mk_event() for _ in range(COPY_EVENT_POOL)] for _ in range(2)]

    def start_token_marks(self, parity: int) -> None:
        self.cev_next[parity] = 0

    def prepare_views(self, on: bool) -> None:
        """Stage B level L3: the arena's slot views are made once, before timing, instead of one view per copy."""
        if not on:
            self.views = None
        elif self.views is None:
            self.views = [self._slot(i) for i in range(self.slots)]

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

    def enqueue(self, loads: list[tuple[int, tuple[int, int]]], marks: list[int] | None = None, parity: int = 0) -> None:
        """Queue real copies on the copy stream; no event, no wait. With `marks` (level 6) an event is recorded on the copy stream after
        every copy and its pool index appended to `marks`."""
        ctx = contextlib.nullcontext() if self.stream is None else self.t.cuda.stream(self.stream)
        nb = not self.rehearsal
        with ctx:
            if marks is not None and self.cev_on:
                views = self.views
                for slot, key in loads:
                    src = self.sources[(key[0] * 131 + key[1]) % ra.SOURCE_RECORDS] if views is not None else self.sources[ra.source_index(key)]
                    (views[slot] if views is not None else self._slot(slot)).copy_(src, non_blocking=nb)
                    i = self.cev_next[parity]
                    if i >= COPY_EVENT_POOL:
                        raise RuntimeError("the per-copy event pool is exhausted: a token issued more copies than COPY_EVENT_POOL")
                    self.cev_next[parity] = i + 1
                    self.rec_copy(self.cpool[parity][i])
                    marks.append(i)
            elif self.views is not None:
                views, sources = self.views, self.sources
                for slot, key in loads:
                    views[slot].copy_(sources[(key[0] * 131 + key[1]) % ra.SOURCE_RECORDS], non_blocking=nb)
            else:
                for slot, key in loads:
                    self._slot(slot).copy_(self.sources[ra.source_index(key)], non_blocking=nb)
        self.copies += len(loads)
        self.issued_copies += len(loads)

    def compute_part(self, which: int) -> None:
        (self.standin.part1 if which == 1 else self.standin.part2)()

    def compute_layer(self) -> None:
        """The whole layer in one call: one graph replay (L5) or both eager kernels."""
        if self.use_graph:
            self.standin.run_layer()
        else:
            self.standin.part1()
            self.standin.part2()

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


ROW_FIELDS = ("n_dem", "n_pf", "n_pf_ahead", "pre", "start", "mid", "end", "dem_a", "dem_b", "pf_a", "pf_b")
COPY_FIELDS = ("row", "kind", "k", "t_done", "dur")


def detail_record(arena, parity: int, n: int, had_demand: list[bool], had_pf: list[bool], det: dict) -> dict:
    """Row timing: every event of a token as a time in ms since the token's first event (`rows`, one list per decode row, `ROW_FIELDS`),
    and at level 6 the completion time and duration of every copy (`copies`, `COPY_FIELDS`; kind 0 demand, 1 prefetch, k the position in
    the batch, duration against the previous event on the copy stream: the batch's first event for k = 0)."""
    ev = arena.pool[parity]
    ts = arena.tok_start[parity]

    def rel(e) -> float:
        return round(arena.elapsed(ts, e), 4)

    rows, copies = [], []
    for i in range(n):
        da, db = (rel(ev["dem_a"][i]), rel(ev["dem_b"][i])) if had_demand[i] else (None, None)
        pa, pb = (rel(ev["pf_a"][i]), rel(ev["pf_b"][i])) if had_pf[i] else (None, None)
        rows.append([det["ndem"][i], det["npf"][i], det["ahead"][i], rel(ev["pre"][i]), rel(ev["start"][i]), rel(ev["mid"][i]),
                     rel(ev["end"][i]), da, db, pa, pb])
        if arena.cev_on and arena.cpool is not None:
            for kind, marks, first in ((0, det["dm"][i], ev["dem_a"][i]), (1, det["pm"][i], ev["pf_a"][i])):
                prev = first
                for k, idx in enumerate(marks):
                    e = arena.cpool[parity][idx]
                    copies.append([i, kind, k, rel(e), round(arena.elapsed(prev, e), 4)])
                    prev = e
    return {"rows": rows, "copies": copies}


ROWSUM_TOLERANCE_MS = 0.02                                                # rows are stored rounded to 1e-4 ms: 40 rows a token stay far below this


def detail_check(tokens: list[dict]) -> dict:
    """Gates (h) and (i): per token the per-row records re-add to the token's per-token sums (stall, busy), and at level 6 there is one
    per-copy record for every copy the bookkeeping issued (rows with `n_dem`/`n_pf`)."""
    rowsum_ok = copyevents_ok = True
    worst = 0.0
    for t in tokens:
        rows = t.get("rows")
        if not rows:
            continue
        i_pre, i_start, i_end = ROW_FIELDS.index("pre"), ROW_FIELDS.index("start"), ROW_FIELDS.index("end")
        stall = sum(r[i_start] - r[i_pre] for r in rows)
        busy = sum(r[i_end] - r[i_start] for r in rows)
        err = max(abs(stall - t["stall_ms"]), abs(busy - t["busy_ms"]))
        worst = max(worst, err)
        if err > ROWSUM_TOLERANCE_MS * max(1, len(rows)) / 10:
            rowsum_ok = False
        if t.get("copies"):
            issued = sum(r[0] + r[1] for r in rows)
            if len(t["copies"]) != issued:
                copyevents_ok = False
    return {"rowsum_ok": rowsum_ok, "copyevents_ok": copyevents_ok, "worst_rowsum_error_ms": worst}


def make_book(slots: int, n_layers: int, mode: str, level: int):
    """The bookkeeping object of a level: `L0` digest on, `L1` digest off, `L2` and above the lean book."""
    if level >= 2:
        return LeanBook(slots, n_layers, mode)
    return PfBook(slots, n_layers, mode, digest=(level == 0))


def run_cell(arrays, slots: int, n_layers: int, arena: OverlapArena, arm: str, *, level: int = 0, max_decode_tokens: int | None = None,
             readback_every: int = ra.READBACK_EVERY, deadline: suite.Deadline | None = None, seed: int = 0, row_detail: bool = False) -> dict:
    """The timed replay of one arm at one level. `N`: stand-in plus per-layer synchronization only. Otherwise the bookkeeping and
    copies of the arm (N3 record section 3, stage B record section 3 for the levels). Prefill admissions populate the arena untimed.
    Token 0 is a warm-up and is excluded from the timing statistics (not from the counts).

    Levels (cumulative): 0 stage A as run; 1 no digest; 2 lean bookkeeping; 3 pre-built slot views; 4 bare (only the events
    correctness needs, no per-layer breakdown); 5 the layer as one CUDA-graph replay (when `arena.use_graph`); 6 level 3 plus an event
    after every copy (row timing, `docs/CUDA-ROW-TIMING-RECORD.md`: implies `row_detail`).
    `row_detail` (levels 0 to 3 and 6, arms with a book) also returns every event of every token as a per-row record (`detail_record`)."""
    mode = ARM_MODE.get(arm)
    pred = predicted_lookup(arrays, mode) if mode and mode_parts(mode)[0] == "pred" else {}
    plan = decode_plan(arrays, max_decode_tokens)
    book = make_book(slots, n_layers, mode or "none", level) if arm != "N" else None
    instr = level < 4 or level == 6
    row_detail = (row_detail or level == 6) and book is not None
    copy_ev = level == 6 and book is not None
    arena.enable_copy_events(copy_ev)
    arena.prepare_views(level >= 3)
    arena.use_graph = level == 5 and arena.standin.graph is not None
    tokens: list[dict] = []
    readbacks = readback_fail = 0
    next_check = readback_every
    truncated = False
    cur_tok = None
    pending_records: tuple | None = None                  # (parity, n, had_demand, had_pf, wall_ms, host_ms, tok, detail) of the previous token
    carry_pf = 0                                          # copies the previous token's last row left queued for this token's first row
    parity = 0
    early = arm in ("P0", "O")                            # prefetch issued before the layer's launch
    late_nowait = arm in ("P0L", "L2") or arm in GATE_ARMS
    lead = mode_lead(mode) if mode else 1                         # after the launch, no wait
    late_wait = arm == "P40"                              # after the launch, after the first kernel through an event wait

    def flush(rec) -> None:
        par, n, hd, hp, wall, host, tok, det = rec
        r = token_record(arena, par, n, hd, hp)
        r.update(wall_ms=wall, host_ms=host, tok=tok)
        if det is not None:
            r.update(detail_record(arena, par, n, hd, hp, det))
        tokens.append(r)

    def enq(loads: list, marks: list | None) -> None:
        if marks is None:
            arena.enqueue(loads)                              # the stage A to E call, unchanged
        else:
            arena.enqueue(loads, marks, parity)

    def prefetch_for(idx: int) -> list:
        """Bookkeeping of the prefetch for plan item `idx` (a D row); loads to copy."""
        ahead = row_prefetch_set(mode, pred, plan, idx) if mode else None
        if not ahead:
            return []
        it = plan[idx]
        return book.prefetch(it[1], it[2], ahead)

    k = 0
    started_pf = set()                                    # D items whose prefetch was already done at the previous boundary
    perf = time.perf_counter
    while k < len(plan):
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
        j = k
        while j < len(plan) and plan[j][0] == "D" and plan[j][1] == tok:
            j += 1
        rows = range(k, j)
        n = j - k
        if n > arena.n_layers:
            raise ValueError("a token has more layers than the event pool")
        ev = arena.pool[parity]
        had_demand, had_pf = [False] * n, [False] * n
        det = {"ndem": [0] * n, "npf": [0] * n, "ahead": [0] * n, "dm": [[] for _ in range(n)], "pm": [[] for _ in range(n)]} if row_detail else None
        if copy_ev:
            arena.start_token_marks(parity)
        host_s = 0.0
        # no device synchronize here: a prefetch still in flight from the previous token's last boundary (arm `O` only) belongs to this
        # token's first layer and must be waited for by its compute, not hidden outside the timed span
        w0 = perf()
        if instr:
            arena.rec_compute(arena.tok_start[parity])
        for pos, idx in enumerate(rows):
            h0 = perf()
            _, _t, layer, experts, _p = plan[idx]
            demand_loads: list = []
            pf_loads: list = []
            if book is not None:
                if idx not in started_pf:                 # the first row after a prefill (or of the run) has no earlier boundary
                    demand_loads = prefetch_for(idx)      # needed by this row now: the row's compute waits for them like a miss
                demand_loads = demand_loads + book.demand(tok, layer, experts)
                book.end_row(layer)
                nxt = idx + lead
                if (nxt < len(plan) and plan[nxt][0] == "D") if lead == 1 else issue_ahead_ok(plan, idx, lead):
                    pf_loads = prefetch_for(nxt)
                    started_pf.add(nxt)
                # the batch event is always recorded and waited for: it completes after every copy queued before it, so a row whose
                # keys were all prefetched still waits for them (a stall) and never runs ahead of their arrival
                if instr:
                    arena.rec_copy(ev["dem_a"][pos])
                enq(demand_loads, det["dm"][pos] if copy_ev else None)
                arena.rec_copy(ev["dem_b"][pos])
                had_demand[pos] = True
                if det is not None:
                    det["ndem"][pos] = len(demand_loads)
                    det["ahead"][pos] = carry_pf
            if pf_loads and early:
                if instr:
                    arena.rec_copy(ev["pf_a"][pos])
                enq(pf_loads, det["pm"][pos] if copy_ev else None)
                if instr:
                    arena.rec_copy(ev["pf_b"][pos])
                had_pf[pos] = True
            if instr:
                arena.rec_compute(ev["pre"][pos])
            if had_demand[pos]:
                arena.compute_wait(ev["dem_b"][pos])
            if instr:
                arena.rec_compute(ev["start"][pos])
                arena.compute_part(1)
                arena.rec_compute(ev["mid"][pos])
                arena.compute_part(2)
            else:
                arena.compute_layer()
            arena.rec_compute(ev["end"][pos])
            if pf_loads and late_nowait:
                if instr:
                    arena.rec_copy(ev["pf_a"][pos])
                enq(pf_loads, det["pm"][pos] if copy_ev else None)
                if instr:
                    arena.rec_copy(ev["pf_b"][pos])
                had_pf[pos] = True
            elif pf_loads and late_wait:
                arena.copy_wait(ev["mid"][pos])           # the event is recorded already: the copy stream waits for it
                arena.rec_copy(ev["pf_a"][pos])
                arena.enqueue(pf_loads)
                arena.rec_copy(ev["pf_b"][pos])
                had_pf[pos] = True
            host_s += perf() - h0
            if det is not None:
                det["npf"][pos] = len(pf_loads)
            carry_pf = len(pf_loads)
            arena.sync_compute(ev["end"][pos])            # the router synchronization
        wall = (perf() - w0) * 1e3
        if instr:
            rec = (parity, n, had_demand, had_pf, wall, host_s * 1e3, tok, det)
            if pending_records is not None:
                flush(pending_records)
            pending_records = rec
        else:
            tokens.append({"wall_ms": wall, "host_ms": host_s * 1e3, "tok": tok, "bare": True})
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
    arena.use_graph = False
    return {"arm": arm, "level": level, "tokens": tokens, "truncated": truncated, "readbacks": readbacks,
            "readback_failures": readback_fail, "book": book.summary() if book is not None else None, "memory": arena.memory(),
            "nbytes": arena.nbytes, "issued_copies": arena.issued_copies}


# ------------------------------------------------------------------------------------------------ analysis

def closure(tokens: list[dict]) -> dict:
    """N3-P10: per token, |wall - GPU span| / wall where span = lead + stalls + busy; the share of tokens within tolerance."""
    use = tokens[1:] if len(tokens) > 1 else tokens
    if use and use[0].get("bare"):                           # stage B levels 4 and 5 record no per-layer events: nothing to close
        return {"share": None, "median_error": None, "n": 0, "ok": True, "meets_prediction": None, "bare": True}
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
    for arm in ("D", "P0", "P0L", "P40", "O", "L2", *GATE_ARMS):
        if arm in walls:
            out[f"E_{arm}"] = walls[arm] - floor
    if "E_D" in out and out["E_D"] > 0:
        for arm in ("P0", "P0L", "P40", "O", "L2", *GATE_ARMS):
            if f"E_{arm}" in out:
                out[f"H_{arm}"] = 1.0 - out[f"E_{arm}"] / out["E_D"]
    return out


def cell_rows(run_id: str, cell: dict, res: dict, hash_ok: bool | None, calib: dict, source_ms: list[float] | None) -> list[dict]:
    arm = f"n3.{cell['id']}"
    toks = res["tokens"][1:] if len(res["tokens"]) > 1 else res["tokens"]
    rows: list[dict] = []
    src = "cuda_events+host_monotonic"
    bare = bool(toks) and toks[0].get("bare", False)
    if toks:
        for field, key, tag, note in (
                ("wall_ms_token", "wall_ms", "measured", "host wall time of a token, first event to last synchronization"),
                ("compute_busy_ms_token", "busy_ms", "measured", "sum of the stand-in event durations (compute lane resource time)"),
                ("compute_stall_ms_token", "stall_ms", "measured", "compute stream wait for copies, event pair before and after the wait"),
                ("host_gap_ms_token", "gap_ms", "measured", "GPU timeline between a layer's end and the next layer's pre event, plus the lead"),
                ("copy_busy_demand_ms_token", "dem_busy_ms", "measured", "copy lane resource time of demand batches (not a wall time)"),
                ("copy_busy_prefetch_ms_token", "pf_busy_ms", "measured", "copy lane resource time of prefetch batches (not a wall time)"),
                ("host_issue_ms_token", "host_ms", "measured", "host time in bookkeeping, copy issue and launch")):
            if bare and key not in ("wall_ms", "host_ms"):
                continue                                     # stage B levels 4 and 5 record no per-layer events
            s = suite.summarize([t[key] for t in toks], seed=1)
            rows.append(mf.row(run_id, arm, field, s["p50"], "ms", tag, source=src, n=s["n"], p50=s["p50"], p95=s["p95"],
                               ci_low=s.get("ci_low"), ci_high=s.get("ci_high"), ci_method=s.get("ci_method"), note=note))
    cl = closure(res["tokens"])
    if cl.get("bare"):
        rows.append(mf.row(run_id, arm, "closure_share", None, "fraction", "unavailable", source="level 4 and above record no per-layer events",
                           note="no closure at this level"))
    else:
        rows.append(mf.row(run_id, arm, "closure_share", cl["share"], "fraction", "derived", source="wall against GPU span, tolerance 5 %",
                           n=cl["n"], note=f"median error {cl['median_error']}"))
    b = res.get("book")
    if b:
        if b.get("chain2") is not None:
            rows.append(mf.row(run_id, arm, "bookkeeping_chain2", float(b["chain2"] % (1 << 52)), "count", "measured",
                               source="rolling integer over every demand miss and prefetch load, low 52 bits"))
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


def group_rows(run_id: str, group_id: str, c: float, walls: dict[str, float], busy: dict[str, float], c_per_copy: float | None,
               ed_ref: float | None = None) -> list[dict]:
    """Derived rows across arms of one (group, C): exposed time, hidden share, drift, compute interference, and the record's `c`.
    `group_id` may carry the level (`p17.L3`). `ed_ref` is the same C's level-0 `E_D`, used for `H_*_vs_L0_E_D` where a level has no `D`."""
    eh = exposed_and_hidden(walls)
    arm = f"n3.{group_id}.C{c}"
    rows = []
    for key, val in eh.items():
        unit = "ms" if key.startswith(("E_", "floor")) else "fraction"
        rows.append(mf.row(run_id, arm, key, val, unit, "derived", source="p50 token wall times across arms, N controls bracketing"))
    if ed_ref and "E_D" not in eh:
        for a in ("P0", "P0L", "P40", "O", "L2", *GATE_ARMS):
            if f"E_{a}" in eh:
                rows.append(mf.row(run_id, arm, f"H_{a}_vs_L0_E_D", 1.0 - eh[f"E_{a}"] / ed_ref, "fraction", "derived",
                                   source="this level's exposed time against the level-0 demand exposure of the same C"))
    if "N0" in busy and "N1" in busy:
        nb = (busy["N0"] + busy["N1"]) / 2
        for a in ("D", "P0", "P0L", "P40", "O", "L2", *GATE_ARMS):
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
    for prof in ("smoke", "full", "b-smoke", "b-full", "c-smoke", "c-full", "d-smoke", "d-full", "e-smoke", "e-full", "f-smoke", "f-full"):
        for g in cell_defs(prof):
            slots = ra.slots_for(g["percent"])
            cs = cells_of(g)
            cvals = sorted({c["c"] for c in cs})
            print(f"{prof:7s} {g['id']:4s} {g['percent']:2d} % slots={slots:5d} arena={slots * ra.RECORD_BYTES / 2**30:5.1f} GiB "
                  f"tokens={g['tokens']} C={cvals} cells={len(cs)}: " + " ".join(c["id"].split('.', 1)[1] for c in cs))
    return 0


def cmd_expect(args) -> int:
    trace = ra.load_arrays(args.trace)
    with np.load(args.trace) as data:
        full = {k: data[k] for k in data.files if k != "segments"}
    out = {"trace_sha256": file_sha(args.trace), "record_bytes": ra.RECORD_BYTES, "n_layers": ra.trace_layer_count(trace), "cells": {}}
    seen = set()
    for prof in ("smoke", "full", "c-smoke", "c-full", "d-smoke", "d-full"):
        for g in cell_defs(prof):
            nl = out["n_layers"]
            slots = ra.slots_for(g["percent"])
            arms = {a for e in g["plan"] for a in e["arms"]} if "plan" in g else set(g["arms"])
            for mode in sorted({ARM_MODE[a] for a in arms if a in ARM_MODE}):
                key = f"{g['percent']}.{g['tokens']}.{mode}"
                if key in seen:
                    continue
                seen.add(key)
                ref = reference_pass_pf(full, slots, nl, mode, g["tokens"])
                mine = bookkeeping_pass_pf(full, slots, nl, mode, g["tokens"])
                ok = ref["hash"] == mine["hash"] and ref["demand_misses"] == mine["demand_misses"] and ref["prefetch_loads"] == mine["prefetch_loads"]
                out["cells"][key] = {"slots": slots, "hash": ref["hash"], "demand_per_token": ref["demand_per_token"],
                                     "loads_per_token": ref["loads_per_token"], "tokens": ref["tokens"], "arena_equals_reference": ok}
                if mode_parts(mode)[1] is not None:       # gate (g): equal to the offline coverage model on the same tokens
                    import prefetch_coverage as pcov

                    cov = pcov.coverage(full, g["percent"], "pred", max_decode_tokens=g["tokens"], min_weight=mode_parts(mode)[1])
                    gok = (abs(cov["demand_misses_per_token"] - ref["demand_per_token"]) < 1e-9
                           and abs(cov["prefetch_loads_per_token"] - ref["loads_per_token"]) < 1e-9)
                    out["cells"][key].update({"coverage_demand_per_token": cov["demand_misses_per_token"],
                                              "coverage_loads_per_token": cov["prefetch_loads_per_token"], "equals_coverage_model": gok})
                    if not gok:
                        print(f"FAIL: gated bookkeeping {key} differs from prefetch_coverage.py", file=sys.stderr)
                        return 4
                elif mode_lead(mode) > 1:                  # gate (g) for the lead-2 arm: the timed coverage model, issue point two rows ahead
                    import prefetch_coverage as pcov

                    cov = pcov.coverage_timed(full, g["percent"], mode_lead(mode), g["tokens"])
                    gok = (abs(cov["demand_misses_per_token"] - ref["demand_per_token"]) < 1e-9
                           and abs(cov["prefetch_loads_per_token"] - ref["loads_per_token"]) < 1e-9)
                    out["cells"][key].update({"coverage_demand_per_token": cov["demand_misses_per_token"],
                                              "coverage_loads_per_token": cov["prefetch_loads_per_token"], "equals_coverage_model": gok})
                    if not gok:
                        print(f"FAIL: lead-{mode_lead(mode)} bookkeeping {key} differs from prefetch_coverage.py --timed", file=sys.stderr)
                        return 4
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
    """Run every cell of a group in order; returns (rows, status). Used by the CLI and, with `rehearsal=True`, by the Mac tests.
    A group with a `plan` (stage B) runs its (level, C) entries in order; otherwise stage A's C values with level 0."""
    trace = {k: arrays_full[k] for k in ("phase", "layer", "position", "experts")}
    nl = ra.trace_layer_count(trace)
    slots = group.get("slots") or ra.slots_for(group["percent"])
    rows: list[dict] = []
    status = "complete"
    arena = OverlapArena(slots, nbytes, rehearsal_cpu=rehearsal, n_layers=nl)
    source_ms = arena.qualify_sources()
    c_per_copy = sorted(source_ms)[len(source_ms) // 2]
    entries = group.get("plan") or [{"level": 0, "c": c, "arms": group["arms"]} for c in group["c_values"]]
    calib_for: dict[float, dict] = {}
    ed_ref: dict[float, float] = {}
    graph_ok: dict[float, bool] = {}
    first_cell_of_run = True
    for entry in entries:
        c, level = entry["c"], entry.get("level", 0)
        label = f"{group['id']}.L{level}" if "plan" in group else group["id"]
        if c not in calib_for:
            calib = arena.standin.calibrate(c, weight_start or (256 * 2**20), tolerance=calib_tolerance)
            calib_for[c] = calib
            raw.setdefault("calibration", {})[f"{group['id']}.C{c}"] = calib
            gates["calibration"] &= calib["ok"]
            if not calib["ok"]:
                failures.append({"cell": f"{group['id']}.C{c}", "error": f"stand-in calibration off by {calib['error']:.3f}"})
        if not calib_for[c]["ok"]:
            continue
        calib = calib_for[c]
        if level == 5:
            if c not in graph_ok:
                built = arena.standin.build_graph()
                gc = None
                if built:
                    gc = calibration_result(c, arena.standin.measure_graph(), arena.standin.nbytes(), calib_tolerance)
                graph_ok[c] = bool(built and gc and gc["ok"])
                raw.setdefault("graph", {})[f"{group['id']}.C{c}"] = {"built": built, "error": arena.standin.graph_error, "calibration": gc}
            if not graph_ok[c]:
                rows.append(mf.row(run_id, f"n3.{label}.C{c}", "graph_unavailable", None, "text", "unavailable", source="CUDA graph",
                                   note=(arena.standin.graph_error or "graph replay time outside the calibration tolerance")[:150]))
                continue
        cs = [x for x in cells_of(group) if x["c"] == c and x.get("level", 0) == level]
        walls: dict[str, float] = {}
        busy: dict[str, float] = {}
        for cell in cs:
            if dl.expired():
                status = "truncated"
                break
            try:
                mode = ARM_MODE.get(cell["arm"])
                hash_ok = None
                first = None
                if mode is not None:
                    first = bookkeeping_pass_pf(arrays_full, slots, nl, mode, cell["tokens"])
                    if expected is not None:
                        exp = expected["cells"][f"{group['percent']}.{cell['tokens']}.{mode}"]
                        hash_ok = first["hash"] == exp["hash"]
                    else:
                        hash_ok = first["hash"] == reference_pass_pf(arrays_full, slots, nl, mode, cell["tokens"])["hash"]
                    gates["event_hash"] &= hash_ok
                    if expected is not None and "coverage_demand_per_token" in exp:      # gate (g)
                        gates.setdefault("coverage", True)
                        gates["coverage"] &= (abs(first["demand_per_token"] - exp["coverage_demand_per_token"]) < 1e-9
                                              and abs(first["loads_per_token"] - exp["coverage_loads_per_token"]) < 1e-9)
                    if not hash_ok:
                        failures.append({"cell": cell["id"], "error": "event hash differs from the Mac reference"})
                        continue
                cap = suite.Deadline(min(HARD_CELL_SECONDS, max(1.0, dl.remaining())))
                res = run_cell(arrays_full, slots, nl, arena, cell["arm"], level=level, max_decode_tokens=cell["tokens"], deadline=cap,
                               readback_every=readback_every, row_detail=cell.get("rows", False))
                if mode is not None:
                    again = bookkeeping_pass_pf(arrays_full, slots, nl, mode, cell["tokens"])
                    b = res["book"]
                    same = again["hash"] == first["hash"]
                    if level < 2:
                        same = same and (res["truncated"] or b["chain"] == first["chain"])
                    gates["determinism"] &= same
                    if not res["truncated"]:
                        gates["counts"] &= b["demand_misses"] == first["demand_misses"] and b["prefetch_loads"] == first["prefetch_loads"]
                        gates.setdefault("chain2", True)
                        gates["chain2"] &= b["chain2"] == first["chain2"]           # slot assignments equal the reference's
                gates["readback"] &= res["readback_failures"] == 0
                if any("rows" in t for t in res["tokens"]):
                    chk = detail_check(res["tokens"])
                    gates.setdefault("rowsum", True)
                    gates.setdefault("copyevents", True)
                    gates["rowsum"] &= chk["rowsum_ok"]
                    gates["copyevents"] &= chk["copyevents_ok"]
                    if not (chk["rowsum_ok"] and chk["copyevents_ok"]):
                        failures.append({"cell": cell["id"], "error": f"row detail inconsistent: {chk}"})
                cl = closure(res["tokens"])
                gates["closure"] &= cl["ok"]
                if res["truncated"]:
                    status = "truncated"
                rows += cell_rows(run_id, cell, res, hash_ok, calib, source_ms if first_cell_of_run else None)
                first_cell_of_run = False
                toks = res["tokens"][1:] if len(res["tokens"]) > 1 else res["tokens"]
                if toks:
                    walls[cell["role"]] = p50([t["wall_ms"] for t in toks])
                    if "busy_ms" in toks[0]:
                        busy[cell["role"]] = p50([t["busy_ms"] for t in toks])
                raw[cell["id"]] = {"wall_ms": [t["wall_ms"] for t in res["tokens"]], "host_ms": [t["host_ms"] for t in res["tokens"]],
                                   "closure": cl}
                if any("rows" in t for t in res["tokens"]):
                    raw[cell["id"]].update({"tok": [t["tok"] for t in res["tokens"]], "row_fields": list(ROW_FIELDS),
                                            "rows": [t.get("rows") for t in res["tokens"]], "copy_fields": list(COPY_FIELDS),
                                            "copies": [t.get("copies") for t in res["tokens"]]})
                if res["tokens"] and "busy_ms" in res["tokens"][0]:
                    raw[cell["id"]].update({"busy_ms": [t["busy_ms"] for t in res["tokens"]],
                                            "stall_ms": [t["stall_ms"] for t in res["tokens"]],
                                            "span_ms": [t["span_ms"] for t in res["tokens"]]})
            except Exception as exc:                                  # a failed cell is recorded, never hidden
                failures.append({"cell": cell["id"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        eh = exposed_and_hidden(walls)
        if "drift" in eh:
            gates["drift"] &= abs(eh["drift"]) <= drift_limit
            if abs(eh["drift"]) > drift_limit:
                failures.append({"cell": f"{label}.C{c}", "error": f"N control drift {eh['drift']:.3f} above {drift_limit}"})
        if level == 0 and "E_D" in eh:
            ed_ref[c] = eh["E_D"]
        rows += group_rows(run_id, label, c, walls, busy, c_per_copy, ed_ref.get(c))
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
    stage = {"f": "n3f", "e": "n3e", "d": "n3d", "c": "n3c", "b": "n3b"}.get(args.cells[:1] if args.cells[1:2] == "-" else "", "n3a")
    run_id = f"{stamp}-{stage}-{args.cells}-{report['manifest_id'][:6]}"
    out = Path(args.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    mf.write_json(out / "machine-manifest.json", report)
    dl = suite.Deadline(args.max_seconds)
    gates = {"event_hash": True, "determinism": True, "counts": True, "chain2": True, "readback": True, "calibration": True, "drift": True,
             "closure": True, "sanity": True, "coverage": True}
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
        "instrument": {"name": "benchmarks/cuda/overlap_arena.py", "version": {"n3f": "N3F-1", "n3e": "N3E-1", "n3d": "N3D-1", "n3c": "N3C-1", "n3b": "N3B-1"}.get(stage, "N3A-1")}, "code": report["code"], "mode": "timing",
        "workload": {"id": f"N3-stage{stage[-1].upper()}-{args.cells}", "class": "N3 layer-ahead prefetch with a stand-in compute and real copies (no model)",
                     "trace_sha256": expected["trace_sha256"], "record_bytes": ra.RECORD_BYTES, "source_records": ra.SOURCE_RECORDS,
                     "policy": "lru", "claims": "overlap mechanics only: exposed copy time against compute window, copy-lane saturation, interference"},
        "protocol": {"ordering": "per (level, C): N start, the arms, N end; 400 decode tokens after the same prefill; token 0 a warm-up"},
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
    r.add_argument("--cells", choices=("smoke", "full", "b-smoke", "b-full", "c-smoke", "c-full", "d-smoke", "d-full", "e-smoke", "e-full", "f-smoke", "f-full"), required=True)
    r.add_argument("--execute", action="store_true")
    r.add_argument("--max-seconds", type=float, default=1500.0)
    r.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "cuda"))
    r.add_argument("--note", default="")
    r.add_argument("--region")
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "expect": cmd_expect, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
