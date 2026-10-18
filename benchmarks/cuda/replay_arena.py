"""
N2 stage 1 harness (docs/CUDA-N2S1-EXPERIMENT-RECORD.md): a real fixed-size slot arena and real pinned host-to-device copies
driven by a routing trace. Standalone: it imports nothing from `src/cachalot`, because the GPU host has only
`benchmarks/cuda/` and numpy and torch.

Layering, so the Mac can test everything except the GPU calls:

* `ArenaLRU`: the cache bookkeeping, written from the store's specification and **independent of
  `benchmarks/simulate_policies.Store`**. Its decisions are compared with the simulator's event for event (gate (a));
  a mismatch is a harness bug until proven otherwise.
* `walk()`: turns trace arrays into the same request sequence `cache_sim` replays (prefill layer by layer, then decode rows).
* `bookkeeping_pass()` / `reference_pass()`: the hash of the event stream from `ArenaLRU` and from the simulator's `Store`
  (Mac only: the reference imports `simulate_policies` lazily). `expect` writes the reference hashes next to the trace's
  sha256; the host refuses to run a cell whose own hash differs.
* `run_cell()`: the timed replay through an arena object. `TorchArena` runs on a GPU, or on the CPU with an event shim
  (`rehearsal_cpu=True`, never exposed on the command line): its numbers are never a CUDA measurement.

Stage 1 claims exposed demand-fetch copy time only (no compute stand-in, no prefetch, no overlap); see the record.

Usage (Mac, no GPU):
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/cuda/replay_arena.py expect \
        --trace benchmarks/results/routing-trace-deepseek-long/trace.npz --out benchmarks/cuda/n2s1-expected.json
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/replay_arena.py plan
GPU host: `run --trace T --expected benchmarks/cuda/n2s1-expected.json --cells smoke|full [--execute]`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import manifest as mf  # noqa: E402
import suite  # noqa: E402

PHASE_PREFILL, PHASE_DECODE = 0, 1            # the trace's phase ids (src/cachalot/metrics/routing_trace.py)
RECORD_BYTES = 9_950_000                       # DeepSeek expert record (plan section 3)
SOURCE_RECORDS = 64                            # emulated host tier: distinct pinned records, indexed modulo this
TOTAL_EXPERTS = 40 * 384
DEFAULT_LAYERS = 40
READBACK_EVERY = 2000                          # gate (b): verify an arena slot against its source every N copies
GEN_PREFILL_TOKENS = 200                       # the generator cell's prompt; the record left this open (stated in the results)
HARD_CELL_SECONDS = 900


# ------------------------------------------------------------------------------------------------ the walk

def token_ordinals(layers: np.ndarray) -> np.ndarray:
    """0, 1, 2 ... per decode token for records in generation order: a token ends when the layer stops ascending."""
    if layers.size == 0:
        return np.zeros(0, np.int64)
    new = np.concatenate([[True], layers[1:] <= layers[:-1]])
    return np.cumsum(new) - 1


def load_arrays(path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in ("phase", "layer", "position", "experts")}


def trace_layer_count(arrays) -> int:
    return int(np.unique(arrays["layer"]).size)


def walk(arrays, max_decode_tokens: int | None = None):
    """Yield ('P', layer, ordered_keys) per prefill layer and ('D', token, layer, experts) per decode row, in trace order.

    Same sequence `cache_sim.simulate` replays with no segment marks: split at every prefill/decode change; a prefill
    segment admits layer by layer in ascending layer order, experts by descending count then first appearance.
    """
    phase, layer, experts = arrays["phase"], arrays["layer"], arrays["experts"]
    drows = np.nonzero(phase == PHASE_DECODE)[0]
    token_of_row = np.full(len(layer), -1, dtype=np.int64)
    token_of_row[drows] = token_ordinals(layer[drows])
    change = np.nonzero(np.diff(phase.astype(np.int8)) != 0)[0] + 1
    bounds = [0, *change.tolist(), len(layer)]
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if b <= a:
            continue
        if phase[a] == PHASE_PREFILL:
            for la in np.unique(layer[a:b]).tolist():
                flat = experts[a:b][layer[a:b] == la].ravel().tolist()
                count: dict[int, int] = {}
                first: dict[int, int] = {}
                for i, e in enumerate(flat):
                    count[e] = count.get(e, 0) + 1
                    first.setdefault(e, i)
                order = sorted(count, key=lambda e: (-count[e], first[e]))
                yield ("P", int(la), [(int(la), int(e)) for e in order])
            continue
        for i in range(a, b):
            tok = int(token_of_row[i])
            if max_decode_tokens is not None and tok >= max_decode_tokens:
                return
            yield ("D", tok, int(layer[i]), experts[i].tolist())


# ------------------------------------------------------------------------------------------------ bookkeeping

class ArenaLRU:
    """LRU with the store's prefill quota rule, assigning arena slots. Written from the specification (docs and the store's
    contract), not from `Store`'s code, so that equality with the simulator is a test. Keys are (layer, expert)."""

    def __init__(self, slots: int, n_layers: int):
        if slots < 1 or n_layers < 1:
            raise ValueError("slots and n_layers must be positive")
        self.slots, self.n_layers = slots, n_layers
        self.lru: OrderedDict[tuple[int, int], int] = OrderedDict()     # key -> slot, least recently used first
        self.free = list(range(slots - 1, -1, -1))

    def _evict(self, avoid_layer: int | None = None):
        victim = None
        for k in self.lru:
            if avoid_layer is None or k[0] != avoid_layer:
                victim = k
                break
        if victim is None:
            victim = next(iter(self.lru))
        slot = self.lru.pop(victim)
        self.free.append(slot)
        return victim, slot

    def prefill_layer(self, layer: int, ordered_keys: list[tuple[int, int]]):
        """Returns (hits, misses, ops, sets) with ops = ((0|1, layer, expert), ...) (0 delete, 1 insert) and sets = [(slot, key)]."""
        quota = self.slots // self.n_layers + (1 if layer < self.slots % self.n_layers else 0)
        lru = self.lru
        retained = [k for k in ordered_keys if k in lru][:quota]
        retained_set = set(retained)
        ops: list[tuple[int, int, int]] = []
        sets: list[tuple[int, tuple[int, int]]] = []
        for k in [k for k in lru if k[0] == layer and k not in retained_set]:
            self.free.append(lru.pop(k))
            ops.append((0, k[0], k[1]))
        admit: list[tuple[int, int]] = []
        for k in ordered_keys:
            if len(retained) + len(admit) >= quota:
                break
            if k in retained_set or k in admit:
                continue
            if k not in lru:
                admit.append(k)
        admit_set = set(admit)
        hits = misses = 0
        for k in ordered_keys:
            if k in lru:
                hits += 1
                continue
            misses += 1
            if k in admit_set:
                while len(lru) >= self.slots:
                    victim, _ = self._evict(avoid_layer=layer)
                    ops.append((0, victim[0], victim[1]))
                slot = self.free.pop()
                lru[k] = slot
                ops.append((1, k[0], k[1]))
                sets.append((slot, k))
        return hits, misses, tuple(ops), sets

    def decode_request(self, key: tuple[int, int]):
        """Returns (hit, slot, ops): a hit refreshes recency; a miss evicts from the LRU end until a slot is free."""
        lru = self.lru
        if key in lru:
            lru.move_to_end(key)
            return True, lru[key], ()
        ops = []
        while len(lru) >= self.slots:
            victim, _ = self._evict()
            ops.append((0, victim[0], victim[1]))
        slot = self.free.pop()
        lru[key] = slot
        ops.append((1, key[0], key[1]))
        return False, slot, tuple(ops)

    def check_invariants(self) -> None:
        slots = list(self.lru.values())
        assert len(set(slots)) == len(slots), "two keys share a slot"
        assert all(0 <= s < self.slots for s in slots), "slot out of range"
        assert len(slots) + len(self.free) == self.slots, "slots lost or duplicated"


def _digest_feed(h, event) -> None:
    h.update(repr(event).encode())


def bookkeeping_pass(arrays, slots: int, n_layers: int, max_decode_tokens: int | None = None) -> dict:
    """ArenaLRU over the walk: event hash, miss counts, and a rolling chain of (token, layer, expert, slot) of every decode miss."""
    st = ArenaLRU(slots, n_layers)
    h = hashlib.sha256()
    misses = requests = 0
    chain = 0
    per_token: dict[int, int] = {}
    for item in walk(arrays, max_decode_tokens):
        if item[0] == "P":
            hits, miss, ops, _sets = st.prefill_layer(item[1], item[2])
            _digest_feed(h, ("P", item[1], hits, miss, ops))
            continue
        _, tok, la, experts = item
        per_token.setdefault(tok, 0)
        for e in experts:
            hit, slot, ops = st.decode_request((la, e))
            requests += 1
            _digest_feed(h, ("D", tok, la, e, hit, ops))
            if not hit:
                misses += 1
                per_token[tok] += 1
                chain = (chain * 1000003 + ((tok * 64 + la) * 1024 + e) * 8192 + slot) & ((1 << 61) - 1)
    st.check_invariants()
    return {"hash": h.hexdigest(), "chain": chain, "misses": misses, "requests": requests,
            "tokens": len(per_token), "misses_per_token": [per_token[k] for k in sorted(per_token)]}


def reference_pass(arrays, slots: int, n_layers: int, max_decode_tokens: int | None = None) -> dict:
    """The simulator's `Store` over the same walk, logged into the same event format. Mac only (needs benchmarks/)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from simulate_policies import (
        Store,  # noqa: PLC0415  (lazy: the GPU host has no benchmarks/ beyond cuda/)
    )

    log: list[tuple[int, int, int]] = []

    class _Logged(OrderedDict):
        def __delitem__(self, key):
            log.append((0, key[0], key[1]))
            super().__delitem__(key)

        def pop(self, key, *default):
            if key in self:
                log.append((0, key[0], key[1]))
            return super().pop(key, *default)

        def popitem(self, last=True):
            key, val = super().popitem(last)
            log.append((0, key[0], key[1]))
            return key, val

        def __setitem__(self, key, value):
            if key not in self:
                log.append((1, key[0], key[1]))
            super().__setitem__(key, value)

    st = Store(slots, "lru", n_layers=n_layers)
    st.items = _Logged()
    h = hashlib.sha256()
    misses = requests = 0
    per_token: dict[int, int] = {}
    for item in walk(arrays, max_decode_tokens):
        if item[0] == "P":
            log.clear()
            h0, m0 = st.hits, st.misses
            st.prefill_layer(item[1], item[2])
            _digest_feed(h, ("P", item[1], st.hits - h0, st.misses - m0, tuple(log)))
            continue
        _, tok, la, experts = item
        per_token.setdefault(tok, 0)
        for e in experts:
            key = (la, e)
            hit = st._resident(key)
            log.clear()
            st.decode_request(key)
            requests += 1
            _digest_feed(h, ("D", tok, la, e, hit, tuple(log)))
            if not hit:
                misses += 1
                per_token[tok] += 1
    return {"hash": h.hexdigest(), "misses": misses, "requests": requests, "tokens": len(per_token),
            "misses_per_token": [per_token[k] for k in sorted(per_token)]}


# ------------------------------------------------------------------------------------------------ cells

def slots_for(percent: float) -> int:
    return int(round(TOTAL_EXPERTS * percent / 100))


def cell_defs(profile: str) -> list[dict]:
    """The registered cells (record section 3, item 5). `smoke` is a short GPU smoke of the stream code on the same arrays."""
    if profile == "smoke":
        return [
            {"id": "trace.p17.t40", "source": "trace", "percent": 17, "tokens": 40},
            {"id": "gen.p17.t40", "source": "gen", "percent": 17, "tokens": 40},
        ]
    if profile == "full":
        return [{"id": f"trace.p{p}", "source": "trace", "percent": p, "tokens": None} for p in (8, 12, 17)] + [
            {"id": "gen.p17", "source": "gen", "percent": 17, "tokens": None}]
    raise ValueError(f"unknown profile {profile!r}")


def generator_arrays(decode_tokens: int = 1972):
    import routing_gen as rg

    spec = rg.RoutingSpec(layers=40, experts=384, topk=6, tokens=GEN_PREFILL_TOKENS + decode_tokens, skew=1.0, locality=0.3)
    return rg.to_arrays(spec, prefill_tokens=GEN_PREFILL_TOKENS)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cell_arrays(cell: dict, trace_arrays):
    return trace_arrays if cell["source"] == "trace" else generator_arrays()


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = a + b x. Returns (a, b, r_squared); r_squared is 0.0 when x has no spread."""
    n = len(xs)
    if n < 2 or n != len(ys):
        raise ValueError("need at least two paired points")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    return a, b, (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def source_index(key: tuple[int, int]) -> int:
    return (key[0] * 131 + key[1]) % SOURCE_RECORDS


# ------------------------------------------------------------------------------------------------ the arena (torch)

class _CpuEvent:
    def record(self, stream=None):
        self.t = time.perf_counter()

    def synchronize(self):
        pass

    def elapsed_time(self, other) -> float:
        return (other.t - self.t) * 1e3


class TorchArena:
    """One device buffer of `slots x nbytes` bytes allocated once, `SOURCE_RECORDS` distinct pinned source records, one copy
    stream, one reused event pair. Imports torch lazily; `rehearsal_cpu` runs the same code on the CPU for the Mac tests."""

    def __init__(self, slots: int, nbytes: int, seed: int = 20261018, device_index: int = 0, rehearsal_cpu: bool = False):
        import torch

        self.t, self.slots, self.nbytes, self.rehearsal = torch, slots, nbytes, rehearsal_cpu
        self.dev_name = "cpu" if rehearsal_cpu else torch.device("cuda", device_index)
        if not rehearsal_cpu:
            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() is False")
            torch.cuda.set_device(self.dev_name)
        self.arena = torch.empty(slots * nbytes, dtype=torch.uint8, device=self.dev_name)
        gen = torch.Generator().manual_seed(seed)
        self.sources = []
        for _ in range(SOURCE_RECORDS):
            rec = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=gen)
            self.sources.append(rec if rehearsal_cpu else rec.pin_memory())
        self.stream = None if rehearsal_cpu else torch.cuda.Stream(device=self.dev_name)
        self.ev_a, self.ev_b = (_CpuEvent(), _CpuEvent()) if rehearsal_cpu else (
            torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        self.copies = 0

    def _slot(self, slot: int):
        return self.arena[slot * self.nbytes:(slot + 1) * self.nbytes]

    def sync(self) -> None:
        if not self.rehearsal:
            self.t.cuda.synchronize()

    def populate(self, sets: list[tuple[int, tuple[int, int]]]) -> None:
        """Untimed initial fill (prefill admissions): real copies, then a synchronize."""
        for slot, key in sets:
            self._slot(slot).copy_(self.sources[source_index(key)], non_blocking=not self.rehearsal)
            self.copies += 1
        self.sync()

    def fetch(self, misses: list[tuple[int, tuple[int, int]]]) -> tuple[float, float]:
        """Issue the misses of one layer as queued copies on the copy stream, record events, wait. Returns (host ms, event ms)."""
        import contextlib

        ctx = contextlib.nullcontext() if self.stream is None else self.t.cuda.stream(self.stream)
        h0 = time.perf_counter_ns()
        with ctx:
            self.ev_a.record(self.stream)
            for slot, key in misses:
                self._slot(slot).copy_(self.sources[source_index(key)], non_blocking=not self.rehearsal)
            self.ev_b.record(self.stream)
        self.ev_b.synchronize()
        host_ms = (time.perf_counter_ns() - h0) / 1e6
        self.copies += len(misses)
        return host_ms, self.ev_a.elapsed_time(self.ev_b)

    def verify(self, slot: int, key: tuple[int, int]) -> bool:
        self.sync()
        return bool(self.t.equal(self._slot(slot).cpu(), self.sources[source_index(key)]))

    def memory(self) -> dict:
        if self.rehearsal:
            return {"arena_bytes": self.slots * self.nbytes}
        t = self.t.cuda
        return {"arena_bytes": self.slots * self.nbytes, "reserved_bytes": int(t.memory_reserved(self.dev_name)),
                "allocated_bytes": int(t.memory_allocated(self.dev_name))}

    def qualify_sources(self, iters: int = 15) -> list[float]:
        """EXPLORATORY (not a registered cell): the isolated copy time of every source record into slot 0, event ms, p50 each.
        S1, S2 and the N2 smoke run showed that the H2D rate of a 9.95 MB pinned buffer depends on the buffer (about 0.18 ms or
        about 0.35 ms, stable per buffer), so the replay's cost per miss is a mixture; this table decomposes it."""
        import contextlib

        out = []
        for rec in self.sources:
            ts = []
            for i in range(iters + 3):
                ctx = contextlib.nullcontext() if self.stream is None else self.t.cuda.stream(self.stream)
                with ctx:                       # the copy must run on the stream the events are recorded on
                    self.ev_a.record(self.stream)
                    self._slot(0).copy_(rec, non_blocking=not self.rehearsal)
                    self.ev_b.record(self.stream)
                self.ev_b.synchronize()
                if i >= 3:
                    ts.append(self.ev_a.elapsed_time(self.ev_b))
            out.append(sorted(ts)[len(ts) // 2])
        return out

    def control(self, iters: int = 30) -> list[float]:
        """Isolated copies of one record into slot 0, host ms each (the start and end drift bracket)."""
        out = []
        for _ in range(iters + 3):
            h0 = time.perf_counter_ns()
            self._slot(0).copy_(self.sources[0], non_blocking=not self.rehearsal)
            self.sync()
            out.append((time.perf_counter_ns() - h0) / 1e6)
        return out[3:]


def run_cell(arrays, slots: int, n_layers: int, arena, *, max_decode_tokens: int | None = None,
             readback_every: int = READBACK_EVERY, deadline: suite.Deadline | None = None, seed: int = 0) -> dict:
    """The timed replay. Prefill admissions populate the arena untimed; every decode layer's misses are fetched and timed.
    Gate (b) verifies arena slots against their sources during the run and over a final sample of resident slots."""
    st = ArenaLRU(slots, n_layers)
    layer_rows: list[tuple[int, int, int, float, float]] = []       # token, layer, k, host_ms, event_ms
    book_ns: dict[int, int] = {}
    per_token_miss: dict[int, int] = {}
    chain = 0
    misses = requests = 0
    readbacks = readback_fail = 0
    truncated = False
    next_check = readback_every
    prefill_copies = 0
    for item in walk(arrays, max_decode_tokens):
        if item[0] == "P":
            _h, _m, _ops, sets = st.prefill_layer(item[1], item[2])
            arena.populate(sets)
            prefill_copies += len(sets)
            continue
        _, tok, la, experts = item
        if deadline is not None and tok not in per_token_miss and deadline.expired():
            truncated = True
            break
        per_token_miss.setdefault(tok, 0)
        b0 = time.perf_counter_ns()
        miss_list: list[tuple[int, tuple[int, int]]] = []
        for e in experts:
            key = (la, e)
            hit, slot, _ops = st.decode_request(key)
            requests += 1
            if not hit:
                miss_list.append((slot, key))
                chain = (chain * 1000003 + ((tok * 64 + la) * 1024 + e) * 8192 + slot) & ((1 << 61) - 1)
        book_ns[tok] = book_ns.get(tok, 0) + time.perf_counter_ns() - b0
        k = len(miss_list)
        if k:
            host_ms, ev_ms = arena.fetch(miss_list)
            misses += k
            per_token_miss[tok] += k
            if arena.copies >= next_check:                    # outside the timed span, after the layer's event
                next_check += readback_every
                slot, key = miss_list[-1]
                readbacks += 1
                if not arena.verify(slot, key):
                    readback_fail += 1
        else:
            host_ms = ev_ms = 0.0
        layer_rows.append((tok, la, k, host_ms, ev_ms))
    # final gate: a seeded sample of resident slots against their sources
    rng = np.random.default_rng(seed)
    resident = list(st.lru.items())
    sample = [resident[i] for i in rng.choice(len(resident), size=min(48, len(resident)), replace=False)] if resident else []
    for key, slot in sample:
        readbacks += 1
        if not arena.verify(slot, key):
            readback_fail += 1
    st.check_invariants()
    toks = sorted(per_token_miss)
    tok_ms = {t: 0.0 for t in toks}
    tok_ev = {t: 0.0 for t in toks}
    for t, _la, _k, hm, em in layer_rows:
        tok_ms[t] += hm
        tok_ev[t] += em
    return {
        "tokens": len(toks), "requests": requests, "misses": misses, "prefill_copies": prefill_copies, "chain": chain,
        "truncated": truncated, "readbacks": readbacks, "readback_failures": readback_fail,
        "token_misses": [per_token_miss[t] for t in toks], "token_host_ms": [tok_ms[t] for t in toks],
        "token_event_ms": [tok_ev[t] for t in toks], "token_book_ms": [book_ns.get(t, 0) / 1e6 for t in toks],
        "layer_rows": layer_rows, "memory": arena.memory(), "nbytes": arena.nbytes,
    }


def per_copy_by_event_size(layer_rows) -> dict:
    """Mean host ms per copy over layer events with 1 miss and over events with 5 or more (N2-P4)."""
    one = [hm / k for _t, _l, k, hm, _e in layer_rows if k == 1]
    many = [hm / k for _t, _l, k, hm, _e in layer_rows if k >= 5]
    return {"k1_mean_ms": (sum(one) / len(one)) if one else None, "k1_events": len(one),
            "k5plus_mean_ms": (sum(many) / len(many)) if many else None, "k5plus_events": len(many)}


def cell_rows(run_id: str, cell: dict, res: dict, hash_ok: bool, control_start: list[float], control_end: list[float],
              timer_overhead_ms: float) -> list[dict]:
    arm = f"n2s1.{cell['id']}"
    src = "host_monotonic"
    rows: list[dict] = []
    nb = res["nbytes"]
    s = suite.summarize(res["token_host_ms"], seed=1) if res["token_host_ms"] else None
    if s:
        rows.append(mf.row(run_id, arm, "exposed_copy_ms_token", s["p50"], "ms", "measured", source=src, n=s["n"],
                           p50=s["p50"], p95=s["p95"], ci_low=s.get("ci_low"), ci_high=s.get("ci_high"),
                           ci_method=s.get("ci_method"), overhead_ms=timer_overhead_ms,
                           note="N2 stage 1; sum over a token's layers of host time from first copy issue to the layer's event"))
        ev = suite.summarize(res["token_event_ms"], seed=1)
        rows.append(mf.row(run_id, arm, "exposed_copy_event_ms_token", ev["p50"], "ms", "measured", source="cuda_events", n=ev["n"],
                           p50=ev["p50"], p95=ev["p95"]))
        bk = suite.summarize(res["token_book_ms"], seed=1)
        rows.append(mf.row(run_id, arm, "bookkeeping_ms_token", bk["p50"], "ms", "measured", source=src, n=bk["n"],
                           p50=bk["p50"], p95=bk["p95"], note="miss decisions and issue list, no copies, no hashing"))
        if len(set(res["token_misses"])) > 1:
            a, b, r2 = fit_line([float(x) for x in res["token_misses"]], res["token_host_ms"])
            rows.append(mf.row(run_id, arm, "token_ms_per_extra_miss", b, "ms", "derived", source="least squares, token time on misses",
                               n=len(res["token_misses"]), note=f"intercept {a:.3f} ms, R-squared {r2:.4f}"))
            rows.append(mf.row(run_id, arm, "token_fit_r2", r2, "fraction", "derived", source="least squares"))
        rows.append(mf.row(run_id, arm, "misses_per_token", sum(res["token_misses"]) / len(res["token_misses"]), "count",
                           "measured", source="replay", n=len(res["token_misses"])))
        rows.append(mf.row(run_id, arm, "bytes_h2d_per_token", nb * sum(res["token_misses"]) / len(res["token_misses"]), "bytes",
                           "derived", source="misses x record bytes"))
    pc = per_copy_by_event_size(res["layer_rows"])
    for key, label in (("k1_mean_ms", "per_copy_ms_events_of_1"), ("k5plus_mean_ms", "per_copy_ms_events_of_5_plus")):
        if pc[key] is None:
            rows.append(mf.row(run_id, arm, label, None, "ms", "unavailable", source=src, note="no events of that size in the cell"))
        else:
            rows.append(mf.row(run_id, arm, label, pc[key], "ms", "measured", source=src,
                               n=pc["k1_events" if key == "k1_mean_ms" else "k5plus_events"]))
    mem = res["memory"]
    if "reserved_bytes" in mem:
        rows.append(mf.row(run_id, arm, "reserved_beyond_arena_mib", (mem["reserved_bytes"] - mem["arena_bytes"]) / 2**20, "MiB",
                           "measured", source="torch.cuda.memory_reserved - arena"))
    if control_start and control_end:
        c0, c1 = sorted(control_start)[len(control_start) // 2], sorted(control_end)[len(control_end) // 2]
        rows.append(mf.row(run_id, arm, "control_drift", abs(c1 / c0 - 1.0), "fraction", "derived",
                           source="isolated one-record copy p50 at the end / at the start - 1"))
    q = res.get("source_ms")
    if q:
        qs = sorted(q)
        rows.append(mf.row(run_id, arm, "source_copy_ms_min", qs[0], "ms", "measured", source="exploratory source qualification, cuda events"))
        rows.append(mf.row(run_id, arm, "source_copy_ms_median", qs[len(qs) // 2], "ms", "measured", source="exploratory source qualification, cuda events"))
        rows.append(mf.row(run_id, arm, "source_copy_ms_max", qs[-1], "ms", "measured", source="exploratory source qualification, cuda events"))
        rows.append(mf.row(run_id, arm, "source_records_slow_share", sum(1 for x in qs if x >= 1.5 * qs[0]) / len(qs), "fraction",
                           "derived", source="records at or above 1.5 x the fastest record's copy time"))
    rows.append(mf.row(run_id, arm, "event_hash_equals_expected", 1 if hash_ok else 0, "bool", "measured", source="gate (a)"))
    rows.append(mf.row(run_id, arm, "readback_failures", res["readback_failures"], "count", "measured", source="gate (b)",
                       n=res["readbacks"]))
    return rows


# ------------------------------------------------------------------------------------------------ CLI

def _trace_cells(profile: str) -> list[dict]:
    return cell_defs(profile)


def cmd_plan(args) -> int:
    for prof in ("smoke", "full"):
        for c in cell_defs(prof):
            slots = slots_for(c["percent"])
            print(f"{prof:5s} {c['id']:16s} source={c['source']:5s} slots={slots:5d} arena={slots * RECORD_BYTES / 2**30:5.1f} GiB "
                  f"tokens={c['tokens'] or 'all'}")
    return 0


def cmd_expect(args) -> int:
    trace = load_arrays(args.trace)
    out = {"trace_sha256": file_sha256(args.trace), "record_bytes": RECORD_BYTES, "n_layers": trace_layer_count(trace),
           "gen_prefill_tokens": GEN_PREFILL_TOKENS, "cells": {}}
    gen = None
    for prof in ("smoke", "full"):
        for c in cell_defs(prof):
            if c["id"] in out["cells"]:
                continue
            if c["source"] == "gen" and gen is None:
                gen = generator_arrays()
            arrays = trace if c["source"] == "trace" else gen
            slots, nl = slots_for(c["percent"]), (trace_layer_count(trace) if c["source"] == "trace" else DEFAULT_LAYERS)
            ref = reference_pass(arrays, slots, nl, c["tokens"])
            mine = bookkeeping_pass(arrays, slots, nl, c["tokens"])
            ok = ref["hash"] == mine["hash"] and ref["misses_per_token"] == mine["misses_per_token"]
            out["cells"][c["id"]] = {"slots": slots, "n_layers": nl, "hash": ref["hash"], "misses": ref["misses"],
                                     "tokens": ref["tokens"], "arena_equals_simulator": ok}
            print(f"{c['id']:16s} slots={slots:5d} misses/token={ref['misses'] / max(ref['tokens'], 1):7.2f} "
                  f"hash {ref['hash'][:12]} equal_to_simulator={ok}")
            if not ok:
                print("FAIL: ArenaLRU differs from the simulator: nothing downstream counts", file=sys.stderr)
                return 4
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


def cmd_run(args) -> int:
    cells = cell_defs(args.cells)
    if not args.execute:
        cmd_plan(args)
        print("\n# DRY RUN: nothing executed. Add --execute on the provisioned GPU host.", file=sys.stderr)
        return 0
    import doctor

    expected = json.loads(Path(args.expected).read_text())
    if file_sha256(args.trace) != expected["trace_sha256"]:
        print("STOP: the trace file differs from the one the expected hashes were made from", file=sys.stderr)
        return 5
    report = doctor.collect(note=args.note, region=args.region, session_cap_s=int(args.max_seconds))
    ok, why = doctor.cuda_ready(report)
    if not ok:
        print(f"SKIP: no CUDA run possible here: {why}", file=sys.stderr)
        return 3
    trace = load_arrays(args.trace)
    gen = None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{stamp}-n2s1-{args.cells}-{report['manifest_id'][:6]}"
    out = Path(args.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    mf.write_json(out / "machine-manifest.json", report)
    dl = suite.Deadline(args.max_seconds)
    rows: list[dict] = []
    gates = {"event_hash": True, "determinism": True, "readback": True, "sanity": True}
    failures: list[dict] = []
    raw: dict = {}
    status = "complete"
    for c in cells:
        if dl.expired():
            status = "truncated"
            break
        try:
            if c["source"] == "gen" and gen is None:
                gen = generator_arrays()
            arrays = trace if c["source"] == "trace" else gen
            slots = slots_for(c["percent"])
            nl = trace_layer_count(trace) if c["source"] == "trace" else DEFAULT_LAYERS
            exp = expected["cells"][c["id"]]
            first = bookkeeping_pass(arrays, slots, nl, c["tokens"])
            hash_ok = first["hash"] == exp["hash"]
            gates["event_hash"] &= hash_ok
            if not hash_ok:
                failures.append({"cell": c["id"], "error": "event hash differs from the Mac simulator's"})
                continue
            arena = TorchArena(slots, RECORD_BYTES)
            cap = suite.Deadline(min(HARD_CELL_SECONDS, max(1.0, dl.remaining())))
            c0 = arena.control()
            q0 = arena.qualify_sources()
            res = run_cell(arrays, slots, nl, arena, max_decode_tokens=c["tokens"], deadline=cap)
            res["source_ms"] = q0
            c1 = arena.control()
            again = bookkeeping_pass(arrays, slots, nl, c["tokens"])
            same = again["hash"] == first["hash"] and (res["truncated"] or res["chain"] == first["chain"])
            gates["determinism"] &= same
            gates["readback"] &= res["readback_failures"] == 0
            if res["truncated"]:
                status = "truncated"
            rows += cell_rows(run_id, c, res, hash_ok, c0, c1, 0.0)
            for r in rows[-12:]:
                if r["field"] == "exposed_copy_ms_token" and r.get("value") is not None:
                    bw = RECORD_BYTES * (sum(res["token_misses"]) / max(len(res["token_misses"]), 1)) / 1e9 / (r["value"] / 1e3)
                    if bw > suite.PCIE_SANITY_GB_S:
                        gates["sanity"] = False
            raw[c["id"]] = {k: res[k] for k in ("token_misses", "token_host_ms", "token_event_ms", "token_book_ms")}
            raw[c["id"]]["source_ms"] = q0
            del arena
            import torch

            torch.cuda.empty_cache()                 # release the cached arena so the next cell's reserved-beyond-arena is its own
        except Exception as exc:                                    # a failed cell is recorded, never hidden
            failures.append({"cell": c["id"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
    manifest = {
        "schema": mf.SCHEMA_NAME, "schema_version": mf.SCHEMA_VERSION, "platform": mf.PLATFORM, "kind": "run",
        "created_utc": mf.utc_now(), "machine_manifest_id": report["manifest_id"], "run_id": run_id,
        "instrument": {"name": "benchmarks/cuda/replay_arena.py", "version": "N2S1-1"}, "code": report["code"],
        "mode": "timing",
        "workload": {"id": f"N2-stage1-{args.cells}", "class": "N2 stage 1 trace replay with real copies (no model)",
                     "trace_sha256": expected["trace_sha256"], "record_bytes": RECORD_BYTES, "source_records": SOURCE_RECORDS,
                     "policy": "lru", "claims": "exposed demand-fetch copy time a token only; no compute, prefetch or overlap"},
        "protocol": {"ordering": "one pass per cell, isolated-copy control at the start and end of each cell"},
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
    e = sub.add_parser("expect", help="Mac only: reference event hashes from the simulator's Store")
    e.add_argument("--trace", required=True)
    e.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--trace", required=True)
    r.add_argument("--expected", required=True)
    r.add_argument("--cells", choices=("smoke", "full"), required=True)
    r.add_argument("--execute", action="store_true")
    r.add_argument("--max-seconds", type=float, default=900.0)
    r.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "cuda"))
    r.add_argument("--note", default="")
    r.add_argument("--region")
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "expect": cmd_expect, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
