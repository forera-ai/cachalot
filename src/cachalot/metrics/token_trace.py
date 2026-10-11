"""
Per-token critical-path trace for the DeepSeek decode path (charter L2, HANDOFF 18.114, docs/L2-IN-SITU-PRICE-RECORD.md).

Off by default: the store's `token_trace` attribute is None and every hook site costs one `getattr` and one `is not None`
test. When on, each thread appends (timestamp, code, a, b) to its own preallocated columns (no lock, no allocation on the
hot path); the columns are merged only when `drain()` is called, between requests.

`active` lets a benchmark switch recording per decode token without removing the object (the price benchmark alternates
it by token parity). `begin_token` always stamps the token boundary, active or not: the stamp is the instrument that
measures a token's wall time in both states.
"""
from __future__ import annotations

import threading
from array import array
from time import perf_counter_ns

import numpy as np

TOKEN = 0
SYNC_START = 1
SYNC_END = 2
GET_START = 3
GET_END = 4
READ_START = 5
READ_END = 6

THREAD_CAPACITY = 1 << 16  # events per thread between drains (about 150 tokens of the busiest thread)


class _Columns:
    __slots__ = ("ts", "code", "a", "b", "n", "dropped")

    def __init__(self, capacity: int) -> None:
        self.ts, self.code, self.a, self.b = (array("q", bytes(8 * capacity)) for _ in range(4))
        self.n = 0
        self.dropped = 0


class TokenTrace:
    def __init__(self, capacity: int = THREAD_CAPACITY) -> None:
        self.capacity = capacity
        self.active = True
        # None: record every token. 0 or 1: record only tokens whose position has that parity (a benchmark alternates
        # recording per token to price the tracer inside one process); -1 records nothing (the off-against-off control).
        self.alternate: int | None = None
        self._local = threading.local()
        self._all: list[_Columns] = []
        self._register = threading.Lock()
        self._token_stamps = array("q")  # every token boundary, active or not
        self._token_positions = array("q")

    def _columns(self) -> _Columns:
        cols = getattr(self._local, "cols", None)
        if cols is None:
            cols = _Columns(self.capacity)
            self._local.cols = cols
            with self._register:
                self._all.append(cols)
        return cols

    def emit(self, code: int, a: int = 0, b: int = 0) -> None:
        if not self.active:
            return
        cols = getattr(self._local, "cols", None) or self._columns()
        i = cols.n
        if i < self.capacity:
            cols.ts[i] = perf_counter_ns()
            cols.code[i] = code
            cols.a[i] = a
            cols.b[i] = b
            cols.n = i + 1
        else:
            cols.dropped += 1

    def begin_token(self, position: int) -> None:
        """Stamp a token boundary (always) and record it as an event (when active)."""
        now = perf_counter_ns()
        if self.alternate is not None:
            self.active = position % 2 == self.alternate
        self._token_stamps.append(now)
        self._token_positions.append(position)
        if self.active:
            cols = getattr(self._local, "cols", None) or self._columns()
            i = cols.n
            if i < self.capacity:
                cols.ts[i] = now
                cols.code[i] = TOKEN
                cols.a[i] = position
                cols.b[i] = 0
                cols.n = i + 1
            else:
                cols.dropped += 1

    def drain(self) -> dict:
        """Merge and clear. Returns the raw columns (sorted by time) and the token boundary stamps."""
        with self._register:
            cols = list(self._all)
        ts, code, a, b = [], [], [], []
        dropped = 0
        for c in cols:
            n = c.n
            ts.append(np.frombuffer(c.ts, dtype=np.int64, count=n).copy())
            code.append(np.frombuffer(c.code, dtype=np.int64, count=n).copy())
            a.append(np.frombuffer(c.a, dtype=np.int64, count=n).copy())
            b.append(np.frombuffer(c.b, dtype=np.int64, count=n).copy())
            dropped += c.dropped
            c.n = 0
            c.dropped = 0
        stamps = np.frombuffer(self._token_stamps, dtype=np.int64).copy()
        positions = np.frombuffer(self._token_positions, dtype=np.int64).copy()
        del self._token_stamps[:]
        del self._token_positions[:]
        if ts:
            ts_all, code_all, a_all, b_all = (np.concatenate(x) for x in (ts, code, a, b))
        else:
            ts_all = code_all = a_all = b_all = np.zeros(0, dtype=np.int64)
        order = np.argsort(ts_all, kind="stable")
        return {
            "ts": ts_all[order],
            "code": code_all[order],
            "a": a_all[order],
            "b": b_all[order],
            "token_stamps": stamps,
            "token_positions": positions,
            "dropped": dropped,
        }


def _union(starts: np.ndarray, ends: np.ndarray) -> list[tuple[int, int]]:
    """Merge intervals; inputs are parallel arrays."""
    if len(starts) == 0:
        return []
    order = np.argsort(starts, kind="stable")
    merged: list[list[int]] = []
    for s, e in zip(starts[order].tolist(), ends[order].tolist()):
        if merged and s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Total length of the intersection of two sorted disjoint interval lists."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _paired(ts: np.ndarray, code: np.ndarray, a: np.ndarray, b: np.ndarray, start_code: int, end_code: int, within: tuple[int, int],
            key_threads: bool = False):
    """Pair START and END events of one kind inside [within) by their `a` key (layer) in time order; returns starts, ends."""
    lo, hi = within
    sel = (ts >= lo) & (ts < hi)
    s_mask = sel & (code == start_code)
    e_mask = sel & (code == end_code)
    s_t, s_a = ts[s_mask], a[s_mask]
    e_t, e_a = ts[e_mask], a[e_mask]
    starts, ends = [], []
    pending: dict[int, list[int]] = {}
    # merge by time; a (layer) keys the pairing, which for the main thread is one open interval at a time
    events = sorted([(t, 0, k) for t, k in zip(s_t.tolist(), s_a.tolist())] + [(t, 1, k) for t, k in zip(e_t.tolist(), e_a.tolist())])
    for t, kind, k in events:
        if kind == 0:
            pending.setdefault(k, []).append(t)
        elif pending.get(k):
            starts.append(pending[k].pop(0))
            ends.append(t)
    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def summarize(drained: dict) -> list[dict]:
    """
    One row per decode token (token boundary to the next boundary): wall, sync, blocked (exposed I/O), read busy,
    hidden read, reads, bytes, events. Rows exist only for tokens whose boundary events were recorded (active tokens).
    """
    ts, code, a, b = drained["ts"], drained["code"], drained["a"], drained["b"]
    stamps = drained["token_stamps"]
    positions = drained["token_positions"]
    rows: list[dict] = []
    if len(stamps) < 2:
        return rows
    active_starts = {int(t) for t in ts[code == TOKEN].tolist()}
    for k in range(len(stamps) - 1):
        lo, hi = int(stamps[k]), int(stamps[k + 1])
        if lo not in active_starts:
            continue
        sel = (ts >= lo) & (ts < hi)
        sync_s, sync_e = _paired(ts, code, a, b, SYNC_START, SYNC_END, (lo, hi))
        get_s, get_e = _paired(ts, code, a, b, GET_START, GET_END, (lo, hi))
        # a worker's read can end after the main thread has moved on; pair reads in [lo, hi + tail) by (layer, expert) order
        r_sel = (ts >= lo) & (ts < hi)
        rs_t = ts[r_sel & (code == READ_START)]
        re_t = ts[r_sel & (code == READ_END)]
        rb = b[r_sel & (code == READ_END)]
        n_pair = min(len(rs_t), len(re_t))
        # reads are paired by arrival order per thread-agnostic start/end counts; a read that straddles the boundary is dropped
        read_union = _union(np.sort(rs_t)[:n_pair], np.sort(re_t)[:n_pair]) if n_pair else []
        get_union = _union(get_s, get_e)
        busy = sum(e - s for s, e in read_union)
        blocked = sum(e - s for s, e in get_union)
        hidden = busy - _overlap(read_union, get_union)
        rows.append({
            "position": int(positions[k]),
            "wall_ms": (hi - lo) / 1e6,
            "sync_ms": float((sync_e - sync_s).sum()) / 1e6 if len(sync_s) else 0.0,
            "blocked_ms": blocked / 1e6,
            "read_busy_ms": busy / 1e6,
            "hidden_read_ms": hidden / 1e6,
            "reads": int(n_pair),
            "bytes": int(rb[:n_pair].sum()) if n_pair else 0,
            "events": int(sel.sum()),
        })
    return rows
