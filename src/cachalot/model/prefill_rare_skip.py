"""Rare-expert skip for DeepSeek prefill (opt-in approximation, default off; docs/prefill-rare-skip/RECORD.md).

A follow-up prefill of a few thousand tokens is bound by expert reads: every routed expert that is not resident is read
once per chunk and layer (HANDOFF 18.89). In the 1017 Hermes routing traces about a fifth to a quarter of those reads are
experts that only one row of the chunk routes to, and those experts carry 0.2-0.5 % of the layer's routing mass in chunks
of 1,000 rows or more (HANDOFF 18.259). `CACHALOT_PREFILL_RARE_SKIP=k` drops, in a prefill call of at least
`CACHALOT_PREFILL_RARE_MIN_ROWS` rows (default 1,024), every routed expert that at most `k` rows route to, and rescales
each affected row's remaining router weights to the row's original total, as the decode miss budget does.

The rule does not look at residency, so a chunk's result does not depend on what the cache happened to hold: a disk
snapshot taken with the knob on equals a cold prefill with the same setting, and snapshots are keyed on it
(`snapshot_store.numerics_version`). It changes outputs and is gated by the quality harness before any default.
"""
from __future__ import annotations

import os
from collections import Counter

DEFAULT_MIN_ROWS = 1024

# Counters since start (or the last reset): calls the rule applied to, experts skipped, routed (row, slot) pairs dropped,
# and the router weight those pairs held over the weight of every pair in the calls it applied to.
STATS: Counter = Counter()


def log_field() -> str:
    """Cumulative counters for the server's [request] line; empty until the rule has applied once."""
    if not STATS["calls"]:
        return ""
    return f" rare_skip={STATS['skipped_experts']}/{100 * STATS['dropped_mass'] / max(STATS['mass'], 1e-12):.2f}%"


def rare_skip() -> tuple[int, int] | None:
    """(k, min_rows) from the environment, or None when off (unset, empty, 0, off)."""
    raw = os.environ.get("CACHALOT_PREFILL_RARE_SKIP", "").strip().lower()
    if raw in ("", "0", "off", "exact"):
        return None
    k = int(raw)
    if k <= 0:
        return None
    min_rows = int(os.environ.get("CACHALOT_PREFILL_RARE_MIN_ROWS", str(DEFAULT_MIN_ROWS)))
    return k, max(1, min_rows)


def snapshot_tag() -> str:
    """Suffix for the snapshot key: empty when off."""
    cfg = rare_skip()
    return "" if cfg is None else f"-rare{cfg[0]}m{cfg[1]}"


def plan_rare_skip(
    assignments: dict[int, list[tuple[int, int, float]]],
    n_tokens: int,
    k: int,
    min_rows: int,
) -> tuple[dict[int, list[tuple[int, int, float]]], list[int]]:
    """Drop experts routed by at most `k` rows and rescale the affected rows' kept weights.

    `assignments` maps expert id -> [(token_index, route_slot, router_weight), ...] in the order
    `moe_prefill_grouped` built it. Returns the new mapping (same order, skipped experts removed) and the skipped ids.
    Below `min_rows` rows the mapping is returned unchanged."""
    if n_tokens < min_rows or k <= 0:
        return assignments, []
    skipped = [e for e, rows in assignments.items() if len(rows) <= k]
    if not skipped:
        return assignments, []
    skip_set = set(skipped)
    total: dict[int, float] = {}
    kept: dict[int, float] = {}
    affected: set[int] = set()
    for e, rows in assignments.items():
        for t, _, w in rows:
            total[t] = total.get(t, 0.0) + w
            if e in skip_set:
                affected.add(t)
            else:
                kept[t] = kept.get(t, 0.0) + w
    scale = {t: total[t] / (kept.get(t, 0.0) or 1.0) for t in affected}
    out: dict[int, list[tuple[int, int, float]]] = {}
    dropped_pairs = 0
    dropped_mass = 0.0
    for e, rows in assignments.items():
        if e in skip_set:
            dropped_pairs += len(rows)
            dropped_mass += sum(w for _, _, w in rows)
            continue
        out[e] = [(t, s, w * scale[t]) if t in scale else (t, s, w) for t, s, w in rows]
    STATS["calls"] += 1
    STATS["skipped_experts"] += len(skipped)
    STATS["dropped_pairs"] += dropped_pairs
    STATS["dropped_mass"] += dropped_mass
    STATS["mass"] += sum(total.values())
    return out, skipped
