"""
Why does the SLRU store model collapse at small resident fractions? (docs/CUDA-SLRU-COLLAPSE-RECORD.md)

Offline, Mac only. Replays routing traces through `simulate_policies.Store` with an instrumented SLRU subclass and a sweep
of the protected fraction, and tests the hypothesis that the collapse is probation starvation: the probation segment,
(1 - f) x slots, is smaller than the keys a decode token inserts.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/slru_collapse_probe.py --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cuda_offline_policy_table as t  # noqa: E402
from cachalot.metrics.routing_trace import load_trace  # noqa: E402
from simulate_policies import Store  # noqa: E402

MINIMAX = Path(__file__).resolve().parent / "results" / "routing-trace-minimax-long" / "trace.npz"
DEEPSEEK = Path(__file__).resolve().parent / "results" / "routing-trace-deepseek-long" / "trace.npz"
COLLAPSE = 1.10


class ProbeStore(Store):
    """SLRU with counters: where hits are served, promotions, and the life of every decode-inserted key."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.req = 0
        self.birth: dict[tuple[int, int], int] = {}
        self.rehit: set[tuple[int, int]] = set()
        self.lifetimes: list[int] = []          # requests between insertion and eviction, decode-inserted keys
        self.evicted_never_rehit = 0
        self.evicted_inserted = 0
        self.hit_from = {"protected": 0, "probation": 0}
        self.promotions = 0
        self.series: list[tuple[int, int, int, int]] = []     # (token index, protected size, probation size, promotions so far)
        self._tok = 0

    def decode_request(self, key):
        self.req += 1
        if key in self.protected:
            self.hit_from["protected"] += 1
            self.rehit.add(key)
        elif key in self.items:
            self.hit_from["probation"] += 1
            self.rehit.add(key)
            self.promotions += 1
        super().decode_request(key)
        if key not in self.birth and key not in self.protected and key in self.items and key not in self.rehit:
            self.birth[key] = self.req

    def _evict_one(self, avoid_layer=None):
        if avoid_layer is None:                  # a decode-time eviction (prefill passes the layer)
            src = self.items if self.items else self.protected
            victim = next(iter(src), None)
            if victim is not None and victim in self.birth:
                self.lifetimes.append(self.req - self.birth.pop(victim))
                self.evicted_inserted += 1
                if victim not in self.rehit:
                    self.evicted_never_rehit += 1
                self.rehit.discard(victim)
        super()._evict_one(avoid_layer)

    def token_done(self):
        self._tok += 1
        self.series.append((self._tok, len(self.protected), len(self.items), self.promotions))


def run_cell(arrays, slots: int, policy: str, f: float | None = None, probe: bool = False):
    n_layers = t.trace_layers(arrays)
    if probe:
        store = ProbeStore(slots, "slru", n_layers=n_layers, protected_fraction=f)
        r = t.replay(arrays, slots, "slru", store=store, on_token=lambda _tok, s: s.token_done())
        return r, store
    store = Store(slots, policy, n_layers=n_layers, **({"protected_fraction": f} if f is not None else {}))
    return t.replay(arrays, slots, policy, store=store), None


def internals(store: ProbeStore, tokens: int, requests_per_token: float, f: float) -> dict:
    cap = int(store.slots * f)
    half = [row for row in store.series if row[0] > tokens // 2]
    late = [row for row in store.series if row[0] > 200]
    promos_late = (late[-1][3] - late[0][3]) / max(len(late) - 1, 1) if len(late) > 1 else float("nan")
    total_hits = sum(store.hit_from.values())
    return {
        "protected_cap": cap,
        "protected_min_frac_of_cap_after_token_200": min((r[1] for r in late), default=0) / cap,
        "protected_min_frac_of_cap_last_half": min((r[1] for r in half), default=0) / cap,
        "median_lifetime_requests": statistics.median(store.lifetimes) if store.lifetimes else None,
        "median_lifetime_tokens": (statistics.median(store.lifetimes) / requests_per_token) if store.lifetimes else None,
        "decode_inserted_evicted": store.evicted_inserted,
        "share_evicted_never_rehit": store.evicted_never_rehit / max(store.evicted_inserted, 1),
        "share_hits_from_protected": store.hit_from["protected"] / max(total_hits, 1),
        "promotions_per_token_after_200": promos_late,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()

    mm, _ = load_trace(str(MINIMAX))
    ds, _ = load_trace(str(DEEPSEEK))
    out: dict = {}

    # S0: the default-preserving change
    lru584, _ = run_cell(mm, 584, "lru")
    s0, _ = run_cell(mm, 584, "slru")
    out["S0"] = {"slru_584_default_misses_per_token": s0["misses"] / s0["tokens"],
                 "lru_584_misses_per_token": lru584["misses"] / lru584["tokens"]}
    print("S0 slru@584 default:", out["S0"], flush=True)

    # S1: internals, MiniMax, 584 slots, f = 0.8
    r, store = run_cell(mm, 584, "slru", 0.8, probe=True)
    req_per_token = r["requests"] / r["tokens"]
    out["S1"] = {"misses_per_token": r["misses"] / r["tokens"], **internals(store, r["tokens"], req_per_token, 0.8)}
    print("S1:", json.dumps(out["S1"], indent=1), flush=True)

    # S2/S3/S4: the sweeps; S5 collects every cell
    cells = []

    def sweep(label, arrays, slots, fracs):
        base, _ = run_cell(arrays, slots, "lru")
        lru_m = base["misses"] / base["tokens"]
        for f in fracs:
            r, _ = run_cell(arrays, slots, "slru", f)
            m = r["misses"] / r["tokens"]
            probation = (1 - f) * slots
            cells.append({"trace": label, "slots": slots, "f": f, "lru": lru_m, "slru": m, "ratio": m / lru_m,
                          "collapse": m > COLLAPSE * lru_m, "probation_slots": probation,
                          "rule_predicts_collapse": probation < lru_m})
            print(f"{label:9} slots {slots:4} f {f:.2f} lru {lru_m:7.2f} slru {m:7.2f} ratio {m / lru_m:5.2f} "
                  f"collapse {cells[-1]['collapse']} (probation {probation:.0f} vs misses {lru_m:.0f})", flush=True)

    sweep("minimax", mm, 584, [0.5, 0.65, 0.75, 0.8, 0.9])
    sweep("minimax", mm, 876, [0.8, 0.9])
    sweep("deepseek", ds, round(15360 * 0.03), [0.5, 0.8])
    out["cells"] = cells
    wrong = [c for c in cells if c["collapse"] != c["rule_predicts_collapse"]]
    out["S5_misclassified"] = [(c["trace"], c["slots"], c["f"]) for c in wrong]
    print("S5 misclassified by the rule (probation < LRU misses):", out["S5_misclassified"])
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
