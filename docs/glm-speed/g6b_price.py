"""G6b's price: a predicted-set read ahead for short GLM prefill chunks (docs/glm-speed/G6B-RECORD.md).

Usage: g6b_price.py <weighted routing trace .npz with pred_* arrays> [slots, 3489 = 46 GiB] [decode miss budget, 2]

What G6b would do (not built when this was written): a prefill chunk under a threshold scores the next MoE layer's
router on this layer's MoE input for every row (the decode predictor of HANDOFF 18.35, already installed as
`StreamingSwitchGLU.predict`), takes the union of each row's top-k, and passes the non-resident part as `speculate`
(the store reads it into transient slots while this layer computes). Experts the next layer routes to that the union
missed are read on demand, as lever A of 18.274 reads all of them.

The trace records predictions for decode tokens only (`pred_*`: source layer l, the top 5 of layer l+1's router on
layer l's MoE input). Prefill rows have no recorded prediction, so this prices G6b on a stand-in: a group of N
consecutive decode tokens of one reply plays a prefill chunk of N rows. Both are rows of Hermes text through the same
router; the stand-in's routed-set size is printed beside the real prefill chunks' (217 tokens: 224 of 288 a layer,
18.274) so the reader can judge it. Derived, not measured.

Residency: one LRU of `slots` keys warmed exactly as g6_price.py does (the most-routed decode experts, then one pass
over every decode segment under the budget), held fixed while groups are priced.

Columns, per group of N rows, summed over the 41 predicted layers (4..44):
  routed    mean experts a layer the group routes to
  shipped   reads with the whole-layer read ahead (N >= 128) or demand only (N < 128)
  A         reads of lever A: the routed non-resident set, all on demand (18.274's D arm)
  G ahead   reads G6b starts a layer early (the predicted union's non-resident part)
  G late    routed non-resident experts the union missed, read on demand
  G waste   predicted reads no row routes to
  recall    share of routed non-resident experts that were predicted (read ahead)
"""
import sys
from collections import Counter, OrderedDict

import numpy as np

z = np.load(sys.argv[1])
SLOTS = int(sys.argv[2]) if len(sys.argv) > 2 else 3489
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 2
E_PER_LAYER, SPEC_MIN = 288, 128
ph, L, pos, EX = z["phase"], z["layer"].astype(int), z["position"], z["experts"].astype(int)
W = z["weights"].astype(np.float64)
ps, pt, pp, PE = z["pred_source"].astype(int), z["pred_target"].astype(int), z["pred_position"], z["pred_experts"].astype(int)
PW = z["pred_weights"].astype(np.float64)

change = np.flatnonzero(np.diff(ph.astype(np.int8)) != 0) + 1
bounds = list(zip(np.r_[0, change], np.r_[change, len(ph)]))

# --- residency, as g6_price.py ---
lru: OrderedDict = OrderedDict()
dec = Counter()
for a, b in bounds:
    if ph[a] == 1:
        for i in range(a, b):
            for x in EX[i].tolist():
                dec[(int(L[i]), x)] += 1
for key, _ in dec.most_common(SLOTS):
    lru[key] = None
for a, b in bounds:
    if ph[a] != 1:
        continue
    for i in range(a, b):
        l = int(L[i])
        misses = 0
        for j in np.argsort(-W[i], kind="stable").tolist():
            key = (l, int(EX[i, j]))
            if key in lru:
                lru.move_to_end(key)
                continue
            misses += 1
            if BUDGET >= 0 and misses > BUDGET:
                continue
            lru[key] = None
            if len(lru) > SLOTS:
                lru.popitem(last=False)
resident = {}
for (l, e) in lru:
    resident.setdefault(l, set()).add(e)

# --- decode rows by (position, layer); predictions by (position, target layer) ---
actual = {}
for i in np.flatnonzero(ph == 1).tolist():
    actual[(int(pos[i]), int(L[i]))] = EX[i]
pred = {}
for i in range(len(ps)):
    order = np.argsort(-PW[i], kind="stable")
    pred[(int(pp[i]), int(pt[i]))] = PE[i][order]
targets = sorted(set(pt.tolist()))

# decode segments as position ranges (the positions are reply-token ordinals, contiguous per segment)
segs = []
for a, b in bounds:
    if ph[a] == 1:
        p = np.unique(pos[a:b])
        segs.append(p)

print(f"slots {SLOTS}, decode miss budget {BUDGET}; stand-in groups of N consecutive decode tokens of one reply")
print(f"{'N':>4} {'k':>2} {'groups':>6} | {'routed':>6} | {'shipped':>7} {'A':>6} | {'G ahead':>7} {'G late':>6} {'G waste':>7} "
      f"{'G total':>7} {'recall':>6} | {'A/shp':>6} {'G/shp':>6}")
for N in (24, 64, 128, 172, 256, 384):
    for K in (1, 2, 3, 5):
        acc = Counter()
        groups = 0
        for p in segs:
            for g0 in range(0, len(p) - N + 1, N):
                rows = p[g0:g0 + N]
                groups += 1
                for t in targets:
                    res = resident.get(t, set())
                    A = set()
                    P = set()
                    for r in rows.tolist():
                        A.update(actual[(r, t)].tolist())
                        P.update(pred[(r, t)][:K].tolist())
                    a_miss = A - res
                    p_miss = P - res
                    acc["routed"] += len(A)
                    acc["shipped"] += (E_PER_LAYER - len(res)) if N >= SPEC_MIN else len(a_miss)
                    acc["A"] += len(a_miss)
                    acc["ahead"] += len(p_miss)
                    acc["late"] += len(a_miss - P)
                    acc["waste"] += len(p_miss - A)
        if not groups:
            continue
        nl = len(targets) * groups
        g_total = acc["ahead"] + acc["late"]
        rec = 1 - acc["late"] / max(acc["A"], 1)
        print(f"{N:4d} {K:2d} {groups:6d} | {acc['routed'] / nl:6.1f} | {acc['shipped'] / groups:7.0f} {acc['A'] / groups:6.0f} | "
              f"{acc['ahead'] / groups:7.0f} {acc['late'] / groups:6.0f} {acc['waste'] / groups:7.0f} {g_total / groups:7.0f} "
              f"{rec:6.1%} | {acc['A'] / max(acc['shipped'], 1):6.1%} {g_total / max(acc['shipped'], 1):6.1%}")
print("reads are per group, summed over the predicted layers; recall = predicted share of the routed non-resident set")
