"""G6's price: what a GLM short prefill reads, and what two levers would leave unread (docs/glm-speed/G6-RECORD.md).

Usage: g6_price.py <routing trace .npz> [slots, 3489 = 46 GiB] [decode miss budget, 2; -1 exact] [B threshold, 0.20]
A trace without weights prices lever A only (decode then keeps the routed order, not the weight order).

Model (derived, for pricing only):
- Residency: one global LRU of `slots` (layer, expert) keys. Decode tokens touch their routed experts; with a decode
  miss budget b, only the b highest-weight misses of a layer are read and admitted, the rest dropped (serve-glm.sh's
  default is 2). Prefill does not change residency (the store's scan path: residents kept, misses through transient
  slots). The LRU starts with the trace's most-routed decode experts (a warm-set stand-in) and runs once over the
  whole trace before any prefill is priced, so early segments are not priced on an empty cache.
- Prefill: chunks of 2,048 tokens. Each MoE layer reads its routed non-resident experts (demand). From 128 tokens
  (SPECULATE_MIN_TOKENS) layer l also queues every non-resident expert of layer l+1 (speculation), so a chunk's reads
  are the first layer's demand plus every other layer's whole non-resident set.
- Lever A (bit-identical): no whole-layer speculation for the chunk; reads = routed non-resident experts only.
- Lever B (output-changing, MiniMax's 0.42.0 rule on top of A): a missing expert routed to at most 32 rows whose
  weight share is under 0.20 in every one of them is not read; its rows change. The rule also needs a resident
  runner-up among the next four ranks of each row, which a top-8 trace cannot show: B is an upper bound.
- Per affected row (18.260's lesson): the share of the row's own routing weight that B takes away.
"""
import sys
from collections import Counter, OrderedDict

import numpy as np

z = np.load(sys.argv[1])
SLOTS = int(sys.argv[2]) if len(sys.argv) > 2 else 3489
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 2
THR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.20
E_PER_LAYER, CH, SPEC_MIN, MAX_ROWS = 288, 2048, 128, 32
ph, L, pos, EX = z["phase"], z["layer"].astype(int), z["position"], z["experts"].astype(int)
HAS_W = "weights" in z.files
W = z["weights"].astype(np.float64) if HAS_W else np.tile(np.arange(EX.shape[1], 0, -1, dtype=np.float64), (len(EX), 1))
layers = sorted(set(L.tolist()))

change = np.flatnonzero(np.diff(ph.astype(np.int8)) != 0) + 1
bounds = list(zip(np.r_[0, change], np.r_[change, len(ph)]))

lru: OrderedDict = OrderedDict()
dec = Counter()
for a, b in bounds:
    if ph[a] == 1:
        for i in range(a, b):
            for x in EX[i].tolist():
                dec[(int(L[i]), x)] += 1
for key, _ in dec.most_common(SLOTS):
    lru[key] = None


def decode_segment(a, b):
    for i in range(a, b):
        l = int(L[i])
        order = np.argsort(-W[i], kind="stable")
        misses = 0
        for j in order.tolist():
            key = (l, int(EX[i, j]))
            if key in lru:
                lru.move_to_end(key)
                continue
            misses += 1
            if BUDGET >= 0 and misses > BUDGET:
                continue  # dropped, not read, not admitted
            lru[key] = None
            if len(lru) > SLOTS:
                lru.popitem(last=False)


# one warm-up pass over every decode segment, so the first prefills see a steady-state cache
for a, b in bounds:
    if ph[a] == 1:
        decode_segment(a, b)

print(f"B threshold {THR}; weights {'yes' if HAS_W else 'no: B columns meaningless'}")
print(f"slots {SLOTS} ({SLOTS / (len(layers) * E_PER_LAYER):.1%} of {len(layers)} x {E_PER_LAYER}); decode miss budget {BUDGET}")
print(f"{'tokens':>6} {'chunk':>5} | {'routed':>6} {'resid':>5} | {'shipped':>7} {'A':>6} {'A save':>6} | {'B':>6} {'B save':>6} | "
      f"{'rows hit':>8} {'med lost':>8} {'p90 lost':>8} {'max lost':>8}")
for a, b in bounds:
    if ph[a] == 1:
        decode_segment(a, b)
        continue
    p = pos[a:b]
    ntok = int(np.unique(p).size)
    chunk_of = (p - p.min()) // CH
    for c in np.unique(chunk_of).tolist():
        m = chunk_of == c
        idx = np.flatnonzero(m) + a
        rows_in_chunk = int(np.unique(pos[idx]).size)
        shipped = demand = keep_b = routed = resid = 0
        lost_rows = []
        first = True
        for l in layers:
            li = idx[L[idx] == l]
            if li.size == 0:
                continue
            ex = EX[li]
            w = W[li] / W[li].sum(axis=1, keepdims=True)
            cnt = Counter(ex.ravel().tolist())
            res_l = sum(1 for e in range(E_PER_LAYER) if (l, e) in lru)
            miss = [e for e in cnt if (l, e) not in lru]
            routed += len(cnt)
            resid += res_l
            demand += len(miss)
            if first or rows_in_chunk < SPEC_MIN:
                shipped += len(miss)
            else:
                shipped += E_PER_LAYER - res_l
            first = False
            skip = set()
            for e in miss:
                if cnt[e] > MAX_ROWS:
                    continue
                if (w[ex == e] < THR).all():
                    skip.add(e)
            keep_b += len(miss) - len(skip)
            if skip:
                lost = np.where(np.isin(ex, list(skip)), w, 0.0).sum(axis=1)
                lost_rows.extend(lost[lost > 0].tolist())
        nl = len(layers)
        lr = np.array(lost_rows) if lost_rows else np.zeros(1)
        print(f"{ntok:6d} {c:5d} | {routed / nl:6.1f} {resid / nl:5.1f} | {shipped:7d} {demand:6d} {1 - demand / max(shipped, 1):6.1%} | "
              f"{keep_b:6d} {1 - keep_b / max(shipped, 1):6.1%} | {len(lost_rows) / (rows_in_chunk * nl):8.1%} "
              f"{np.median(lr):8.3f} {np.percentile(lr, 90):8.3f} {lr.max():8.3f}")
print("routed and resid are per layer (mean over layers); shipped, A and B are reads in the chunk; "
      "rows hit is the share of (row, layer) pairs B changes; lost is the routing weight share a changed row loses")
