"""Reads a prefill pays, and how many CACHALOT_PREFILL_RARE_SKIP=k (min rows 1024) would save (docs/prefill-rare-skip/RECORD.md).
Usage: price_reads.py <routing trace .npz> <slots, 5179 = 48 GiB> <k>
Residency model: per-layer LRU, quota = slots / 40 (prepare_prefill_layer's even split), shared by prefill and decode.
Approximate (no hotlist pinning, no cross-layer reclaim); for pricing only."""
import sys
from collections import Counter, OrderedDict
import numpy as np

z = np.load(sys.argv[1]); slots = int(sys.argv[2]) if len(sys.argv) > 2 else 5179
ph, L, pos, E, W = z["phase"], z["layer"], z["position"], z["experts"], z["weights"]
NL = 40; Q = slots // NL; CH = 4096; KSKIP = int(sys.argv[3]); MINR = 1024; KS = (KSKIP,)
lru = [OrderedDict() for _ in range(NL)]
def touch(l, x):
    d = lru[l]
    if x in d: d.move_to_end(x); return True
    d[x] = None
    if len(d) > Q: d.popitem(last=False)
    return False
change = np.flatnonzero(np.diff(ph.astype(np.int8)) != 0) + 1
bounds = np.r_[0, change, len(ph)]
print(f"quota {Q}/layer; k {KSKIP}, min rows {MINR}")
pool = Counter()
for a, b in zip(bounds[:-1], bounds[1:]):
    if ph[a] == 1:
        for i in range(a, b):
            for x in E[i].tolist(): touch(int(L[i]), x)
        continue
    p = pos[a:b]; la = L[a:b]; e = E[a:b]; w = W[a:b].astype(np.float64); w /= w.sum(1, keepdims=True)
    chunk = (p - p.min()) // CH; ntok = int(np.unique(p).size)
    reads = 0; rr = Counter(); rm = Counter(); mall = 0.0
    for c in np.unique(chunk):
        for l in np.unique(la).tolist():
            m = (chunk == c) & (la == l)
            if not m.any(): continue
            ee = e[m].ravel().tolist(); ww = w[m].ravel().tolist()
            cnt = Counter(ee); mass = Counter()
            for x, y in zip(ee, ww): mass[x] += y
            mall += sum(ww)
            # most-routed first, as prepare_prefill_layer's order roughly does
            nrows = int(m.sum())
            # shipped DI-1 + keep 140: a prompt of more than one call runs layers 20-39 over 140 rows only (never eligible)
            eligible = nrows >= MINR and not (ntok > CH and l >= 20)
            for x in sorted(cnt, key=lambda x: -cnt[x]):
                if eligible and cnt[x] <= KSKIP:
                    rr["skipped_all"] += 1; rm[KSKIP] += mass[x]
                    if not (x in lru[l]): rr[KSKIP] += 1
                    continue  # skipped: not read, not admitted
                hit = touch(l, x)
                if not hit:
                    reads += 1
    print(f"{ntok:6d} | reads left {reads:6d} | reads saved {rr[KSKIP]:5d} ({rr[KSKIP] / max(reads + rr[KSKIP], 1):5.1%}) | experts skipped {rr['skipped_all']} | mass dropped {rm[KSKIP] / mall:6.2%}")
    if ntok < 16000:
        pool["reads"] += reads + rr[KSKIP]
        for k in KS: pool[k] += rr[k]
print("follow-ups pooled: reads", pool["reads"], " ".join(f"<= {k}: {pool[k]} ({pool[k] / pool['reads']:.1%})" for k in KS))
