"""Offline price (docs/prefill-rare-skip/RECORD.md). Usage: price_rows.py <routing trace .npz>.
In each prefill segment (chunked at 4096 positions), per layer, how many distinct experts are routed
by <= k rows, and what share of the layer's routing mass they carry."""
import sys
import numpy as np
from collections import Counter

z = np.load(sys.argv[1])
ph, L, pos, E, W = z["phase"], z["layer"], z["position"], z["experts"], z["weights"]
change = np.flatnonzero(np.diff(ph.astype(np.int8)) != 0) + 1
bounds = np.r_[0, change, len(ph)]
CH = 4096
KS = (1, 2, 4, 8)
tot = Counter(); tot_mass = Counter(); tot_rows = 0; tot_mass_all = 0.0
print("seg tokens | layers0-19 rows/layer, 20-39 rows/layer | distinct/layer | share of distinct with <=1,2,4,8 rows | mass share <=1,2,4,8")
for a, b in zip(bounds[:-1], bounds[1:]):
    if ph[a] != 0:
        continue
    p = pos[a:b]; la = L[a:b]; e = E[a:b]; w = W[a:b].astype(np.float64)
    w = w / w.sum(1, keepdims=True)
    ntok = int(np.unique(p).size)
    chunk = (p - p.min()) // CH
    dist = 0; rare = Counter(); rmass = Counter(); mass_all = 0.0; nl = 0
    for c in np.unique(chunk):
        for l in np.unique(la):
            m = (chunk == c) & (la == l)
            if not m.any():
                continue
            ee = e[m].ravel(); ww = w[m].ravel()
            cnt = Counter(ee.tolist())
            mass = Counter()
            for x, y in zip(ee.tolist(), ww.tolist()):
                mass[x] += y
            nl += 1; dist += len(cnt); mass_all += ww.sum()
            for k in KS:
                rs = [x for x, n in cnt.items() if n <= k]
                rare[k] += len(rs); rmass[k] += sum(mass[x] for x in rs)
    r_lo = int(((la < 20)).sum()) / max(int(np.unique(la[la < 20]).size), 1)
    r_hi = int(((la >= 20)).sum()) / max(int(np.unique(la[la >= 20]).size), 1)
    print(f"{ntok:6d} | {r_lo:7.0f} {r_hi:7.0f} | {dist / nl:6.1f} | "
          + " ".join(f"{rare[k] / dist:5.1%}" for k in KS) + " | " + " ".join(f"{rmass[k] / mass_all:6.2%}" for k in KS)
          + f" | distinct total {dist}")
    if ntok < 16000:
        for k in KS:
            tot[k] += rare[k]; tot_mass[k] += rmass[k]
        tot_rows += dist; tot_mass_all += mass_all
print("follow-ups pooled: distinct", tot_rows, " ".join(f"<= {k}: {tot[k]} ({tot[k] / tot_rows:.1%}, mass {tot_mass[k] / tot_mass_all:.2%})" for k in KS))
