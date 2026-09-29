"""Price the per-expert (scale, bias) rebuild in MiniMax prefill at Hermes-shaped chunk sizes.

Synthetic pair slots (random weights, random pair index, random 256-entry table) at MiniMax's shapes; one layer's
experts with row counts drawn from a skewed top-4-of-128 routing; 60 layers; eval per layer (as prefill does).
Arms: cur = rebuild_pair per projection then quantized_matmul (shipped); pre = the same matmuls on precomputed
(scale, bias) (what a kernel that reads the pair index itself approaches). ABAB, per chunk size.
"""
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/hamedprooshani/Projects/deepseek-v41-mac/src")
from cachalot.minimax import codes_qmv  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pq2  # noqa: E402

H, I, G = 6144, 3072, 64
E_TOT, K = 128, 4
LAYERS = int(os.environ.get("LAYERS", "60"))
rng = np.random.default_rng(0)


def proj(out, inn):
    w = mx.array(rng.integers(0, 2**32, size=(out, inn * 3 // 32), dtype=np.uint32))
    pidx = mx.array(rng.integers(0, 200, size=(out, inn // G), dtype=np.uint8))
    plut = np.zeros(512, np.uint16)
    s = (rng.random(256) * 0.01 + 0.001).astype(np.float32)
    b = (-5 * s).astype(np.float32)
    bf = lambda a: (a.view(np.uint32) >> 16).astype(np.uint16)
    plut[0::2], plut[1::2] = bf(s), bf(b)
    plut = mx.array(plut).view(mx.bfloat16)
    ps, pb = codes_qmv.rebuild_pair(pidx, plut)
    return w, pidx, plut, ps, pb


experts = [{p: proj(*(I, H) if p != "w2" else (H, I)) for p in ("w1", "w3", "w2")} for _ in range(E_TOT)]
mx.eval([[v for t in e.values() for v in t] for e in experts])
pop = 1.0 / (np.arange(E_TOT) + 1) ** 0.9
pop /= pop.sum()


def qmm(x, t, arm):
    w, pidx, plut, ps, pb = t
    if arm == "new":
        return pq2.pair_mm(x, w, pidx, plut)
    if arm == "cur":
        s, b = codes_qmv.rebuild_pair(pidx, plut)
    else:
        s, b = ps, pb
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=G, bits=3)


def layer(x, counts, arm):
    outs = []
    start = 0
    for e, c in counts:
        xe = x[start % x.shape[0]: start % x.shape[0] + c] if start % x.shape[0] + c <= x.shape[0] else x[:c]
        start += c
        t = experts[e]
        g = qmm(xe, t["w1"], arm)
        u = qmm(xe, t["w3"], arm)
        outs.append(qmm(mx.sigmoid(g) * g * u, t["w2"], arm))
    y = mx.concatenate(outs, axis=0)
    mx.eval(y)
    return y


def routing(n):
    picks = np.array([rng.choice(E_TOT, K, replace=False, p=pop) for _ in range(n)]).reshape(-1)
    e, c = np.unique(picks, return_counts=True)
    return list(zip(e.tolist(), c.tolist()))


for n in [int(v) for v in os.environ.get("NS", "128,400,2048,8192").split(",")]:
    x = mx.random.normal((n, H)).astype(mx.bfloat16)
    routes = [routing(n) for _ in range(LAYERS)]
    ne = np.mean([len(r) for r in routes])
    res = {"cur": [], "pre": [], "new": []}
    layer(x, routes[0], "cur")
    for arm in ["cur", "new", "pre", "new", "cur", "pre", "pre", "cur", "new"]:
        t0 = time.perf_counter()
        for r in routes:
            layer(x, r, arm)
        res[arm].append(time.perf_counter() - t0)
    c, p, nw = np.mean(res["cur"]), np.mean(res["pre"]), np.mean(res["new"])
    print(f"N={n:5d} cur={c:.3f} new={nw:.3f} ({(nw-c)/c*100:+.1f}%) pre={p:.3f} new runs {[round(v,3) for v in res['new']]}", flush=True)
    print(f"N={n:5d} experts/layer={ne:.0f} cur={c:.3f}s pre={p:.3f}s rebuild={c - p:.3f}s ({(c - p) / c * 100:.1f}%)"
          f"  cur runs {['%.3f' % v for v in res['cur']]} pre runs {['%.3f' % v for v in res['pre']]}", flush=True)
