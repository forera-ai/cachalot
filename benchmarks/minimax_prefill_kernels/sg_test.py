import json, os, sys, time
import numpy as np, mlx.core as mx
sys.path.insert(0, "/Users/hamedprooshani/Projects/deepseek-v41-mac/src")
from cachalot.glm.experts import ExpertFormat, tensor_sizes, _typed
from cachalot.minimax.coded_bank import slot_format
from cachalot.cache.slots import SlabSlotPool
from cachalot.minimax import codes_qmv as cq
import sg

f = json.load(open(os.path.expanduser("~/MiniMax-M3-coded-bank/bank.json")))["format"]
fmt = ExpertFormat(kind=f["kind"], bits=f["bits"], group_size=f["group_size"], tensor_names=tuple(f["tensor_names"]),
                   shapes={k: tuple(v) for k, v in f["shapes"].items()}, dtypes=f["dtypes"])
sf = slot_format(fmt, 2)
SLOTS = int(os.environ.get("SLOTS", "300"))
pool = SlabSlotPool(tensor_sizes(sf), SLOTS, slab_slots=int(os.environ.get("SLAB", "64")))
print("slabs", len(pool.slabs), "slab_slots", pool.slab_slots, "record", pool.record_bytes, flush=True)
rng = np.random.default_rng(0)
# fill every slot: random weights, pair indices < 200, a random table of (scale, bias)
for s in pool._slots:
    for name, v in s.views.items():
        a = np.asarray(v)
        if name.endswith(".plut"):
            sc = (rng.random(256) * 0.01 + 0.001).astype(np.float32)
            t = np.zeros(512, np.uint16)
            t[0::2] = sc.view(np.uint32) >> 16
            t[1::2] = (-(rng.integers(3, 8, 256)) * sc).astype(np.float32).view(np.uint32) >> 16
            a.view(np.uint16)[:] = t
        elif name.endswith(".pidx"):
            a[:] = rng.integers(0, 200, a.size, dtype=np.uint8)
        else:
            a[:] = rng.integers(0, 256, a.size, dtype=np.uint8)
mx.eval(pool.slabs)
H, I = 6144, 3072
act = lambda u, g: mx.sigmoid(g) * g * u
G = sg.SlabGather(pool, I, H)


class Slot:
    pass


def fmt_views(i):
    return pool._slots[i]


def loop(xs, seg):
    """current path: rebuild + quantized_matmul per expert, rows already sorted"""
    outs = []
    for slot, a, b in seg:
        xe = xs[a:b]
        s = pool._slots[slot]
        def q(x, p):
            w, pi, _, pl = _typed(s, sf, p)
            S, B = cq.rebuild_pair(pi, pl)
            return mx.quantized_matmul(x, w, S, B, transpose=True, group_size=64, bits=3)
        outs.append(q(act(q(xe, "w3"), q(xe, "w1")), "w2"))
    return mx.concatenate(outs)


def gref(xs, seg):
    """mx.gather_qmm on stacked rebuilt arrays of the used slots (MLX's own rhs kernel)"""
    used = [slot for slot, _, _ in seg]
    idx = np.concatenate([np.full(b - a, j, np.uint32) for j, (_, a, b) in enumerate(seg)])
    def st(p):
        ws, ss, bs = [], [], []
        for slot in used:
            w, pi, _, pl = _typed(pool._slots[slot], sf, p)
            S, B = cq.rebuild_pair(pi, pl)
            ws.append(w); ss.append(S); bs.append(B)
        return mx.stack(ws), mx.stack(ss), mx.stack(bs)
    ix = mx.array(idx)
    x3 = mx.expand_dims(xs, -2)
    g = mx.gather_qmm(x3, *st("w1"), rhs_indices=ix, transpose=True, group_size=64, bits=3, sorted_indices=True)
    u = mx.gather_qmm(x3, *st("w3"), rhs_indices=ix, transpose=True, group_size=64, bits=3, sorted_indices=True)
    y = mx.gather_qmm(act(u, g), *st("w2"), rhs_indices=ix, transpose=True, group_size=64, bits=3, sorted_indices=True)
    return y.reshape(xs.shape[0], H)


pop = 1.0 / (np.arange(128) + 1) ** 0.9
pop /= pop.sum()


def make(n):
    picks = np.array([rng.choice(128, 4, replace=False, p=pop) for _ in range(n)]).reshape(-1)
    e, c = np.unique(picks, return_counts=True)
    slots = rng.choice(SLOTS, len(e), replace=False)
    seg, a = [], 0
    for s, cnt in zip(slots, c):
        seg.append((int(s), a, a + int(cnt)))
        a += int(cnt)
    idx = np.concatenate([[a], np.concatenate([np.full(b - a0, s, np.int32) for s, a0, b in seg])]).astype(np.int32)
    return seg, mx.array(idx), a


for n in [int(v) for v in os.environ.get("NS", "16,128,400,1000,2048").split(",")]:
    seg, idx, R = make(n)
    xs = mx.random.normal((R, H)).astype(mx.bfloat16)
    y_new = G(xs, idx, act)
    y_ref = gref(xs, seg)
    y_cur = loop(xs, seg)
    mx.eval(y_new, y_ref, y_cur)
    eq = bool(mx.array_equal(y_new, y_ref).item())
    d = mx.abs(y_new.astype(mx.float32) - y_cur.astype(mx.float32))
    rel = (mx.max(d) / mx.max(mx.abs(y_cur.astype(mx.float32)))).item()
    t = {}
    for name, fn in [("cur", lambda: loop(xs, seg)), ("new", lambda: G(xs, idx, act))] * 3:
        t0 = time.perf_counter()
        for _ in range(5):
            mx.eval(fn())
        t.setdefault(name, []).append((time.perf_counter() - t0) / 5 * 1000)
    print(f"N={n:5d} rows={R} experts={len(seg)} new==gather_qmm {eq}  max rel diff vs cur {rel:.2e}  "
          f"cur {min(t['cur']):.1f} ms  new {min(t['new']):.1f} ms  ({(min(t['new']) / min(t['cur']) - 1) * 100:+.1f}%)",
          flush=True)
