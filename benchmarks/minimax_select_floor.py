"""S0(b): the all-hit floor with GPU-side expert selection (HANDOFF 18.12 item 3; 18.10's nosync_floor extended).

Replays the identical token (KV rewound each step). Arms alternate:
  A  shipped path (one host sync per MoE layer)
  B  the previous identical step's routing reused on the host (18.10: no per-layer sync, per-expert matmuls)
  C  routing computed on the GPU, mapped through a (layer, expert) -> slot table to stacked copies of the
     routed experts, `gather_qmm` for the three projections: no host read-back at all.
C's stacks hold only the experts step A routed (4 per layer, biases rebuilt from the slot's codes); the table
maps every other expert to slot 0, which the all-hit replay never selects. Logits compared bit for bit.
  D  C, plus each layer submitted (async_eval) and the host reading the previous layer's slot indices: the
     shipped GpuSelectDecoder's shape (0.31.0).
Arm A is the per-layer-sync path, so this script sets CACHALOT_MINIMAX_GPU_SELECT=0 before the model loads.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    env CACHALOT_PAGE_CACHE=0 CACHALOT_MLX_WIRED_LIMIT_GIB=80 MLX_METAL_FAST_SYNCH=1 PYTHONPATH=src \
      CACHALOT_MINIMAX_BANK=$HOME/MiniMax-M3-coded-bank FILLER_FILE=... FILLER_OFFSET=200000 \
      ~/venvs/deepseek-v41/bin/python benchmarks/minimax_select_floor.py $HOME/MiniMax-M3-MLX-3bit
"""
import os

os.environ["CACHALOT_MINIMAX_GPU_SELECT"] = "0"
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

from cachalot.minimax import codes_qmv
from cachalot.minimax.model import MiniMaxModel
import cachalot.minimax.language as lang

MODEL = sys.argv[1]
STEPS = int(os.environ.get("STEPS", "20"))
CONTEXT = int(os.environ.get("CONTEXT", "2048"))

m = MiniMaxModel(MODEL, expert_budget_gib=52, heartbeat_seconds=0, verbose=False)
text = open(os.environ["FILLER_FILE"]).read()[int(os.environ.get("FILLER_OFFSET", "0")):]
tokens = m.tokenizer.encode(text[: CONTEXT * 8], add_special_tokens=False)[: CONTEXT + 1]
cache = m.new_cache()
mx.eval(m.prefill(tokens[:CONTEXT], cache) if hasattr(m, "prefill") else m._forward(tokens[:CONTEXT], cache))
m.store.release_prefill()
tok = tokens[CONTEXT]

recorded = {}
MODE = ["A"]
moes = [(i, l.block_sparse_moe) for i, l in enumerate(m.model.layers) if l.is_sparse]
for i, moe in moes:
    sw = moe.switch_mlp
    moe._cachalot_layer = i

    def wrapped(x, indices, prefetch=None, speculate=None, _i=i, _sw=sw, _orig=type(sw).__call__):
        if x.reshape(-1, x.shape[-1]).shape[0] == 1 and MODE[0] == "B" and _i in recorded:
            routes = recorded[_i]
            flat = x.reshape(-1, x.shape[-1])
            outs = [_sw._expert_out(flat, _sw._store.get_many([_sw._index[(_i, e)]])[0].slot) for e in routes]
            return mx.concatenate(outs, axis=0).reshape(*x.shape[:-1], len(routes), x.shape[-1])
        if x.reshape(-1, x.shape[-1]).shape[0] == 1:
            recorded[_i] = [int(e) for e in np.array(indices).reshape(-1)]
        return _orig(_sw, x, indices, prefetch=prefetch, speculate=speculate)

    moe._cachalot_sw = wrapped

stacks = {}
PENDING = [None]
ROUTES_SEEN = []  # layer -> (table, {proj: (w, s, b)})


def build_stacks():
    from cachalot.glm.experts import _typed

    for i, moe in moes:
        sw = moe.switch_mlp
        routes = recorded[i]
        n_exp = moe.gate.weight.shape[0] if hasattr(moe.gate, "weight") else 128
        table = np.zeros(n_exp, np.uint32)
        for pos, e in enumerate(routes):
            table[e] = pos
        slots = [sw._store.get_many([sw._index[(i, e)]])[0].slot for e in routes]
        projs = {}
        for proj in ("w1", "w3", "w2"):
            parts = [_typed(s, sw._fmt, proj) for s in slots]
            w = mx.stack([p[0] for p in parts])
            sc = mx.stack([p[1] for p in parts])
            b = mx.stack([codes_qmv.rebuild_biases(p[1], p[2]) if sw.codes else p[2] for p in parts])
            projs[proj] = (w, sc, b)
        mx.eval(*[a for t in projs.values() for a in t])
        stacks[i] = (mx.array(table), projs)


_orig_block = lang.MiniMaxM3SparseMoeBlock.__call__


def routed_weights(self, x):
    scores, orig_scores = self.route_scores(x)
    k = self.num_experts_per_tok
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    weights = mx.take_along_axis(orig_scores, inds, axis=-1)
    weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    weights = (weights * self.routed_scaling_factor).astype(x.dtype)
    return inds, weights


def block_call(self, x, residual=None):
    if x.shape[1] == 1 and MODE[0] in ("C", "D"):
        inds, weights = routed_weights(self, x)
        table, projs = stacks[self._cachalot_layer]
        idx = table[inds]  # (B, L, k), on the GPU
        sw = self.switch_mlp
        gs, bits = sw._fmt.group_size, sw._fmt.bits
        xe = mx.expand_dims(x, (-2, -3))  # (B, L, 1, 1, D)
        gate = mx.gather_qmm(xe, *projs["w1"], rhs_indices=idx, transpose=True, group_size=gs, bits=bits)
        up = mx.gather_qmm(xe, *projs["w3"], rhs_indices=idx, transpose=True, group_size=gs, bits=bits)
        h = sw._activation(up, gate)
        y = mx.gather_qmm(h, *projs["w2"], rhs_indices=idx, transpose=True, group_size=gs, bits=bits)
        y = y.squeeze(-2)  # (B, L, k, D)
        y = (y * weights[..., None]).sum(axis=-2)
        y = y + self.shared_experts(x)
        if MODE[0] == "D":
            # submit this layer, then read the previous layer's routing on the host (one step behind)
            mx.async_eval(y, idx)
            prev = PENDING[0]
            if prev is not None:
                ROUTES_SEEN.append(np.array(prev))
            PENDING[0] = idx
        return y
    if MODE[0] == "B" and x.shape[1] == 1:
        inds, weights = routed_weights(self, x)
        y = self._cachalot_sw(x, inds)
        y = (y * weights[..., None]).sum(axis=-2)
        return y + self.shared_experts(x)
    decode_hook = getattr(self, "decode_hook", None)
    if decode_hook is not None and x.shape[1] == 1:
        inds, weights = routed_weights(self, x)
        shared, prefetch = decode_hook(x, residual, inds, weights)
        y = self._cachalot_sw(x, inds, prefetch=prefetch)
        y = (y * weights[..., None]).sum(axis=-2)
        return y + shared
    return _orig_block(self, x, residual)


lang.MiniMaxM3SparseMoeBlock.__call__ = block_call


def step():
    PENDING[0] = None
    out = m._forward([tok], cache)
    mx.eval(out)
    for c in cache:
        c.offset -= 1
    return out


MODE[0] = "A"
ref = step()
ref = step()
build_stacks()
arms = os.environ.get("ARMS", "A,B,C,D").split(",")
res = {a: [] for a in arms}
outs = {}
for s in range(STEPS):
    for arm in (arms if s % 2 == 0 else arms[::-1]):
        MODE[0] = arm
        t0 = time.perf_counter()
        outs[arm] = step()
        res[arm].append(time.perf_counter() - t0)
report = " ".join(f"{a} median={1000*statistics.median(res[a]):.1f} min={1000*min(res[a]):.1f}" for a in arms)
same = {a: bool(mx.array_equal(outs[a], ref).item()) for a in arms}
diff = {a: float(mx.max(mx.abs(outs[a].astype(mx.float32) - ref.astype(mx.float32))).item()) for a in arms}
print(f"S0FLOOR context={CONTEXT} {report} identical={same} maxdiff={diff} misses={m.store.stats().cache_misses}",
      flush=True)
m.close()
