"""Price of adding the router weights to the sync that already reads the indices (isolated, no model).
A: mx.eval(indices) then np.array(indices)   B: mx.eval(indices, weights) then np.array of both. Interleaved.

HANDOFF 18.101. An upper bound for the extra GPU work: here the weights are used by nothing else, whereas in a
model they are consumed on the GPU anyway, so evaluating them at the sync moves work rather than adds it.
Run from the repository root: ~/venvs/deepseek-v41/bin/python benchmarks/router_price.py (uses the GPU for ~10 s)."""
import sys, time, statistics as st
sys.path.insert(0, "src")
import numpy as np, mlx.core as mx
from cachalot.third_party.mlx_vlm.models.glm5_next.language import _expert_select

def glm(h, E, k):
    W = mx.random.normal((E, h)).astype(mx.float32) * 0.02; bias = mx.zeros((E,), mx.float32)
    def f(x):
        return _expert_select(x.astype(mx.float32) @ W.T, bias, k, 1, 1, 2.5, True)
    return f, h

def minimax(h, E, k):
    W = mx.random.normal((E, h)).astype(mx.float32) * 0.02; bias = mx.zeros((E,), mx.float32)
    def f(x):
        raw = x.astype(mx.float32) @ W.T
        scores = mx.sigmoid(raw); choice = scores + bias
        inds = mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k]
        w = mx.take_along_axis(scores, inds, axis=-1); w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
        return inds, (w * 2.5).astype(mx.bfloat16)
    return f, h

def run(name, f, h, n=3000):
    ta, tb = [], []
    for i in range(n + 200):
        x = mx.random.normal((1, h)).astype(mx.bfloat16)
        for which in ((0, 1) if i % 2 == 0 else (1, 0)):
            inds, w = f(x)
            t = time.perf_counter()
            if which == 0:
                mx.eval(inds); np.array(inds)
            else:
                mx.eval(inds, w); np.array(inds); np.array(w.astype(mx.float32))
            d = (time.perf_counter() - t) * 1e6
            if i >= 200: (ta if which == 0 else tb).append(d)
    a, b = st.median(ta), st.median(tb)
    print(f"{name}: indices only {a:.0f} us, indices + weights {b:.0f} us, difference {b - a:+.0f} us (p25-p75 of each: {np.percentile(ta,25):.0f}-{np.percentile(ta,75):.0f} / {np.percentile(tb,25):.0f}-{np.percentile(tb,75):.0f})")
    return b - a

d1 = run("GLM router (4096 -> 288, top 8)", *glm(4096, 288, 8))
d2 = run("MiniMax router (6144 -> 128, top 4, bf16 weights)", *minimax(6144, 128, 4))
print(f"per token: GLM 42 layers x {d1:+.0f} us = {42*d1/1000:+.2f} ms (of ~1,500 ms); MiniMax 57 layers x {d2:+.0f} us = {57*d2/1000:+.2f} ms (of ~45-90 ms)")
