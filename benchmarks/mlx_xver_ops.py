"""Cross-version MLX op battery: same seeded inputs, outputs saved for a bitwise compare between MLX builds.

Usage: PYTHONPATH=src <venv>/bin/python benchmarks/mlx_xver_ops.py OUT_DIR   (once per MLX build; HANDOFF section 18.223)
"""
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1234)
results = {}


def bf16(a):
    return mx.array(a.astype(np.float32)).astype(mx.bfloat16)


def rand(*shape, scale=1.0):
    return bf16(rng.standard_normal(shape).astype(np.float32) * scale)


def save(name, arr):
    mx.eval(arr)
    a = np.array(arr.astype(mx.float32)) if arr.dtype in (mx.bfloat16, mx.float16) else np.array(arr)
    results[name] = a


def qweights(out, inp, bits, group):
    """Version-independent packed weights: random uint32 words, positive scales, mixed-sign biases."""
    words = inp * bits // 32
    w = mx.array(rng.integers(0, 2**32, size=(out, words), dtype=np.uint64).astype(np.uint32))
    s = bf16(np.abs(rng.standard_normal((out, inp // group))) * 0.01 + 1e-3)
    b = bf16(rng.standard_normal((out, inp // group)) * 0.02)
    return w, s, b


H, INTER, R, G, GD = 5120, 2304, 1024, 8, 4096

# Routed expert affine 2-bit g128 (DeepSeek bank) and MiniMax-like shapes, decode and prefill row counts.
for (o, i) in [(INTER, H), (H, INTER), (3072, 6144), (6144, 3072)]:
    w, s, b = qweights(o, i, 2, 128)
    for m in (1, 4, 6, 8, 16, 33, 128, 512, 2048):
        x = rand(m, i)
        save(f"qmm2_g128_{o}x{i}_m{m}", mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2))

# Affine 8-bit g32 (fp4_affine8 path) and 4-bit g64.
for bits, group in [(8, 32), (4, 64), (3, 128)]:
    w, s, b = qweights(1280, H, bits, group)
    for m in (1, 8, 512):
        x = rand(m, H)
        save(f"qmm{bits}_g{group}_m{m}", mx.quantized_matmul(x, w, s, b, transpose=True, group_size=group, bits=bits))

# gather_qmm over 6 of 32 experts, prefill-like, sorted and unsorted indices.
E = 32
w = mx.array(rng.integers(0, 2**32, size=(E, INTER, H * 2 // 32), dtype=np.uint64).astype(np.uint32))
s = bf16(np.abs(rng.standard_normal((E, INTER, H // 128))) * 0.01 + 1e-3)
b = bf16(rng.standard_normal((E, INTER, H // 128)) * 0.02)
for m in (1, 64, 512):
    x = rand(m, 1, 1, H)
    idx = mx.array(rng.integers(0, E, size=(m, 6)).astype(np.uint32))
    save(f"gather_qmm_m{m}", mx.gather_qmm(x, w, s, b, rhs_indices=idx, transpose=True, group_size=128, bits=2))
    order = mx.argsort(idx.flatten())
    xs = mx.broadcast_to(x, (m, 6, 1, H)).reshape(m * 6, 1, H)[order // 1]
    save(f"gather_qmm_sorted_m{m}", mx.gather_qmm(
        xs, w, s, b, rhs_indices=idx.flatten()[order], transpose=True, group_size=128, bits=2,
        sorted_indices=True))

# bf16 matmuls: grouped wo_a decode/prefill, lm head, thin shapes.
wo_a = rand(G, R, GD, scale=0.02)
for m in (1, 8, 512):
    o = rand(m, G, GD)
    save(f"wo_a_grouped_m{m}", mx.matmul(wo_a[None], o[..., None])[..., 0])
    save(f"wo_a_einsum_m{m}", mx.einsum("sgd,grd->sgr", o, wo_a))
head = rand(129280, H, scale=0.02)
for m in (1, 4):
    save(f"head_m{m}", rand(m, H) @ head.T)
for (m, k, n) in [(1, H, 1280), (6, H, 1280), (16, H, 1280), (512, H, 1280), (512, 1280, 32768), (2048, 4096, 1024)]:
    save(f"mm_{m}x{k}x{n}", rand(m, k) @ rand(k, n, scale=0.02))

# Attention: decode vector path (1 query), mid rows, prefill causal.
for (q_len, k_len, hq, hk, d) in [(1, 640, 64, 1, 512), (1, 4096, 64, 4, 128), (4, 4096, 64, 4, 128),
                                  (512, 512, 64, 4, 128), (2048, 2048, 64, 1, 512)]:
    q, k, v = rand(1, hq, q_len, d), rand(1, hk, k_len, d), rand(1, hk, k_len, d)
    mask = "causal" if q_len == k_len else None
    save(f"sdpa_q{q_len}_k{k_len}_h{hq}_{hk}_d{d}",
         mx.fast.scaled_dot_product_attention(q, k, v, scale=d ** -0.5, mask=mask))

# Elementwise, norms, reductions, sorting, fp8.
x = rand(512, H)
wn = rand(H)
save("rms_norm", mx.fast.rms_norm(x, wn, 1e-6))
save("rms_norm_1", mx.fast.rms_norm(x[:1], wn, 1e-6))
save("layer_norm", mx.fast.layer_norm(x, wn, None, 1e-5))
xf = x.astype(mx.float32)
for name, fn in [("softmax", lambda a: mx.softmax(a, axis=-1)), ("sigmoid", mx.sigmoid),
                 ("softplus_sqrt", lambda a: mx.sqrt(mx.logaddexp(a, 0.0))), ("exp", mx.exp), ("silu", lambda a: a * mx.sigmoid(a)),
                 ("logsumexp", lambda a: mx.logsumexp(a, axis=-1)), ("cumsum", lambda a: mx.cumsum(a, axis=-1)),
                 ("sum", lambda a: mx.sum(a, axis=-1)), ("mean", lambda a: mx.mean(a, axis=0))]:
    save(f"{name}_bf16", fn(x))
    save(f"{name}_f32", fn(xf))
save("sigmoid_compiled_bf16", mx.compile(mx.sigmoid)(x))
save("silu_compiled_f32", mx.compile(lambda a: a * mx.sigmoid(a))(xf))
scores = mx.array(rng.standard_normal((512, 384)).astype(np.float32))
save("argsort", mx.argsort(-scores, axis=-1))
save("argpartition6", mx.sort(mx.argpartition(-scores, kth=5, axis=-1)[:, :6], axis=-1))
save("topk6", mx.topk(scores, 6, axis=-1))
fp = mx.to_fp8(mx.clip(xf, -448, 448))
save("to_fp8", fp)
save("from_fp8", mx.from_fp8(fp, dtype=mx.bfloat16))
wq, sq, bq = mx.quantize(rand(INTER, H, scale=0.02), group_size=128, bits=2)
save("quantize2_w", wq)
save("quantize2_s", sq)
save("quantize2_b", bq)
save("dequantize2", mx.dequantize(wq, sq, bq, group_size=128, bits=2))
neg = -mx.abs(rand(64, 128))
qn = mx.quantize(neg, group_size=128, bits=2)
save("quantize2_allneg_dequant", mx.dequantize(*qn, group_size=128, bits=2))

# The repo's own Metal kernels (same source on both builds; checks the JIT path).
try:
    from cachalot.model.fp8_fused_metal import fp8_linear_fused, fp8_roundtrip_fused
    wq8 = mx.array(rng.integers(0, 255, size=(1280, H), dtype=np.uint8))
    sq8 = mx.array(rng.integers(110, 130, size=((1280 + 31) // 32, H // 32), dtype=np.uint8))
    save("repo_fp8_linear_fused", fp8_linear_fused(rand(H), wq8, sq8))
    save("repo_fp8_roundtrip_fused", fp8_roundtrip_fused(rand(G * GD)))
    from cachalot.minimax import codes_qmv
    results["repo_codes_qmv_self_check"] = np.array([codes_qmv.self_check()])
except Exception as e:  # noqa: BLE001
    print("repo kernels skipped:", repr(e))

np.savez(OUT / "ops.npz", **results)
print("mlx", mx.__version__, "saved", len(results), "outputs to", OUT, time.strftime("%H:%M:%S"))
