"""Error of quantized_matmul at M=1 against a float64 reference built from mx.dequantize (same on both builds).

Usage: PYTHONPATH=src <venv>/bin/python benchmarks/mlx_qmv_accuracy.py   (once per MLX build; HANDOFF section 18.223)
"""
import mlx.core as mx
import numpy as np

rng = np.random.default_rng(7)
out = {}
for bits, group, (o, i) in [(2, 128, (2304, 5120)), (2, 128, (5120, 2304)), (3, 64, (3072, 6144)), (4, 64, (1280, 5120))]:
    errs = []
    for _ in range(8):
        w = mx.array(rng.integers(0, 2**32, size=(o, i * bits // 32), dtype=np.uint64).astype(np.uint32))
        s = mx.array((np.abs(rng.standard_normal((o, i // group))) * 0.01 + 1e-3).astype(np.float32)).astype(mx.bfloat16)
        b = mx.array((rng.standard_normal((o, i // group)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
        x = mx.array(rng.standard_normal((1, i)).astype(np.float32)).astype(mx.bfloat16)
        y = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=group, bits=bits)
        W = np.array(mx.dequantize(w, s, b, group_size=group, bits=bits).astype(mx.float32)).astype(np.float64)
        ref = np.array(x.astype(mx.float32)).astype(np.float64) @ W.T
        e = np.array(y.astype(mx.float32)).astype(np.float64) - ref
        errs.append(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(ref ** 2)))
    print(f"mlx {mx.__version__} bits {bits} {o}x{i}: rel rms error {np.mean(errs):.3e}")
