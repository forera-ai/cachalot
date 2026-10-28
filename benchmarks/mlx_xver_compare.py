"""Bitwise compare of two ops.npz dumps: ~/venvs/deepseek-v41/bin/python benchmarks/mlx_xver_compare.py A/ops.npz B/ops.npz (HANDOFF section 18.223)"""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
same = diff = 0
for k in sorted(set(a.files) | set(b.files)):
    if k not in a.files or k not in b.files:
        print(f"MISSING  {k}")
        continue
    x, y = a[k], b[k]
    if x.shape != y.shape:
        print(f"SHAPE    {k} {x.shape} {y.shape}")
        diff += 1
        continue
    if x.tobytes() == y.tobytes():
        same += 1
        continue
    diff += 1
    if np.issubdtype(x.dtype, np.floating):
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        ref = np.abs(x.astype(np.float64)).max() or 1.0
        print(f"DIFF     {k:42s} elems {np.mean(d > 0) * 100:6.2f} %  max|d| {d.max():.3e}  rel {d.max() / ref:.2e}")
    else:
        print(f"DIFF     {k:42s} elems {np.mean(x != y) * 100:6.2f} %  (int/bool)  a={x.ravel()[:3]} b={y.ravel()[:3]}")
print(f"identical {same}, different {diff}")
