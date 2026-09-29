"""Prototype: MLX's qmv_wide and qmm_t with the (scale, bias) pair looked up from a byte index (no rebuild)."""
import os
import re
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/hamedprooshani/Projects/deepseek-v41-mac/src")
from cachalot.minimax import codes_qmv as cq  # noqa: E402

KDIR = os.path.join(os.path.dirname(mx.__file__), "include")


def _inline(rel, seen):
    if rel in seen:
        return ""
    seen.add(rel)
    out = []
    for line in open(os.path.join(KDIR, rel)).read().split("\n"):
        m = re.match(r'#include "(mlx/backend/metal/kernels/[^"]+)"', line)
        if m:
            out.append(_inline(m.group(1), seen))
        elif line.startswith("#pragma once"):
            continue
        else:
            out.append(line)
    return "\n".join(out)


def header():
    seen = set()
    steel = _inline("mlx/backend/metal/kernels/steel/gemm/mma.h", seen) + "\n" + \
        _inline("mlx/backend/metal/kernels/steel/gemm/loader.h", seen)
    q = open(os.path.join(KDIR, "mlx/backend/metal/kernels/quantized.h")).read()
    # the helpers and the two impls we copy, not the [[kernel]] wrappers
    start = q.index("using namespace metal;")
    end = q.index("template <typename T, int group_size, int bits, int D>\nMETAL_FUNC void qmv_quad_impl(")
    helpers = q[start:end]
    helpers = helpers.replace("#define MLX_MTL_CONST static constant constexpr const",
                              "#ifndef MLX_MTL_CONST\n#define MLX_MTL_CONST static constant constexpr const\n#endif")
    loader = q[q.index("template <\n    typename T,\n    short BROWS,"):q.index("template <typename T, int group_size, int bits, int D>\nMETAL_FUNC void qmv_quad_impl(")]
    loader = (loader.replace("struct QuantizedBlockLoader {", "struct QuantizedBlockLoaderPair {").replace("  QuantizedBlockLoader(\n", "  QuantizedBlockLoaderPair(\n")
              .replace("  const device T* scales;\n  const device T* biases;", "  const device uint8_t* scales;\n  const device T* biases;")
              .replace("      const device T* scales_,", "      const device uint8_t* scales_,")
              .replace("        biases(biases_ + bi * src_ld / group_size) {}", "        biases(biases_) {}")
              .replace("    T scale = *scales;\n    T bias = *biases;",
                       "    const int pi_ = *scales;\n    T scale = biases[2 * pi_];\n    T bias = biases[2 * pi_ + 1];")
              .replace("          biases++;\n", "").replace("        biases++;\n", "")
              .replace("      biases += group_stride;\n", ""))
    # remove the loader from helpers (it is inside [start:end]); keep original too (harmless)
    wide_s = q.index("template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>\nMETAL_FUNC void qmv_wide_impl(")
    wide_e = q.index("template <typename T, const int group_size, const int bits>\nMETAL_FUNC void qvm_impl(")
    wide = (q[wide_s:wide_e].replace("qmv_wide_impl(", "qmv_wide_pair_impl(")
            .replace("    const device T* scales,\n    const device T* biases,",
                     "    const device uint8_t* scales,\n    const device T* biases,")
            .replace("const constant int& ", "const int ")
            .replace("  const device T* srow = scales + row * in_vec_size_g;\n  const device T* brow = biases + row * in_vec_size_g;",
                     "  const device uint8_t* srow = scales + row * in_vec_size_g;")
            .replace("    U scale = srow[g];\n    U bias = brow[g];",
                     "    const int pi_ = srow[g];\n    U scale = biases[2 * pi_];\n    U bias = biases[2 * pi_ + 1];"))
    mm_s = q.index("template <\n    typename T,\n    const int group_size,\n    const int bits,\n    const bool aligned_N,\n    const int BM = 32,")
    mm_e = q.index("template <", mm_s + 10)
    mm = (q[mm_s:mm_e].replace("qmm_t_impl(", "qmm_t_pair_impl(")
          .replace("    const device T* scales,\n    const device T* biases,",
                   "    const device uint8_t* scales,\n    const device T* biases,")
          .replace("QuantizedBlockLoader<", "QuantizedBlockLoaderPair<")
          .replace("  biases += y_col * K_g;\n", "")
          .replace("const constant int& ", "const int "))
    assert "biases += y_col" not in mm and "QuantizedBlockLoaderPair<" in mm
    return steel + "\n" + helpers + "\n" + loader + "\n" + wide + "\n" + mm


_WIDE = r"""
  qmv_wide_pair_impl<bfloat16_t, 64, 3, NV, 8>(
      w, s, l, x, y, IN_SIZE, OUT_SIZE, m[0],
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_MM = r"""
  threadgroup bfloat16_t Xs[32 * 40];
  threadgroup bfloat16_t Ws[32 * 40];
  qmm_t_pair_impl<bfloat16_t, 64, 3, true, 32, 32, 32>(
      w, s, l, x, y, Xs, Ws, IN_SIZE, OUT_SIZE, m[0], IN_SIZE,
      threadgroup_position_in_grid, thread_index_in_threadgroup,
      simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_K = {}


def kern(name):
    if name not in _K:
        src = _WIDE if name == "wide" else _MM
        _K[name] = mx.fast.metal_kernel(name=f"minimax_pair_{name}", input_names=["w", "s", "l", "x", "m"],
                                        output_names=["y"], source=src, header=header(),
                                        ensure_row_contiguous=True)
    return _K[name]


def wide(x, w, pidx, plut, nv=4):
    M, out = x.shape[0], w.shape[0]
    return kern("wide")(inputs=[w, pidx, plut, x, mx.array([M], dtype=mx.int32)],
                        template=[("IN_SIZE", x.shape[-1]), ("OUT_SIZE", out), ("NV", nv)],
                        grid=(32 * ((M + nv - 1) // nv), 2 * (out // 8), 1), threadgroup=(32, 2, 1),
                        output_shapes=[(M, out)], output_dtypes=[x.dtype])[0]


def mm(x, w, pidx, plut):
    M, out = x.shape[0], w.shape[0]
    return kern("mm")(inputs=[w, pidx, plut, x, mx.array([M], dtype=mx.int32)],
                      template=[("IN_SIZE", x.shape[-1]), ("OUT_SIZE", out)],
                      grid=(128 * (out // 32), (M + 31) // 32, 1), threadgroup=(128, 1, 1),
                      output_shapes=[(M, out)], output_dtypes=[x.dtype])[0]


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    for out, inp in ((3072, 6144), (6144, 3072)):
        w = mx.array(rng.integers(0, 2**32, size=(out, inp * 3 // 32), dtype=np.uint32))
        pidx = mx.array(rng.integers(0, 200, size=(out, inp // 64), dtype=np.uint8))
        s = (rng.random(256) * 0.01 + 0.001).astype(np.float32)
        pl = np.zeros(512, np.uint16)
        pl[0::2] = s.view(np.uint32) >> 16
        pl[1::2] = (-5 * s).astype(np.float32).view(np.uint32) >> 16
        plut = mx.array(pl).view(mx.bfloat16)
        S, B = cq.rebuild_pair(pidx, plut)
        wide_eq, mm_eq = [], []
        for M in list(range(2, 41)) + [64, 100, 128, 300]:
            x = mx.random.normal((M, inp)).astype(mx.bfloat16)
            ref = mx.quantized_matmul(x, w, S, B, transpose=True, group_size=64, bits=3)
            if bool(mx.array_equal(wide(x, w, pidx, plut), ref).item()):
                wide_eq.append(M)
            if bool(mx.array_equal(mm(x, w, pidx, plut), ref).item()):
                mm_eq.append(M)
        print(out, inp, "wide==MLX at", wide_eq, "\n   mm==MLX at", mm_eq, flush=True)
