"""
MiniMax-M3 expert slots that hold 4-bit bias codes instead of bf16 biases (HANDOFF 18.10, M13b of 18.6).

Every group bias of this checkpoint is bf16_rne(float32(k) * float32(scale)) for k in -7..-3 (18.4), so a slot can
keep the code k + 7 (a nibble, two groups per byte) instead of the bias: 0.42 MiB instead of 1.69 MiB per expert,
22.35 MiB a slot instead of 23.62, ~5.7 % more resident experts in the same budget.

Decode (one bf16 row) runs MLX's own 3-bit `qmv_fast` kernel, copied at import time from the installed MLX headers
into an `mx.fast.metal_kernel`, with one change: the group bias is rebuilt from the code in the kernel, with the
bank's rounding. Gate and up share their input and run in one launch. Against `mx.quantized_matmul` on the
real biases this is bit-identical (`self_check`, run when the model loads; a mismatch, e.g. after an MLX upgrade
changed the kernel, turns the kernel off and every matmul goes through the rebuild below instead).

Everything else (prefill chunks, any other shape) rebuilds the bf16 biases on the GPU with one small kernel per
expert (the same rounding, bit-identical) and calls `mx.quantized_matmul` as before.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

K_OFFSET = 7  # nibble c means k = c - 7
GROUP, BITS = 64, 3

# 0 forces the rebuild path everywhere (an int, so TF_ALTERNATE can flip it)
KERNEL = int(os.environ.get("CACHALOT_MINIMAX_CODES_KERNEL", "1"))


def _mlx_header() -> str:
    path = os.path.join(os.path.dirname(mx.__file__), "include/mlx/backend/metal/kernels/quantized.h")
    lines = open(path).read().split("\n")
    # SIMD_SIZE, get_pack_factor, get_bytes_per_pack, load_vector, qdot: MLX's text, unchanged
    start = next(i for i, s in enumerate(lines) if s.startswith("using namespace metal"))
    end = next(i for i, s in enumerate(lines) if s.startswith("inline U qdot(")) + 1
    while lines[end] != "}":
        end += 1
    return "\n".join(lines[start:end + 1]) + "\n"


_IMPL = r"""
constant constexpr int K_OFFSET = 7;

// MLX's qmv_fast_impl (quantized.h) with the group bias rebuilt from a 4-bit code as the bank rounds it.
template <typename T, int group_size, int bits>
METAL_FUNC void qmv_codes_impl(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* codes,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  int goff = out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  scales += goff;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;

      U s = sl[0];
      int g = goff + row * in_vec_size_g;
      int c = (codes[g >> 1] >> ((g & 1) * 4)) & 0xF;
      float p = float(c - K_OFFSET) * float(sl[0]);
      uint u = as_type<uint>(p);
      ushort hb = ushort((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
      U b = static_cast<U>(as_type<T>(hb));
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    goff += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}
"""

_ONE = r"""
  qmv_codes_impl<bfloat16_t, 64, 3>(
      w, s, c, x, y, IN_SIZE, OUT_SIZE,
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

# gate and up: the grid's first half of threadgroup rows computes w1, the second half w3, same x
_TWO = r"""
  const int nh = OUT_SIZE / 8;
  uint3 t = threadgroup_position_in_grid;
  if (int(t.y) < nh) {
    qmv_codes_impl<bfloat16_t, 64, 3>(
        w1, s1, c1, x, g, IN_SIZE, OUT_SIZE, t, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
  } else {
    t.y -= nh;
    qmv_codes_impl<bfloat16_t, 64, 3>(
        w3, s3, c3, x, u, IN_SIZE, OUT_SIZE, t, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
  }
"""

# biases for the other paths: bf16_rne(float32(code - 7) * float32(scale)), one thread per group
_REBUILD = r"""
  uint g = thread_position_in_grid.x;
  int c = (c_in[g >> 1] >> ((g & 1) * 4)) & 0xF;
  float p = float(c - 7) * float(s_in[g]);
  uint u = as_type<uint>(p);
  b_out[g] = as_type<bfloat16_t>(ushort((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16));
"""

_KERNELS: dict[str, object] = {}


def _kernel(name: str):
    k = _KERNELS.get(name)
    if k is None:
        if name == "rebuild":
            k = mx.fast.metal_kernel(
                name="minimax_bias_rebuild", input_names=["s_in", "c_in"], output_names=["b_out"],
                source=_REBUILD, ensure_row_contiguous=True,
            )
        else:
            header = _mlx_header() + _IMPL
            if name == "one":
                k = mx.fast.metal_kernel(
                    name="minimax_qmv_codes", input_names=["w", "s", "c", "x"], output_names=["y"],
                    source=_ONE, header=header, ensure_row_contiguous=True,
                )
            else:
                k = mx.fast.metal_kernel(
                    name="minimax_qmv_codes_gate_up", input_names=["w1", "s1", "c1", "w3", "s3", "c3", "x"],
                    output_names=["g", "u"], source=_TWO, header=header, ensure_row_contiguous=True,
                )
        _KERNELS[name] = k
    return k


def qmv(x, w, s, c):
    """x [1, in] (bf16) @ dequant(w, s, codes).T: MLX's qmv_fast with the bias rebuilt in the kernel."""
    out = w.shape[0]
    return _kernel("one")(
        inputs=[w, s, c, x],
        template=[("IN_SIZE", x.shape[-1]), ("OUT_SIZE", out)],
        # MLX's dispatch: threadgroup (32, 2, 1), grid of threadgroups (M, out / 8, 1)
        grid=(32, 2 * (out // 8), 1), threadgroup=(32, 2, 1),
        output_shapes=[(1, out)], output_dtypes=[x.dtype],
    )[0]


def gate_up(x, p1, p3):
    """(x @ w1.T, x @ w3.T) in one launch, each exactly as `qmv`."""
    out = p1[0].shape[0]
    return _kernel("two")(
        inputs=[p1[0], p1[1], p1[2], p3[0], p3[1], p3[2], x],
        template=[("IN_SIZE", x.shape[-1]), ("OUT_SIZE", out)],
        grid=(32, 2 * 2 * (out // 8), 1), threadgroup=(32, 2, 1),
        output_shapes=[(1, out), (1, out)], output_dtypes=[x.dtype, x.dtype],
    )


def rebuild_biases(scales, codes):
    """The bf16 biases a codes slot stands for, on the GPU (bit-identical to the bank's CPU table)."""
    n = scales.size
    return _kernel("rebuild")(
        inputs=[scales.reshape(-1), codes],
        grid=(n, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(n,)], output_dtypes=[mx.bfloat16],
    )[0].reshape(scales.shape)


def usable(x) -> bool:
    """The kernel handles one bf16 row with the shapes MLX's qmv_fast takes (in % 512 == 0, out % 8 == 0)."""
    return bool(KERNEL) and _OK and x.ndim == 2 and x.shape[0] == 1 and x.dtype == mx.bfloat16


# -- CPU side: codes for a slot ---------------------------------------------------------------------------------

def _nibble_table(k_base: int) -> np.ndarray:
    """table[b] for a byte b of four 2-bit bank codes (k = c + k_base): two bytes of nibbles (k + 7), little-endian."""
    t = np.empty(256, np.uint16)
    for b in range(256):
        c = [((b >> (2 * i)) & 3) + k_base + K_OFFSET for i in range(4)]
        t[b] = (c[0] | (c[1] << 4)) | ((c[2] | (c[3] << 4)) << 8)
    return t


_TABLES: dict[int, np.ndarray] = {}


def nibbles_from_packed(packed: np.ndarray, k_base: int, out: np.ndarray) -> None:
    """Bank 2-bit codes (4 per byte) into slot nibbles (2 per byte), `out` a uint8 view of 2 x len(packed) bytes."""
    t = _TABLES.get(k_base)
    if t is None:
        t = _TABLES[k_base] = _nibble_table(k_base)
    np.take(t, packed, out=out.view(np.uint16))


def nibbles_from_biases(scales: np.ndarray, biases: np.ndarray, out: np.ndarray) -> None:
    """Nibbles for bf16 scale and bias bit patterns (a raw record); raises if any bias is not k * scale."""
    s = (scales.astype(np.uint32) << 16).view(np.float32)
    b = (biases.astype(np.uint32) << 16).view(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.round(b / s)
    k = np.where(s == 0, -K_OFFSET, k)
    if not (np.all(np.isfinite(k)) and k.min() >= -K_OFFSET and k.max() <= 15 - K_OFFSET):
        raise ValueError("bias outside the 4-bit code range")
    k = k.astype(np.int32)
    u = (s * (k.astype(np.float32))).view(np.uint32)
    if not np.array_equal(((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16), biases):
        raise ValueError("a bias is not bf16(k * scale)")
    c = (k + K_OFFSET).astype(np.uint8)
    np.bitwise_or(c[0::2], c[1::2] << 4, out=out)


# -- the load-time check ----------------------------------------------------------------------------------------

_OK = True


def _bits(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def self_check(shapes=((3072, 6144), (6144, 3072)), trials: int = 2) -> bool:
    """The kernels against `mx.quantized_matmul` on random experts of the given (out, in) shapes, bit for bit.
    Sets the module's verdict: False sends every matmul through `rebuild_biases` + `mx.quantized_matmul`."""
    global _OK
    rng = np.random.default_rng(0)
    ok = True
    for out, inp in shapes:
        for _ in range(trials):
            w = mx.array(rng.integers(0, 2**32, size=(out, inp * BITS // 32), dtype=np.uint32))
            sc = _bits((np.abs(rng.normal(0, 0.004, size=(out, inp // GROUP))) + 1e-4).astype(np.float32))
            k = rng.integers(-7, -2, size=sc.shape)
            b = _bits((sc.astype(np.uint32) << 16).view(np.float32) * k.astype(np.float32))
            flat = (k + K_OFFSET).astype(np.uint8).reshape(-1)
            c = mx.array((flat[0::2] | (flat[1::2] << 4)).astype(np.uint8))
            s = mx.array(sc).view(mx.bfloat16)
            bias = mx.array(b).view(mx.bfloat16)
            x = mx.array(rng.normal(0, 1, size=(1, inp)).astype(np.float32)).astype(mx.bfloat16)
            ref = mx.quantized_matmul(x, w, s, bias, transpose=True, group_size=GROUP, bits=BITS)
            got = qmv(x, w, s, c)
            g, u = gate_up(x, (w, s, c), (w, s, c))
            rb = rebuild_biases(s, c)
            ok &= all(bool(mx.array_equal(a, ref).item()) for a in (got, g, u))
            ok &= bool(mx.array_equal(rb, bias).item())
    _OK = ok
    return ok
