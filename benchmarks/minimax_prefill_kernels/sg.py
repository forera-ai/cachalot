"""Prototype: MLX's affine_gather_qmm_rhs (sorted rows, BM 16) over slab slots with pair-index (scale, bias)."""
import os

import mlx.core as mx
import numpy as np

import pq

KDIR = pq.KDIR


def header():
    u = open(os.path.join(KDIR, "mlx/backend/metal/kernels/quantized_utils.h")).read()
    u = u.replace("#include <metal_simdgroup>", "").replace("#include <metal_stdlib>", "")
    return (pq.header() + "\n" + u +
            "\nconstant constexpr bool align_M = false;\nconstant constexpr bool align_N = true;\n"
            "constant constexpr bool align_K = true;\n")


def _pick(n_slabs):
    return " : ".join(f"sid == {i} ? slab{i}" for i in range(n_slabs - 1)) + f" : slab{n_slabs - 1}"


_BODY = r"""
  constexpr int BM = 16, BN = 32, BK = 32, WM = 1, WN = 2;
  constexpr int group_size = 64, bits = 3;
  typedef bfloat16_t T;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoaderPair<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const uint simd_group_id = simdgroup_index_in_threadgroup;
  const uint simd_lane_id = thread_index_in_simdgroup;
  uint3 tid = threadgroup_position_in_grid;
  const int M = idx[0];
  const device int* rows_slot = idx + 1;
  const int K = IN_SIZE;
  const int N = OUT_SIZE;
  const int NT = N / BN;
#if TWO
  const bool second = int(tid.x) >= NT;
  if (second) tid.x -= NT;
  device T* y = second ? y2 : y1;
  const int OW = second ? OFF_W3 : OFF_W1;
  const int OP = second ? OFF_P3 : OFF_P1;
  const int OL = second ? OFF_L3 : OFF_L1;
#else
  device T* y = y1;
  const int OW = OFF_W1;
  const int OP = OFF_P1;
  const int OL = OFF_L1;
#endif
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  if (y_row >= M) return;
  const short tgp_bm = short(min(BM, M - y_row));
  const short tgp_bn = BN;
  const device T* xp = x + size_t(y_row) * K;
  y += size_t(y_row) * N + y_col;

  int index;
  short offset;
  int index_next = rows_slot[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (rows_slot[y_row + n] != index) {
        offset_next = n;
        index_next = rows_slot[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);
    const int sid = index / SLAB_SLOTS;
    const device uint8_t* base = PICK;
    base += ulong(index % SLAB_SLOTS) * ulong(RECORD);
    thread mma_t mma_op(simd_group_id, simd_lane_id);
    thread loader_x_t loader_x(xp, K, Xs, simd_group_id, simd_lane_id);
    thread loader_w_t loader_w(
        base + OW + size_t(y_col) * K_w,
        base + OP + size_t(y_col) * K_g,
        (const device T*)(base + OL),
        K, Ws, simd_group_id, simd_lane_id);
    if (tgp_bm == BM) {
      gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
      if (offset_next - offset == BM) {
        mma_op.store_result(y, N);
      } else {
        mma_op.store_result_slice(y, N, short2(0, offset), short2(BN, offset_next));
      }
    } else {
      gemm_loop_unaligned<false, true, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
      mma_op.store_result_slice(y, N, short2(0, offset), short2(BN, offset_next));
    }
  }
"""


class SlabGather:
    """Prefill routed experts over a slab pool: rows sorted by expert, one launch for gate+up, one for down."""

    def __init__(self, pool, inter, dim):
        self.pool, self.inter, self.dim = pool, inter, dim
        n = len(pool.slabs)
        o = pool.offsets
        common = (_BODY.replace("PICK", _pick(n)).replace("SLAB_SLOTS", str(pool.slab_slots))
                  .replace("RECORD", str(pool.record_bytes)))
        names = [f"slab{i}" for i in range(n)] + ["idx", "x"]
        two = ("#define TWO 1\n" + common.replace("OFF_W1", str(o["w1.weight"])).replace("OFF_P1", str(o["w1.pidx"]))
               .replace("OFF_L1", str(o["w1.plut"])).replace("OFF_W3", str(o["w3.weight"]))
               .replace("OFF_P3", str(o["w3.pidx"])).replace("OFF_L3", str(o["w3.plut"])))
        one = ("#define TWO 0\n" + common.replace("OFF_W1", str(o["w2.weight"])).replace("OFF_P1", str(o["w2.pidx"]))
               .replace("OFF_L1", str(o["w2.plut"])))
        h = header()
        self._gu = mx.fast.metal_kernel(name="minimax_slab_gather_gu", input_names=names, output_names=["y1", "y2"],
                                        source=two, header=h, ensure_row_contiguous=True)
        self._dn = mx.fast.metal_kernel(name="minimax_slab_gather_dn", input_names=names, output_names=["y1"],
                                        source=one, header=h, ensure_row_contiguous=True)

    def __call__(self, xs, idx, activation):
        """xs [R, D] rows sorted by expert; idx int32 [R + 1]: R, then each row's slot."""
        R = xs.shape[0]
        slabs = self.pool.kernel_slabs()
        g, u = self._gu(inputs=slabs + [idx, xs], template=[("IN_SIZE", self.dim), ("OUT_SIZE", self.inter)],
                        grid=(64 * 2 * (self.inter // 32), (R + 15) // 16, 1), threadgroup=(64, 1, 1),
                        output_shapes=[(R, self.inter), (R, self.inter)], output_dtypes=[xs.dtype, xs.dtype])
        h = activation(u, g)
        return self._dn(inputs=slabs + [idx, h], template=[("IN_SIZE", self.inter), ("OUT_SIZE", self.dim)],
                        grid=(64 * (self.dim // 32), (R + 15) // 16, 1), threadgroup=(64, 1, 1),
                        output_shapes=[(R, self.dim)], output_dtypes=[xs.dtype])[0]
