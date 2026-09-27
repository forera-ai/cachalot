"""
MiniMax-M3 decode with its experts selected on the GPU (HANDOFF 18.12, S1 of 18.11; the pattern is Splash's
GPU-side MoE over packed expert storage).

The shipped decode layer waits on the host for every MoE layer's routing before its experts run: ~0.4-0.65 ms of
GPU idle per layer, ~22-37 ms a token of an all-hit floor of ~60-72 ms (18.10, 18.12). Here the expert slots live in
slabs (`cache.slots.SlabSlotPool`), the store keeps a (layer, expert) -> slot table (`track_slots`), and one decode
token runs as:

    for each layer i:   A_i  attention, routing, slot lookup in the table      -> submitted (async_eval)
                        check layer i-1 on the host: its slot indices were computed by A_{i-1}, long finished
                        M_i  the routed experts from the slabs, the shared expert, the layer output -> submitted

so the host learns each layer's routing one step late while the GPU keeps working. A layer whose experts were all
resident is done. A layer with a miss (slot -1; its kernel read a placeholder slot) is fixed: the store reads the
missing experts (`get_many`, the same reads and admission as the shipped path), only the missing rows are recomputed,
the layer output is rebuilt, and the attention already run for the next layer is rewound (KV offset - 1) and rerun.

Bit-identical to the shipped path: the kernels are `codes_qmv`'s (MLX's qmv_fast with the in-kernel bias rebuild)
with the slot base computed from the index, and every sum is taken in the same order.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

from cachalot.minimax import codes_qmv

# 0 keeps the shipped per-layer sync; an int so TF_ALTERNATE can flip it per token
GPU_SELECT = int(os.environ.get("CACHALOT_MINIMAX_GPU_SELECT", "1"))


def _slot_base(n_slabs: int, slab_slots: int, record_bytes: int) -> str:
    pick = " : ".join(f"sid == {i} ? slab{i}" for i in range(n_slabs - 1)) + f" : slab{n_slabs - 1}"
    return f"""
  uint j = threadgroup_position_in_grid.z;
  int slot = idx[j];
  if (slot < 0) slot = 0;
  int sid = slot / {slab_slots};
  const device uint8_t* base = {pick};
  base += ulong(slot % {slab_slots}) * ulong({record_bytes});
  uint3 t = threadgroup_position_in_grid;
  t.z = 0;
"""


def _proj(offsets: dict[str, int], p: str, tag: str) -> str:
    return (f"  const device uint32_t* w{tag} = (const device uint32_t*)(base + {offsets[p + '.weight']});\n"
            f"  const device bfloat16_t* s{tag} = (const device bfloat16_t*)(base + {offsets[p + '.scales']});\n"
            f"  const device uint8_t* c{tag} = base + {offsets[p + '.codes']};\n")


_GATE_UP = """
  const int nh = OUT_SIZE / 8;
  if (int(t.y) < nh) {
    qmv_codes_impl<bfloat16_t, 64, 3>(w1, s1, c1, x, g + j * OUT_SIZE, IN_SIZE, OUT_SIZE, t,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
  } else {
    t.y -= nh;
    qmv_codes_impl<bfloat16_t, 64, 3>(w3, s3, c3, x, u + j * OUT_SIZE, IN_SIZE, OUT_SIZE, t,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
  }
"""

_DOWN = """
  qmv_codes_impl<bfloat16_t, 64, 3>(w2, s2, c2, x + j * IN_SIZE, y + j * OUT_SIZE, IN_SIZE, OUT_SIZE, t,
      simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""


class SlabExperts:
    """The routed experts of any layer, from a slab pool and a slot index array on the GPU."""

    def __init__(self, pool, fmt) -> None:
        self.pool = pool
        self.inter = int(fmt.shapes["w1.weight"][0])
        self.dim = int(fmt.shapes["w2.weight"][0])
        n = len(pool.slabs)
        base = _slot_base(n, pool.slab_slots, pool.record_bytes)
        head = codes_qmv._mlx_header() + codes_qmv._IMPL
        names = [f"slab{i}" for i in range(n)] + ["idx", "x"]
        self._gate_up = mx.fast.metal_kernel(
            name="minimax_slab_gate_up", input_names=names, output_names=["g", "u"],
            source=base + _proj(pool.offsets, "w1", "1") + _proj(pool.offsets, "w3", "3") + _GATE_UP,
            header=head, ensure_row_contiguous=True,
        )
        self._down = mx.fast.metal_kernel(
            name="minimax_slab_down", input_names=names, output_names=["y"],
            source=base + _proj(pool.offsets, "w2", "2") + _DOWN, header=head, ensure_row_contiguous=True,
        )

    def __call__(self, x, slots, activation):
        """x [1, D] bf16, slots int32 [k] -> the k experts' outputs [k, D], each exactly `codes_qmv`'s."""
        k = slots.shape[0]
        slabs = self.pool.kernel_slabs()
        g, u = self._gate_up(
            inputs=slabs + [slots, x], template=[("IN_SIZE", self.dim), ("OUT_SIZE", self.inter)],
            grid=(32, 2 * 2 * (self.inter // 8), k), threadgroup=(32, 2, 1),
            output_shapes=[(k, self.inter), (k, self.inter)], output_dtypes=[x.dtype, x.dtype],
        )
        h = activation(u, g)
        return self._down(
            inputs=slabs + [slots, h], template=[("IN_SIZE", self.inter), ("OUT_SIZE", self.dim)],
            grid=(32, 2 * (self.dim // 8), k), threadgroup=(32, 2, 1),
            output_shapes=[(k, self.dim)], output_dtypes=[x.dtype],
        )[0]


class GpuSelectDecoder:
    """One MiniMax decode token with GPU-side expert selection and a one-step-behind host check."""

    def __init__(self, model) -> None:
        self.m = model
        self.store = model.store
        self.index = model.expert_index
        self.layers = model.model.layers
        self.table = self.store.track_slots(len(self.layers), model.config.num_local_experts)
        self.experts = SlabExperts(self.store.pool, model.slot_format)
        self._gpu_table, self._version = None, -1
        self.miss_layers = 0
        self.hit_layers = 0

    def _slot_table(self) -> mx.array:
        version = self.store.slot_table_version
        if self._gpu_table is None or version != self._version:
            with self.store._lock:
                self._gpu_table = mx.array(self.table)
                self._version = version
        return self._gpu_table

    def _route(self, i, layer, r) -> dict:
        moe = layer.block_sparse_moe
        xn = layer.post_attention_layernorm(r)
        scores, orig = moe.route_scores(xn)
        k = moe.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(orig, inds, axis=-1)
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        weights = (weights * moe.routed_scaling_factor).astype(xn.dtype)
        slots = self._slot_table()[i][inds.reshape(-1)]
        return {"i": i, "r": r, "xn": xn, "inds": inds, "weights": weights, "slots": slots, "k": k}

    def _output(self, rec, y=None):
        moe = self.layers[rec["i"]].block_sparse_moe
        if y is None:
            y = self.experts(rec["xn"].reshape(1, -1), rec["slots"], moe.activation)
            rec["y"] = y
            rec["shared"] = moe.shared_experts(rec["xn"])
        routed = (y.reshape(1, 1, rec["k"], -1) * rec["weights"][..., None]).sum(axis=-2)
        return rec["r"] + (routed + rec["shared"])

    def _check(self, rec) -> bool:
        """The host's read of a submitted layer's slots; True when every expert was resident."""
        slots = np.array(rec["slots"])
        routes = np.array(rec["inds"]).reshape(-1)
        rec["routes"], rec["slot_ids"] = routes, slots
        if (slots >= 0).all():
            store = self.store
            with store._lock:
                for e in routes.tolist():
                    key = (rec["i"], e)
                    store._items.move_to_end(key)
                    store.use_counts[key] += 1
                store.cache_hits += len(routes)
            self.hit_layers += 1
            return True
        self.miss_layers += 1
        return False

    def _fix(self, rec):
        """Read a layer's missing experts and recompute only their rows; the corrected layer output."""
        i, routes, slots = rec["i"], rec["routes"], rec["slot_ids"]
        residents = self.store.get_many([self.index[(i, int(e))] for e in routes])
        miss = [j for j in range(len(routes)) if slots[j] < 0]
        moe = self.layers[i].block_sparse_moe
        fresh = self.experts(rec["xn"].reshape(1, -1),
                             mx.array(np.array([residents[j].slot.index for j in miss], np.int32)), moe.activation)
        rows = [fresh[miss.index(j):miss.index(j) + 1] if j in miss else rec["y"][j:j + 1] for j in range(len(routes))]
        return self._output(rec, mx.concatenate(rows, axis=0))

    def forward(self, token: int, cache) -> mx.array:
        """Logits [1, vocab] for one decode token; `cache` advances by one position, as `_forward` does."""
        inner = self.m.model.model
        h = inner.embed_tokens(mx.array([[token]], dtype=mx.int32))
        i, pending, n = 0, None, len(self.layers)
        while True:
            if i == n:
                if pending is not None and not self._check(pending):
                    h = self._fix(pending)
                h = inner.norm(h)
                if self.m.config.tie_word_embeddings:
                    return inner.embed_tokens.as_linear(h)[:, -1, :]
                return self.m.model.lm_head(h)[:, -1, :]
            layer = self.layers[i]
            r = h + layer.self_attn(layer.input_layernorm(h), None, cache[i])
            if layer.is_sparse:
                rec = self._route(i, layer, r)
                mx.async_eval(rec["slots"], rec["inds"])
            else:
                rec = None
                out = r + layer.mlp(layer.post_attention_layernorm(r))
                mx.async_eval(out)
            if pending is not None:
                if not self._check(pending):
                    p = pending["i"]
                    for j in range(p + 1, i + 1):
                        cache[j].offset -= 1
                    h, i, pending = self._fix(pending), p + 1, None
                    continue
                pending = None
            if rec is not None:
                out = self._output(rec)
                mx.async_eval(out)
                pending = rec
            h = out
            i += 1
