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

Since 0.33.0 (HANDOFF 18.14) a missing expert's rows are zero, so the discarded attention and routing of layer i+1
are those of layer i's output without its misses, nearly the real ones: while layer i's reads run, the host reads
the non-resident experts that routing names, and one layer further (SPEC_PREFETCH, SPEC_DEPTH).

Bit-identical to the shipped path: the kernels are `codes_qmv`'s (MLX's qmv_fast with the in-kernel bias rebuild)
with the slot base computed from the index, and every sum is taken in the same order.
"""

from __future__ import annotations

import os
from functools import partial

import mlx.core as mx
import numpy as np

from cachalot.minimax import codes_qmv

# 0 keeps the shipped per-layer sync; an int so TF_ALTERNATE can flip it per token
GPU_SELECT = int(os.environ.get("CACHALOT_MINIMAX_GPU_SELECT", "1"))
# S1b: A_i also scores layer i+PREFETCH_AHEAD's router on this layer's residual; the host, when it checks layer i,
# starts a read of the best non-resident of the top PREFETCH_TOPK (the store's prefetch path: free transient slots,
# no eviction). Ints so TF_ALTERNATE can flip them; 0 turns the prefetch off.
PREFETCH_TOPK = int(os.environ.get("CACHALOT_MINIMAX_SELECT_PREFETCH", "0"))
PREFETCH_AHEAD = int(os.environ.get("CACHALOT_MINIMAX_SELECT_PREFETCH_AHEAD", "1"))
# HANDOFF 18.14: when layer i misses, the GPU has already run layer i+1's attention and routing on layer i's output
# without the missing experts (their rows are zero). That routing is nearly the real one, and the host holds it while
# layer i's reads run: up to SPEC_PREFETCH of layer i+1's non-resident experts are read right after layer i's own
# (the store's prefetch path). An int so TF_ALTERNATE can flip it; 0 turns it off.
SPEC_PREFETCH = int(os.environ.get("CACHALOT_MINIMAX_SPEC_PREFETCH", "8"))
# 0 (default): the speculated reads are issued as soon as layer i's own are submitted, sharing the drive with them
# (a deeper queue: 3.30 -> 3.10 ms a miss); 1: only after layer i's reads are in, like the 0.28.0 prediction.
SPEC_AFTER_DEMAND = int(os.environ.get("CACHALOT_MINIMAX_SPEC_AFTER_DEMAND", "0"))
# 2 (default): also run layer i+1's hit experts and layer i+2's attention and routing on the speculative output and
# read layer i+2's predicted misses too (-3 % against 1; 3 was +1.7 %)
SPEC_DEPTH = int(os.environ.get("CACHALOT_MINIMAX_SPEC_DEPTH", "2"))


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
    if p + ".pidx" in offsets:
        # a pair slot (HANDOFF 18.13, M23b): a byte per group and the projection's (scale, bias) table
        return (f"  const device uint32_t* w{tag} = (const device uint32_t*)(base + {offsets[p + '.weight']});\n"
                f"  const device uint8_t* s{tag} = base + {offsets[p + '.pidx']};\n"
                f"  const device bfloat16_t* l{tag} = (const device bfloat16_t*)(base + {offsets[p + '.plut']});\n")
    if p + ".sidx" in offsets:
        # a scale-index slot (HANDOFF 18.13): a byte per group and the projection's table
        return (f"  const device uint32_t* w{tag} = (const device uint32_t*)(base + {offsets[p + '.weight']});\n"
                f"  const device uint8_t* s{tag} = base + {offsets[p + '.sidx']};\n"
                f"  const device bfloat16_t* l{tag} = (const device bfloat16_t*)(base + {offsets[p + '.lut']});\n"
                f"  const device uint8_t* c{tag} = base + {offsets[p + '.codes']};\n")
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
  if (idx[j] < 0) {
    // a missing expert (HANDOFF 18.14): zero rows, so the speculative layer output leaves it out
    if (thread_index_in_simdgroup == 0) {
      const int row0 = t.y * 8 + simdgroup_index_in_threadgroup * 4;
      for (int r = 0; r < 4; r++) y[j * OUT_SIZE + row0 + r] = static_cast<bfloat16_t>(0.0f);
    }
    return;
  }
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
        sidx, pair = "w1.sidx" in pool.offsets, "w1.pidx" in pool.offsets
        if pair:
            head = codes_qmv._mlx_header() + codes_qmv._IMPL_PAIR
            gate_up = (_GATE_UP.replace("qmv_codes_impl", "qmv_pair_impl").replace("(w1, s1, c1", "(w1, s1, l1")
                       .replace("(w3, s3, c3", "(w3, s3, l3"))
            down = _DOWN.replace("qmv_codes_impl", "qmv_pair_impl").replace("(w2, s2, c2", "(w2, s2, l2")
        elif sidx:
            head = codes_qmv._mlx_header() + codes_qmv._IMPL_SIDX
            gate_up = (_GATE_UP.replace("qmv_codes_impl", "qmv_sidx_impl").replace("(w1, s1, c1", "(w1, s1, l1, c1")
                       .replace("(w3, s3, c3", "(w3, s3, l3, c3"))
            down = _DOWN.replace("qmv_codes_impl", "qmv_sidx_impl").replace("(w2, s2, c2", "(w2, s2, l2, c2")
        else:
            head, gate_up, down = codes_qmv._mlx_header() + codes_qmv._IMPL, _GATE_UP, _DOWN
        tag = "_pair" if pair else "_sidx" if sidx else ""
        names = [f"slab{i}" for i in range(n)] + ["idx", "x"]
        self._gate_up = mx.fast.metal_kernel(
            name="minimax_slab_gate_up" + tag, input_names=names, output_names=["g", "u"],
            source=base + _proj(pool.offsets, "w1", "1") + _proj(pool.offsets, "w3", "3") + gate_up,
            header=head, ensure_row_contiguous=True,
        )
        self._down = mx.fast.metal_kernel(
            name="minimax_slab_down" + tag, input_names=names, output_names=["y"],
            source=base + _proj(pool.offsets, "w2", "2") + down, header=head, ensure_row_contiguous=True,
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
        self.spec_predicted = 0

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
        rec = {"i": i, "r": r, "xn": xn, "inds": inds, "weights": weights, "slots": slots, "k": k, "pred": None}
        t = i + PREFETCH_AHEAD
        if PREFETCH_TOPK > 0 and t < len(self.layers) and self.layers[t].is_sparse:
            nxt = self.layers[t]
            rec["pred"] = nxt.block_sparse_moe.route_scores(nxt.post_attention_layernorm(r))[0]
            rec["pred_layer"] = t
        return rec

    def _predicted(self, rec):
        """The predicted layer's top-PREFETCH_TOPK experts, best first (read from the sync A_i already had)."""
        if rec["pred"] is None:
            return None
        sc = np.array(rec["pred"]).reshape(-1)
        n = min(PREFETCH_TOPK, sc.shape[0])
        top = np.argpartition(-sc, n - 1)[:n]
        t = rec["pred_layer"]
        return [self.index[(t, int(e))] for e in top[np.argsort(-sc[top])]]

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
                if store._inflight:
                    # get_many is not called on an all-hit layer: advance the decode walk here so stale
                    # predictions still expire (HANDOFF 18.12, S1d)
                    store._advance_decode_pass_locked(rec["i"])
                    store._sweep_inflight_locked(keep=set())
            self.hit_layers += 1
            pred = self._predicted(rec)
            if pred:
                store.prefetch_decode(pred)
            return True
        self.miss_layers += 1
        return False

    def _speculated(self, spec):
        """Layer i+1's non-resident experts by the routing the GPU ran on layer i's output without its misses
        (HANDOFF 18.14); called once layer i's own reads are submitted, when that routing has long finished."""
        slots = np.array(spec["slots"])
        routes = np.array(spec["inds"]).reshape(-1)
        t = spec["i"]
        self.spec_predicted += int((slots < 0).sum())
        return [self.index[(t, int(e))] for e in routes[slots < 0][:SPEC_PREFETCH]]

    def _speculate_deeper(self, spec):
        """SPEC_DEPTH > 1: layer i+1's predicted reads first, then its hit experts and the next layer's attention and
        routing on that speculative output (the KV write rewound at once) and that layer's predicted reads, up to
        SPEC_DEPTH layers ahead. Runs on the calling thread once layer i's own reads are submitted, so none of this
        graph building delays them."""
        cache = spec["cache"]
        for d in range(SPEC_DEPTH):
            ents = self._speculated(spec)
            if ents:
                self.store.prefetch_decode(ents, SPEC_PREFETCH)
            t = spec["i"] + 1
            if d + 1 == SPEC_DEPTH or t >= len(self.layers):
                break
            o1 = self._output(spec)
            nl = self.layers[t]
            r2 = o1 + nl.self_attn(nl.input_layernorm(o1), None, cache[t])
            cache[t].offset -= 1
            if not nl.is_sparse:
                break
            spec = self._route(t, nl, r2)
            mx.async_eval(spec["slots"], spec["inds"])
        return None

    def _fix(self, rec, spec=None):
        """Read a layer's missing experts and recompute only their rows; the corrected layer output."""
        i, routes, slots = rec["i"], rec["routes"], rec["slot_ids"]
        pred = self._predicted(rec)
        limit, after = None, None
        if pred is None and spec is not None and SPEC_PREFETCH > 0:
            pred = partial(self._speculate_deeper if "cache" in spec else self._speculated, spec)
            limit, after = SPEC_PREFETCH, bool(SPEC_AFTER_DEMAND)
        residents = self.store.get_many([self.index[(i, int(e))] for e in routes], prefetch=pred,
                                        prefetch_limit=limit, prefetch_after=after)
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
                mx.async_eval(rec["slots"], rec["inds"], *([rec["pred"]] if rec["pred"] is not None else []))
            else:
                rec = None
                out = r + layer.mlp(layer.post_attention_layernorm(r))
                mx.async_eval(out)
            if pending is not None:
                if not self._check(pending):
                    p = pending["i"]
                    for j in range(p + 1, i + 1):
                        cache[j].offset -= 1
                    spec = rec if rec is not None and rec["i"] == p + 1 else None
                    if spec is not None and SPEC_DEPTH > 1 and i + 1 < n:
                        spec = {**spec, "cache": cache}
                    h, i, pending = self._fix(pending, spec), p + 1, None
                    continue
                pending = None
            if rec is not None:
                out = self._output(rec)
                mx.async_eval(out)
                pending = rec
            h = out
            i += 1
