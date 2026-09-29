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
# HANDOFF 18.19: the routing after the gate matmul (sigmoid, correction bias, top-k, weights, slot lookup) as one
# kernel instead of ~12 dependent ones on the path to the layer's experts; bit-identical. -0.65 ms a token in one
# process, nothing measurable through the server path, so off by default. An int so TF_ALTERNATE can flip it.
FUSED_ROUTE = int(os.environ.get("CACHALOT_MINIMAX_FUSED_ROUTE", "0"))
# HANDOFF 18.21: a missing expert whose share of the layer's routing weight is below MISS_DROP is not read; its row
# stays zero and the other experts' weights are rescaled to the same total (top-k over the rest). NOT bit-identical:
# 0 (default) keeps the exact path. A float so TF_ALTERNATE can flip it; MISS_DROP_ARMED makes the host see the
# weights (a float32 copy inside the sync) so a TF_ALTERNATE arm at 0 costs the same.
MISS_DROP = float(os.environ.get("CACHALOT_MINIMAX_MISS_DROP", "0"))
MISS_DROP_ARMED = MISS_DROP > 0 or os.environ.get("CACHALOT_MINIMAX_MISS_DROP_ARMED", "0") == "1"
# MISS_SUB > 0: a dropped expert is replaced by the best resident one of the next MISS_SUB by selection score (its own
# sigmoid weight, all four renormalised as the router would), instead of leaving the layer with k - 1 experts
MISS_SUB = int(os.environ.get("CACHALOT_MINIMAX_MISS_SUB", "0"))
# with MISS_DROP_ARMED, every missing expert's weight share is appended here (instruments only)
MISS_SHARES: list = []
# HANDOFF 18.23: the same rule in a prefill chunk, but only where it saves a read: a missing expert is left unread
# when every row routed to it has it under PREFILL_MISS_DROP of the row's weight and a resident runner-up among the
# next MISS_SUB ranks (or 4 when MISS_SUB is 0); only those rows change. NOT bit-identical; 0 (default) is exact.
# Experts routed to more than PREFILL_SUB_MAX_ROWS rows are always read (the rule rarely holds for all of them).
PREFILL_MISS_DROP = float(os.environ.get("CACHALOT_MINIMAX_PREFILL_MISS_DROP", "0"))
PREFILL_SUB_MAX_ROWS = int(os.environ.get("CACHALOT_MINIMAX_PREFILL_SUB_MAX_ROWS", "32"))
PREFILL_SUB_STATS = {"missing": 0, "skipped": 0, "rows": 0}

_ROUTE_TAIL = """
    // One threadgroup of E threads. Expert t's selection score and its rank among all E: MLX's argpartition of
    // -score is its stable merge sort, so the larger score comes first and a tie goes to the lower index. The top K
    // are written in rank order, their weights summed in order (as MLX's sum over four), divided, scaled and cast.
    const uint t = thread_position_in_threadgroup.x;
    const int E = meta[0];
    const int K = meta[1];
    const int layer = meta[2];
    threadgroup float neg[256];
    threadgroup float sig[256];
    threadgroup int top[8];
    const float x = raw[t];
    const float y = 1 / (1 + metal::exp(metal::abs(x)));  // MLX's Sigmoid
    const float s = (x < 0) ? y : 1 - y;
    const float me = -(s + bias[t]);
    neg[t] = me;
    sig[t] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int rank = 0;
    for (int j = 0; j < E; ++j) {
        const float o = neg[j];
        rank += (o < me || (o == me && j < int(t))) ? 1 : 0;
    }
    if (rank < K) top[rank] = int(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float sum = sig[top[0]];
        for (int r = 1; r < K; ++r) sum += sig[top[r]];
        const float den = sum + 1e-20f;
        for (int r = 0; r < K; ++r) {
            inds[r] = uint(top[r]);
            weights[r] = static_cast<bfloat16_t>((sig[top[r]] / den) * scale[0]);
            slots[r] = table[layer * E + top[r]];
        }
    }
"""
_route_tail = None


def route_tail(raw: mx.array, bias: mx.array, table: mx.array, layer: int, k: int, scale: float):
    """`raw` the gate's float32 output [1, 1, E]; `bias` float32 [E]; `table` int32 [layers, E]. Returns the
    routed indices [1, 1, k] (uint32), their bfloat16 weights [1, 1, k] and slots [k] (int32), bit for bit what
    `_route`'s MLX ops give."""
    global _route_tail
    if _route_tail is None:
        _route_tail = mx.fast.metal_kernel(name="minimax_route_tail", input_names=["raw", "bias", "table", "meta", "scale"],
                                           output_names=["inds", "weights", "slots"], source=_ROUTE_TAIL)
    e = raw.shape[-1]
    meta = mx.array([e, k, layer], dtype=mx.int32)
    return _route_tail(inputs=[raw.reshape(-1), bias, table, meta, mx.array([scale], mx.float32)],
                       grid=(e, 1, 1), threadgroup=(e, 1, 1),
                       output_shapes=[(1, 1, k), (1, 1, k), (k,)],
                       output_dtypes=[mx.uint32, mx.bfloat16, mx.int32])


def miss_plan(w, miss, threshold, window, sc, og, resident, routes, scale):
    """HANDOFF 18.21: which of a layer's missing experts to skip, and what replaces them.

    `w` the routed weights as the layer computed them (float32 [k], summing to the scaling factor), `miss` the
    positions whose expert is not resident, `threshold` the weight share under which a missing expert is not read.
    With `window` > 0 and the selection scores `sc`, plain sigmoid scores `og` and a residency mask over all experts,
    a skipped expert is replaced by the best-scored resident expert among the next `window` ranks not already
    routed (the heaviest skipped one first), and the weights are the router's own over the new set; without a
    replacement it is left out and the others are rescaled to the same total. Returns None when nothing changes,
    else (left out positions, {position: replacement expert}, routes, weights float32 [k])."""
    total = float(w.sum())
    skip = [j for j in miss if w[j] < threshold * total]
    if not skip:
        return None
    subs = {}
    routes = np.array(routes).copy()
    if window > 0 and sc is not None:
        chosen = set(int(e) for e in routes)
        order = np.argsort(-sc, kind="stable")[:len(routes) + window]
        cand = [int(e) for e in order if int(e) not in chosen and resident[int(e)]]
        for j in sorted(skip, key=lambda j: -w[j]):
            if cand:
                subs[j] = cand.pop(0)
        for j, e in subs.items():
            routes[j] = e
    gone = [j for j in skip if j not in subs]
    if subs:
        w2 = og[routes].astype(np.float32)
        w2[gone] = 0.0
        w2 = w2 / float(w2.sum()) * scale
    else:
        w2 = w.astype(np.float32).copy()
        w2[gone] = 0.0
        w2 = w2 / float(w2.sum()) * total
    return gone, subs, routes, w2


def prefill_miss_plan(sc, og, routes, resident, threshold, window, scale, max_rows=32):
    """HANDOFF 18.23: which missing experts a prefill chunk leaves unread, and the rows that change.

    `sc` selection scores and `og` plain sigmoid scores (float32 [T, E]), `routes` the routed experts [T, k],
    `resident` a mask over the layer's E experts. A missing expert is skipped only when, in every row routed to it,
    its weight share is under `threshold` and a resident expert among the row's next `window` ranks (not already
    routed there) can replace it; experts routed to few rows are tried first, and each replacement is used once
    per row. Changed rows take the router's own weights over their new set, as miss_plan does. Returns None when
    nothing is skipped, else (routes [T, k], changed rows bool [T], weights float32 [T, k], skipped experts)."""
    routes = np.asarray(routes).reshape(len(sc), -1)
    k = routes.shape[1]
    missing = ~resident[routes]
    if not missing.any():
        return None
    w = og[np.arange(len(routes))[:, None], routes].astype(np.float32)
    share = w / w.sum(axis=1, keepdims=True)
    experts, counts = np.unique(routes[missing], return_counts=True)
    new = routes.copy()
    cands: dict[int, list[int]] = {}
    skipped = []
    for e, n in sorted(zip(experts.tolist(), counts.tolist()), key=lambda t: t[1]):
        if n > max_rows:
            break
        pos = np.argwhere(routes == e)
        if (share[pos[:, 0], pos[:, 1]] >= threshold).any():
            continue
        picks = []
        for t, j in pos.tolist():
            c = cands.get(t)
            if c is None:
                order = np.argsort(-sc[t], kind="stable")[:k + window]
                c = cands[t] = [int(x) for x in order if resident[int(x)] and int(x) not in routes[t]]
            if not c:
                break
            picks.append((t, j))
        if len(picks) < len(pos):
            continue
        for t, j in picks:
            new[t, j] = cands[t].pop(0)
        skipped.append(e)
    if not skipped:
        return None
    changed = (new != routes).any(axis=1)
    rows = np.flatnonzero(changed)
    w2 = np.zeros(routes.shape, np.float32)
    g = og[rows[:, None], new[rows]].astype(np.float32)
    w2[rows] = g / g.sum(axis=1, keepdims=True) * scale
    return new, changed, w2, skipped


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
        self._bias = {}
        self.miss_layers = 0
        self.hit_layers = 0
        self.spec_predicted = 0
        self.dropped = 0
        self.substituted = 0

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
        k = moe.num_experts_per_tok
        if FUSED_ROUTE and xn.dtype == mx.bfloat16 and k <= 8 and moe.gate.weight.shape[0] <= 256:
            bias = self._bias.get(i)
            if bias is None:
                bias = self._bias[i] = moe.e_score_correction_bias.astype(mx.float32)
            inds, weights, slots = route_tail(moe.gate(xn.astype(mx.float32)), bias, self._slot_table(), i, k,
                                              moe.routed_scaling_factor)
        else:
            scores, orig = moe.route_scores(xn)
            if MISS_DROP_ARMED:
                sc32, og32 = scores.reshape(-1), orig.reshape(-1)
            inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
            weights = mx.take_along_axis(orig, inds, axis=-1)
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
            weights = (weights * moe.routed_scaling_factor).astype(xn.dtype)
            slots = self._slot_table()[i][inds.reshape(-1)]
        rec = {"i": i, "r": r, "xn": xn, "inds": inds, "weights": weights, "slots": slots, "k": k, "pred": None,
               "w32": weights.astype(mx.float32) if MISS_DROP_ARMED else None}
        if MISS_DROP_ARMED and "sc32" in locals():
            rec["sc32"], rec["og32"] = sc32, og32
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
        want = slots < 0
        if MISS_DROP > 0 and spec.get("w32") is not None:
            w = np.array(spec["w32"]).reshape(-1)
            want &= w >= MISS_DROP * w.sum()
        self.spec_predicted += int(want.sum())
        return [self.index[(t, int(e))] for e in routes[want][:SPEC_PREFETCH]]

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
            mx.async_eval(spec["slots"], spec["inds"], *([spec["w32"]] if spec["w32"] is not None else []))
        return None

    def _fix(self, rec, spec=None):
        """Read a layer's missing experts and recompute only their rows; the corrected layer output."""
        i, routes, slots = rec["i"], rec["routes"], rec["slot_ids"]
        pred = self._predicted(rec)
        limit, after = None, None
        if pred is None and spec is not None and SPEC_PREFETCH > 0:
            pred = partial(self._speculate_deeper if "cache" in spec else self._speculated, spec)
            limit, after = SPEC_PREFETCH, bool(SPEC_AFTER_DEMAND)
        miss = [j for j in range(len(routes)) if slots[j] < 0]
        keep = range(len(routes))
        subs = {}
        if rec["w32"] is not None:
            w = np.array(rec["w32"]).reshape(-1)
            MISS_SHARES.extend(float(w[j]) / float(w.sum()) for j in miss)
            if MISS_DROP > 0:
                sc = og = resident = None
                if MISS_SUB > 0 and "sc32" in rec:
                    sc, og = np.array(rec["sc32"]), np.array(rec["og32"])
                    with self.store._lock:
                        resident = self.table[i] >= 0
                plan = miss_plan(w, miss, MISS_DROP, MISS_SUB, sc, og, resident, routes,
                                 self.layers[i].block_sparse_moe.routed_scaling_factor)
                if plan is not None:
                    gone, subs, routes, w2 = plan
                    self.dropped += len(gone) + len(subs)
                    self.substituted += len(subs)
                    miss = [j for j in miss if j not in gone and j not in subs]
                    keep = [j for j in range(len(routes)) if j not in gone]
                    rec = {**rec, "weights": mx.array(w2.reshape(1, 1, -1)).astype(rec["weights"].dtype)}
        got = self.store.get_many([self.index[(i, int(routes[j]))] for j in keep], prefetch=pred,
                                  prefetch_limit=limit, prefetch_after=after)
        residents = dict(zip(keep, got))
        moe = self.layers[i].block_sparse_moe
        miss = sorted(miss + list(subs))
        if not miss:
            return self._output(rec, rec["y"])
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
                mx.async_eval(rec["slots"], rec["inds"], *([rec["pred"]] if rec["pred"] is not None else []),
                              *([rec["w32"]] if rec["w32"] is not None else []),
                              *([rec["sc32"], rec["og32"]] if "sc32" in rec else []))
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
