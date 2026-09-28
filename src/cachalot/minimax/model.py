"""
MiniMax-M3 on Cachalot: the conversion's model code (`cachalot.minimax.language`) for everything but
the routed experts, which stream from SSD through the wired expert store exactly as GLM's do.

`MiniMaxModel` subclasses `GlmModel` and replaces only what differs: loading (stacked experts, a
plain KV cache per layer, the conversion's 8-bit routers), the forward call, and the chat format
(`<mm:think>` reasoning, `]<]minimax[>[`-namespaced XML tool calls). Prefill (the scan-resistant
expert path), the prefix cache, disk snapshots, `stream` and `generate` are GLM's.

Non-expert weights: 5.3 GiB resident. Routed experts: 57 layers x 128 at 24.8 MB (3-bit), 168 GiB.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from cachalot.cache.resident_store import ResidentExpertStore
from cachalot.glm.engine import _GlmSplitter
from cachalot.glm.experts import StreamingSwitchGLU, tensor_sizes
from cachalot.glm.model import GlmModel, host_available, load_non_expert_weights
from cachalot.minimax.coded_bank import index_from_bank, layout_from_sizes, reader_from_env, slot_format
from cachalot.minimax import gpu_select
from cachalot.minimax.experts import build_minimax_expert_index
from cachalot.minimax.language import FAST_NORM as _FAST_NORM
from cachalot.minimax.language import Model, ModelArgs, MiniMaxM3SparseMoeBlock
from cachalot.third_party.mlx_vlm.models.cache import KVCache

NS = "]<]minimax[>["
THINK_BEGIN = "<mm:think>"
THINK_END = "</mm:think>"
TOOL_START = NS + "<tool_call>"


class MiniMaxSplitter(_GlmSplitter):
    THINK_END = THINK_END
    TOOL_START = TOOL_START


_INVOKE_RE = re.compile(r'<invoke name="([^"]*)">(.*?)</invoke>', re.S)
_OPEN_RE = re.compile(r"<([A-Za-z_][\w.\-]*)>")


def _children(xml: str) -> list[tuple[str, str]] | None:
    """Top-level `<tag>...</tag>` elements of `xml`, or None when it is not a sequence of elements."""
    out, pos, xml = [], 0, xml.strip()
    while pos < len(xml):
        m = _OPEN_RE.match(xml, pos)
        if not m:
            return None
        tag, depth, i = m[1], 1, m.end()
        opener, closer = f"<{tag}>", f"</{tag}>"
        while depth:
            a, b = xml.find(opener, i), xml.find(closer, i)
            if b < 0:
                return None
            if 0 <= a < b:
                depth, i = depth + 1, a + len(opener)
            else:
                depth, i = depth - 1, b + len(closer)
        out.append((tag, xml[m.end():i - len(closer)]))
        pos = i
        while pos < len(xml) and xml[pos].isspace():
            pos += 1
    return out


def _value(raw: str, schema: dict | None):
    """A tool argument from the template's XML: nested elements become dicts or lists (<item>),
    leaves are typed by the tool's JSON schema (a string stays a string)."""
    schema = schema or {}
    kind = schema.get("type")
    kids = _children(raw) if "<" in raw else None
    if kids:
        if all(tag == "item" for tag, _ in kids):
            return [_value(v, schema.get("items")) for _, v in kids]
        props = schema.get("properties") or {}
        return {tag: _value(v, props.get(tag)) for tag, v in kids}
    if kind == "string":
        return raw
    if kind in ("integer", "number", "boolean", "null", "array", "object") or not schema:
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def parse_minimax_tool_calls(text: str, tools=None) -> list[dict[str, Any]]:
    """`]<]minimax[>[<tool_call> ... <invoke name="f"><k>v</k></invoke> ... </tool_call>` to OpenAI calls."""
    schemas = {}
    for tool in tools or []:
        fn = tool.get("function", tool)
        schemas[fn.get("name")] = fn.get("parameters") or {}
    plain = text.replace(NS, "")
    start = plain.find("<tool_call>")
    if start < 0:
        return []
    calls = []
    for name, body in _INVOKE_RE.findall(plain[start:]):
        params = schemas.get(name, {}).get("properties") or {}
        args = {tag: _value(v, params.get(tag)) for tag, v in (_children(body) or [])}
        calls.append({"type": "function", "function": {"name": name, "arguments": args}})
    return calls


def codes_qmv_ok() -> bool:
    """The codes kernel is on and matched mx.quantized_matmul at load (the slab kernels are the same code)."""
    from cachalot.minimax import codes_qmv

    return bool(codes_qmv.KERNEL) and codes_qmv._OK


def _checkpoint_tensor_names(model_path) -> list[str]:
    from cachalot.storage.index import read_safetensors_header

    return [n for shard in sorted(Path(model_path).glob("model-*.safetensors")) for n in read_safetensors_header(shard)[0]]


def _wanted(name: str) -> bool:
    return ".switch_mlp." not in name and ".mtp." not in name and not name.startswith("mtp.")


# HANDOFF 18.1: the decode-step overlap and one-layer-early routing prediction
DECODE_OVERLAP = os.environ.get("CACHALOT_MINIMAX_DECODE_OVERLAP", "1") != "0"
# HANDOFF 18.9: each decode layer ranks the next layer's experts by that layer's own norm and router applied to this
# layer's residual (top PREDICT_TOPK), and once its own misses have arrived reads the best-ranked one that is not
# resident (the store's DECODE_PREFETCH_AFTER_DEMAND / DECODE_PREFETCH_LIMIT, which MiniMaxModel sets to 1 / 1)
# while the GPU finishes the layer: the drive is otherwise idle ~1.5 ms per layer. Same tokens. 0 turns it off.
# (18.1 closed this with the prediction beside the misses and a second GPU round trip per layer: both fixed.)
PREDICT_TOPK = int(os.environ.get("CACHALOT_MINIMAX_PREDICT_TOPK", "2"))
DECODE_BORROW = os.environ.get("CACHALOT_MINIMAX_DECODE_BORROW", "1") != "0"
# HANDOFF 18.10 (M13b): a slot keeps each group's bias as a 4-bit code (k + 7, bias = bf16(k x scale)) instead of the
# bf16 bias: 22.35 MiB a slot instead of 23.62, ~5.7 % more resident experts in the same budget, same tokens (the
# decode matmul is MLX's qmv with the bias rebuilt in the kernel, codes_qmv). Needs the bank. 0 keeps bf16 biases.
SLOT_CODES = os.environ.get("CACHALOT_MINIMAX_SLOT_CODES", "1") != "0"
# HANDOFF 18.13 (M23, M23b): 2 (the default) keeps one byte per group in a codes slot, indexing the projection's
# 256-entry table of (scale, bias) pairs (no projection of this checkpoint has more than 205; the reader refuses
# one with more), and no codes: 21.10 MiB a slot instead of 22.36, 142 more resident experts in 52 GiB, same tokens.
# 1: a byte scale index plus the 4-bit codes (21.52 MiB). 0: bf16 scales plus the codes (0.31.0).
SLOT_SIDX = int(os.environ.get("CACHALOT_MINIMAX_SLOT_SIDX", "2"))
# HANDOFF 18.7: a prefill chunk reads the whole next layer ahead only from this many tokens. GLM's 128 suits top-8
# of 288; MiniMax's routing is skewed (150 tokens reach ~74 of 128 experts per layer), so below ~750 tokens the
# speculative reads cost more than the overlap saves (150 tokens: 14.7 -> 9.0-10.1 s; 1,000: 15.3 vs 15.9 s).
SPECULATE_MIN_TOKENS = int(os.environ.get("CACHALOT_MINIMAX_SPECULATE_MIN_TOKENS", "768"))
DECODE_BORROW_KEEP = 16  # transient slots never borrowed (the store's PREDICT_SLOT_RESERVE)
# HANDOFF 18.8: a prefill chunk of at least PREFILL_PREDICT_MIN tokens reads ahead only the next layer's experts that
# its router, applied to this layer's residual, picks for any of the chunk's tokens (top PREFILL_PREDICT_TOPK each),
# most-picked first, instead of the whole layer from SPECULATE_MIN_TOKENS (or nothing below it). Top-3 of the
# prediction is ~92 % precise and ~80-90 % complete: 150-1,500-token prefills -6 to -15 %, same tokens; below
# ~100 tokens it neither helps nor hurts. Which experts are read early changes, never the arithmetic. 0 turns it off.
# Plain ints so TF_ALTERNATE / a benchmark can flip them in one process.
PREFILL_PREDICT = int(os.environ.get("CACHALOT_MINIMAX_PREFILL_PREDICT", "1"))
PREFILL_PREDICT_MIN = int(os.environ.get("CACHALOT_MINIMAX_PREFILL_PREDICT_MIN", "100"))
PREFILL_PREDICT_TOPK = int(os.environ.get("CACHALOT_MINIMAX_PREFILL_PREDICT_TOPK", "3"))
# above this many tokens a chunk reads the whole next layer again (4,096: 17.7 s whole against 19.4 predicted; 3,000:
# 16.1 against 15.0); reading the predicted experts first and then the rest measured the same as the whole layer
PREFILL_PREDICT_MAX = int(os.environ.get("CACHALOT_MINIMAX_PREFILL_PREDICT_MAX", "3072"))
# per layer pair: experts predicted, experts the next layer routed to, and the overlap (benchmarks read it)
PREFILL_PREDICT_STATS = {"predicted": 0, "actual": 0, "overlap": 0}
# HANDOFF 18.15 (S2): 56 GiB of expert slots decode 13 % faster than 52 on 0.33 (agent benchmark, same tokens),
# but an 8,192-token prefill chunk at 56 hit warning pressure and then Metal's out-of-memory: a chunk that long
# prefills at PREFILL_BUDGET_GIB (the capacity is parked down before it and taken back at the first decode token),
# and the startup budget leaves STARTUP_RESERVE_GIB of what macOS reports available for everything else.
PREFILL_BUDGET_GIB = float(os.environ.get("CACHALOT_MINIMAX_PREFILL_BUDGET_GIB", "52"))
STARTUP_RESERVE_GIB = float(os.environ.get("CACHALOT_MINIMAX_STARTUP_RESERVE_GIB", "16"))
# HANDOFF 18.18: the startup budget also leaves this much of Metal's recommended working set for the non-expert
# weights, the 272 transient slots, the KV cache and other processes' GPU memory. 62 GiB was the most the default
# working set (77.76 GiB on 96 GiB) held without paging (18.17); with `sysctl iogpu.wired_limit_mb=88064` (86 GiB)
# the budget may be 68 (70 swapped), and a restart that loses the sysctl falls back to 62 instead of over-allocating.
GPU_RESERVE_GIB = float(os.environ.get("CACHALOT_MINIMAX_GPU_RESERVE_GIB", "15.75"))
# HANDOFF 18.16: 62 GiB of slots decode 15 % faster than 56 (agent benchmark, same tokens) and 64 ran out of Metal
# memory in a 2,048-token chunk at full capacity, so from 0.35.0 chunks above PREFILL_FULL_TOKENS already give back
# slots (linearly down to PREFILL_BUDGET_GIB at 8,192): a 2,048-token chunk prefills at ~60.7 GiB of 62.
PREFILL_FULL_TOKENS = int(os.environ.get("CACHALOT_MINIMAX_PREFILL_FULL_TOKENS", "512"))



def _mapped_tensors(model_path: Path, prefix: str) -> dict[str, np.ndarray]:
    """Read-only numpy maps of the checkpoint tensors named `prefix` + name (HANDOFF 18.18), by their byte range."""
    from cachalot.storage.index import read_safetensors_header

    out = {}
    for shard in sorted(Path(model_path).glob("model-*.safetensors")):
        header, data_start = read_safetensors_header(shard)
        for name, meta in header.items():
            if name.startswith(prefix) and meta["dtype"] in ("U32", "BF16"):
                a, b = meta["data_offsets"]
                dtype = np.uint32 if meta["dtype"] == "U32" else np.uint16
                out[name[len(prefix):]] = np.memmap(shard, dtype=dtype, mode="r", offset=data_start + a,
                                                    shape=tuple(meta["shape"]))
    return out

class MiniMaxModel(GlmModel):
    # the fused RMSNorm rounds differently (HANDOFF 18.2): snapshots written without it do not match
    # HANDOFF 18.21: with miss substitution on, a reply's decode KV differs from the exact path's; key it apart
    NUMERICS_TAG = ("-fastnorm" if _FAST_NORM else "") + (
        f"-missdrop{gpu_select.MISS_DROP:g}-sub{gpu_select.MISS_SUB}" if gpu_select.MISS_DROP > 0 else "")
    # A 2,048-token chunk already reads nearly every routed expert (168 GiB), so a longer chunk reads the same
    # bytes for more tokens: 8,192 prefills at ~230 tok/s against ~74-85 at 2,048, same NLL (HANDOFF 18.1).
    PREFILL_CHUNK = int(os.environ.get("CACHALOT_MINIMAX_PREFILL_CHUNK", "8192"))
    PREFILL_FULL_TOKENS = PREFILL_FULL_TOKENS

    def __init__(
        self,
        model_path,
        *,
        expert_budget_gib: float = 68.0,
        wired_limit_gib: float | None = None,
        load_workers: int = 8,
        heartbeat_seconds: float = 0.5,
        verbose: bool = True,
    ) -> None:
        t0 = time.perf_counter()
        self.model_path = Path(model_path)
        config = json.loads((self.model_path / "config.json").read_text())
        self.config = ModelArgs.from_dict(config)
        quant = config.get("quantization") or config.get("quantization_config") or {}

        bank = os.environ.get("CACHALOT_MINIMAX_BANK")
        if bank and not any(".switch_mlp." in n for n in _checkpoint_tensor_names(self.model_path)):
            # a checkpoint trimmed to its non-expert weights: the bank is the only copy of the experts (18.4)
            self.expert_format, self.expert_index = index_from_bank(bank)
        else:
            self.expert_format, self.expert_index = build_minimax_expert_index(self.model_path)
        sizes = tensor_sizes(self.expert_format)
        # the bias-free bank when CACHALOT_MINIMAX_BANK names one (HANDOFF 18.4), else the checkpoint
        reader = reader_from_env()
        # HANDOFF 18.10: slots hold 4-bit bias codes instead of bf16 biases when the bank serves every expert
        self.slot_codes = bool(
            SLOT_CODES and hasattr(reader, "covers") and all(reader.covers(*key) for key in self.expert_index)
        )
        self.slot_format = slot_format(self.expert_format, SLOT_SIDX) if self.slot_codes else self.expert_format
        if self.slot_codes:
            from cachalot.minimax import codes_qmv

            if not codes_qmv.self_check():
                print("[minimax] codes kernel differs from mx.quantized_matmul on this MLX: biases are rebuilt "
                      "for every matmul instead (bit-identical, slower)", flush=True)
        slot_sizes = tensor_sizes(self.slot_format)
        expert_bytes = sum(slot_sizes.values())

        # the same memory rules as GLM (HANDOFF 17.1): a capped MLX buffer cache, a wired set
        mx.set_cache_limit(int(float(os.environ.get("CACHALOT_GLM_MLX_CACHE_GIB", "2")) * 1024**3))
        if wired_limit_gib is None:
            # capped at Metal's recommended working set below: by default the whole of it (HANDOFF 18.18)
            wired_limit_gib = float(os.environ.get("CACHALOT_MLX_WIRED_LIMIT_GIB", "96"))
        from cachalot.config import device_memory

        _, recommended = device_memory()
        if wired_limit_gib > 0:
            mx.set_wired_limit(int(min(wired_limit_gib * 1024**3, recommended)))

        n_experts = self.config.num_local_experts
        transient = 2 * n_experts + 16  # one prefill layer's misses plus the next layer read early (PREFILL_SCAN)
        available = host_available()
        if available > 0 and expert_budget_gib * 1024**3 > available - STARTUP_RESERVE_GIB * 1024**3:
            capped = max(PREFILL_BUDGET_GIB * 0.75, (available / 1024**3) - STARTUP_RESERVE_GIB)
            if capped < expert_budget_gib:
                print(f"[minimax] {available / 1024**3:.1f} GiB available: expert budget {expert_budget_gib:.1f} "
                      f"-> {capped:.1f} GiB", flush=True)
                expert_budget_gib = capped
        gpu_cap = recommended / 1024**3 - GPU_RESERVE_GIB
        if recommended > 0 and expert_budget_gib > gpu_cap:
            print(f"[minimax] Metal working set {recommended / 1024**3:.1f} GiB: expert budget {expert_budget_gib:.1f} "
                  f"-> {gpu_cap:.1f} GiB", flush=True)
            expert_budget_gib = gpu_cap
        budget = int(expert_budget_gib * 1024**3)
        self._prefill_budget = int(min(PREFILL_BUDGET_GIB, expert_budget_gib) * 1024**3)
        # HANDOFF 18.12: with the codes kernel, the slots live in slabs so decode can pick its experts on the GPU
        from cachalot.minimax import gpu_select

        pool = None
        self.gpu_select = bool(gpu_select.GPU_SELECT and self.slot_codes and codes_qmv_ok())
        if self.gpu_select:
            from cachalot.cache.slots import SlabSlotPool

            pool = SlabSlotPool(slot_sizes, budget // expert_bytes + transient, verbose=verbose)
        self.store = ResidentExpertStore(
            budget,
            reader,
            tensor_sizes=slot_sizes,
            slot_pool=pool,
            transient_slots=transient,
            load_workers=load_workers,
            verbose=verbose,
        )
        self.store.format = self.slot_format
        bank_layout = getattr(self.store.reader, "layout", None)
        if bank_layout is not None and bank_layout != layout_from_sizes(sizes):
            raise ValueError(f"expert bank {self.store.reader.bank_dir} was written for another checkpoint layout")
        # decode holds the prefill transient slots as residents until the next prefill (HANDOFF 18.1)
        self.store.decode_borrow = max(0, self.store.transient_slots - DECODE_BORROW_KEEP) if DECODE_BORROW else 0

        self.model = Model(self.config)
        n_moe = 0
        for i, layer in enumerate(self.model.layers):
            if layer.is_sparse:
                moe: MiniMaxM3SparseMoeBlock = layer.block_sparse_moe
                moe.switch_mlp = StreamingSwitchGLU(i, self.store, self.expert_index, self.slot_format, moe.activation)
                moe.switch_mlp.speculate_min_tokens = SPECULATE_MIN_TOKENS
                moe.switch_mlp.codes = self.slot_codes
                n_moe += 1

        self._install_decode_hooks()
        self.gpu_decoder = gpu_select.GpuSelectDecoder(self) if self.gpu_select else None
        # the store's decode prefetch timing for the prediction above (module ints, so TF_ALTERNATE can flip them)
        from cachalot.cache import resident_store as _rs

        if "CACHALOT_DECODE_PREFETCH_AFTER_DEMAND" not in os.environ:
            _rs.DECODE_PREFETCH_AFTER_DEMAND = 1
        if "CACHALOT_DECODE_PREFETCH_LIMIT" not in os.environ:
            _rs.DECODE_PREFETCH_LIMIT = 1

        weights = load_non_expert_weights(self.model_path, _wanted)

        def quantized(path, module):
            if not hasattr(module, "to_quantized") or f"{path}.scales" not in weights:
                return False
            spec = quant.get(path)
            return spec if isinstance(spec, dict) else True  # the routers are 8-bit in this conversion

        nn.quantize(
            self.model,
            group_size=int(quant.get("group_size", 64)),
            bits=int(quant.get("bits", 3)),
            mode=quant.get("mode", "affine"),
            class_predicate=quantized,
        )
        params = dict(nn.utils.tree_flatten(self.model.parameters()))
        missing = sorted(set(params) - set(weights))
        if missing:
            raise ValueError(f"{len(missing)} model parameters have no weight, e.g. {missing[:5]}")
        self.model.load_weights([(k, v) for k, v in weights.items() if k in params])
        mx.eval(self.model.parameters())
        self.model.eval()
        self.unused_weights = sorted(set(weights) - set(params))
        self.trunk_bytes = sum(v.nbytes for v in params.values())
        weights.clear()  # the stacked weights below replace q/k/v and gate/up: drop every reference
        params.clear()
        # HANDOFF 18.2: q/k/v and the shared expert's gate/up each as one matmul, the originals dropped
        from cachalot.minimax import language as _lang

        for layer in self.model.layers:
            if _lang.FUSE_QKV:
                layer.self_attn.fuse()
            if _lang.FUSE_SHARED:
                (layer.block_sparse_moe.shared_experts if layer.is_sparse else layer.mlp).fuse()
        inner = self.model.model
        if _lang.HOST_EMBED and isinstance(inner.embed_tokens, nn.QuantizedEmbedding) \
                and not self.config.tie_word_embeddings:
            inner.embed_tokens = _lang.HostEmbedding(inner.embed_tokens, _mapped_tensors(self.model_path, "model.embed_tokens."))
        mx.clear_cache()

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        eos = config.get("eos_token_id")
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()

        # HANDOFF 18.6: the expert capacity shrinks as the KV cache grows, so MLX's footprint stays where
        # the 52 GiB budget was measured (a short context); a negative allowance turns it off
        allowance = float(os.environ.get("CACHALOT_MINIMAX_KV_ALLOWANCE_GIB", "1.0"))
        self._memory_target = int(mx.get_active_memory() + allowance * 1024**3) if allowance >= 0 else None

        self.prefix = []
        self.disk = None
        self.max_seq_len = int(os.environ.get("CACHALOT_MINIMAX_MAX_SEQ_LEN", "131072"))
        self._lock = threading.RLock()
        self._busy = False
        self._idle_since = time.perf_counter()
        self._stop = threading.Event()
        if heartbeat_seconds > 0:
            threading.Thread(target=self._heartbeat, args=(heartbeat_seconds,), daemon=True, name="minimax-heartbeat").start()

        if verbose:
            print(
                f"MiniMax-M3: {len(self.model.layers)} layers ({n_moe} MoE), trunk {self.trunk_bytes / 1024**3:.1f} GiB, "
                f"{len(self.expert_index)} experts x {expert_bytes / 2**20:.2f} MiB, "
                f"{self.store.capacity} resident slots ({self.store.budget_bytes / 1024**3:.1f} GiB), "
                f"loaded in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    # -- decode ------------------------------------------------------------------------------------------
    def _install_decode_hooks(self) -> None:
        """Per MoE layer, what a decode token does between its routing and its routed experts (HANDOFF 18.1).

        DECODE_OVERLAP: the routing is evaluated on its own, then the shared expert is queued (async) so the
        GPU runs it while the misses are read (and, since 0.24.0, the routed hits too: `hit_overlap`), and the routed output is not evaluated at the end of the layer
        (the next layer's routing sync covers it): one GPU round trip per layer instead of two. Bit-identical.
        PREDICT_TOPK > 0: the next MoE layer's routing is predicted from this layer's residual (its own norm,
        gate and bias, in the same sync, ranked on the host) and its best-ranked non-resident expert is read into
        a transient slot once this layer's misses have arrived (the store's prefetch path, HANDOFF 18.9).
        """
        layers = self.model.layers
        store, index = self.store, self.expert_index

        def make(i, moe, nxt):
            def hook(x, residual, inds, weights):
                scores = None
                if nxt is not None and PREDICT_TOPK > 0 and residual is not None:
                    # evaluated in the same sync as this layer's routing and ranked on the host: any MLX op on
                    # the result after the sync (a cast, a sort) is a second GPU round trip, ~21 ms a token
                    scores, _ = nxt.block_sparse_moe.route_scores(nxt.post_attention_layernorm(residual))
                mx.eval(inds, weights, *([scores] if scores is not None else []))
                shared = moe.shared_experts(x)
                mx.async_eval(shared)
                prefetch = None
                if scores is not None:
                    sc = np.array(scores).reshape(-1)
                    top = np.argpartition(-sc, PREDICT_TOPK - 1)[:PREDICT_TOPK]
                    prefetch = [index[(i + 1, int(e))] for e in top[np.argsort(-sc[top])]]
                return shared, prefetch
            return hook

        predicted: dict[int, set[int]] = {}

        def make_prefill(i, nxt):
            def hook(residual, inds):
                n = inds.shape[0] * inds.shape[1]
                if not PREFILL_PREDICT or n < PREFILL_PREDICT_MIN or n > PREFILL_PREDICT_MAX:
                    mine = predicted.pop(i, None)
                    if mine is not None:
                        _count_prediction(mine, inds)
                    return None
                pred = None
                if nxt is not None:
                    scores, _ = nxt.block_sparse_moe.route_scores(nxt.post_attention_layernorm(residual))
                    pred = mx.argpartition(-scores, kth=PREFILL_PREDICT_TOPK - 1, axis=-1)[..., :PREFILL_PREDICT_TOPK]
                # one sync for this layer's routing and the prediction (the switch's own sync is then free)
                mx.eval(inds, *([pred] if pred is not None else []))
                mine = predicted.pop(i, None)
                if mine is not None:
                    _count_prediction(mine, inds)
                if pred is None:
                    return []
                counts = np.bincount(np.array(pred).reshape(-1))
                order = [int(e) for e in np.argsort(-counts, kind="stable") if counts[e] > 0]
                predicted[i + 1] = set(order)
                return [index[(i + 1, e)] for e in order]
            return hook

        def _count_prediction(mine, inds):
            actual = set(np.unique(np.array(inds)).tolist())
            PREFILL_PREDICT_STATS["predicted"] += len(mine)
            PREFILL_PREDICT_STATS["actual"] += len(actual)
            PREFILL_PREDICT_STATS["overlap"] += len(mine & actual)

        for i, layer in enumerate(layers):
            if layer.is_sparse:
                nxt = layers[i + 1] if i + 1 < len(layers) and layers[i + 1].is_sparse else None
                layer.block_sparse_moe.prefill_hook = make_prefill(i, nxt)
            if not layer.is_sparse or not DECODE_OVERLAP:
                continue
            nxt = layers[i + 1] if i + 1 < len(layers) and layers[i + 1].is_sparse else None
            layer.block_sparse_moe.decode_hook = make(i, layer.block_sparse_moe, nxt)
            layer.block_sparse_moe.switch_mlp.decode_eval = False
            # HANDOFF 18.5: the hit experts' matmuls run while the layer's misses are read (bit-identical)
            layer.block_sparse_moe.switch_mlp.hit_overlap = True

    # HANDOFF 18.8: between requests the resident set moves towards the most-requested experts (0.8-2.2 s of
    # reads, cancelled by the next request): with a 4 s pause between agent turns, decode -12 %, short prefills
    # -3 to -6 %, same tokens. CACHALOT_MINIMAX_IDLE_WARM=0 turns it off.
    IDLE_WARM = os.environ.get("CACHALOT_MINIMAX_IDLE_WARM", "1") != "0"

    # HANDOFF 18.6: every layer is a plain KVCache, so a conversation's snapshot is handed on, not copied
    CONSUME_SNAPSHOTS = os.environ.get("CACHALOT_MINIMAX_CONSUME_SNAPSHOTS", "1") != "0"

    # -- model -------------------------------------------------------------------------------------------
    def new_cache(self):
        return [KVCache() for _ in self.model.layers]

    def _forward(self, tokens: list[int], cache) -> mx.array:
        from cachalot.minimax import gpu_select

        if len(tokens) == 1 and self.gpu_decoder is not None and gpu_select.GPU_SELECT:
            return self.gpu_decoder.forward(tokens[0], cache)
        return self._forward_layers(tokens, cache)

    def _forward_layers(self, tokens: list[int], cache) -> mx.array:
        # the lm_head on the last position only: a prefill chunk of 8,192 would otherwise build 8,192 x 200k
        # logits (3.3 GB) to keep one row (HANDOFF 18.1). One token (decode) is the same call as before.
        h = self.model.model(mx.array(tokens, dtype=mx.int32)[None], cache=cache)[:, -1:, :]
        if self.config.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(h)[:, -1, :]
        return self.model.lm_head(h)[:, -1, :]

    # -- chat format -------------------------------------------------------------------------------------
    @property
    def splitter_cls(self):
        return MiniMaxSplitter

    def render_chat(self, messages, *, tools=None, thinking=False, effort=None, add_generation_prompt=True) -> str:
        """M3's template takes thinking_mode enabled/disabled and writes the `<mm:think>` / `</mm:think>`
        prefix itself; it has no effort levels."""
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False,
            thinking_mode="enabled" if thinking else "disabled",
        )

    def parse_tool_calls(self, text: str, tools=None):
        return parse_minimax_tool_calls(text, tools)

    def encode_chat(self, messages, *, tools=None, reasoning_effort=None, add_generation_prompt=True) -> list[int]:
        text = self.render_chat(messages, tools=tools, thinking=reasoning_effort is not None,
                                add_generation_prompt=add_generation_prompt)
        return list(self.tokenizer.encode(text, add_special_tokens=False))

