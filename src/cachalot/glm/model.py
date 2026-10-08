"""
GLM-5.3-Flash on Cachalot: mlx-vlm's model code (vendored, unmodified) for everything but the
routed experts, which stream from SSD through Cachalot's wired expert store.

Only the non-expert weights are loaded into memory (~10 GB for the 4-bit build: attention, the
34 KDA and 11 MLA layers, the dense and shared MLPs, hyper-connections, embeddings, head). The
12,096 routed experts (42 MoE layers x 288, 13.5 MiB each at 4-bit) are read on demand into a
fixed pool of wired slots sized by the expert budget, least recently used evicted first.

Text only for now: the vision tower and the MTP layer are not loaded.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from cachalot.cache.resident_store import ResidentExpertStore
from cachalot.glm.experts import StreamingSwitchGLU, build_glm_expert_index, tensor_sizes
from cachalot.storage.index import read_safetensors_header
from cachalot.storage.reader import ExpertReader
from cachalot.third_party.mlx_vlm.models.glm5_next.config import TextConfig
from cachalot.third_party.mlx_vlm.models.cache import CacheList, KVCache
from cachalot.third_party.mlx_vlm.models.glm5_next.language import Glm5NextMoE, LanguageModel

# the resident expert set survives a restart (HANDOFF 18.2); CACHALOT_WARM_SET=0 turns it off
WARM_SET = os.environ.get("CACHALOT_WARM_SET", "1") != "0"
# HANDOFF 18.16 (M27): a request no longer waits for the whole warm set; it stops the reads (one batch at most)
# and the rest is read back after the request, while idle. 0 restores the wait. An int, so TF_ALTERNATE-style
# scripts can flip it.
WARM_SET_YIELD = int(os.environ.get("CACHALOT_WARM_SET_YIELD", "1"))

_NP_DTYPE = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "U32": np.uint32, "I32": np.int32,
             "U8": np.uint8, "I64": np.int64}


# A model directory may carry the non-expert weights of another quantisation next to the expert shards: one safetensors file
# in mlx-lm's layout (`language_model.*` names, fused conv, `forget_gate.*`), with per-module bit widths in config.json's `quantization`.
NONEXPERT_SIDECAR = "nonexpert-sanitized.safetensors"


def _sidecar_key(name: str) -> str:
    """mlx-lm's names for the linear-attention parts back to the ones `LanguageModel.sanitize` fuses from."""
    return name.replace(".self_attn.forget_gate.", ".self_attn.").replace(".self_attn.conv1d.weight", ".self_attn.qkv_conv.conv.weight")


def _is_routed_expert(name: str) -> bool:
    return ".mlp.experts." in name


def _wanted(name: str) -> bool:
    """Non-expert text weights; the vision tower and the MTP block stay on disk."""
    if name.startswith("model.visual.") or ".mtp." in name or name.startswith("mtp."):
        return False
    return not _is_routed_expert(name)


def load_non_expert_weights(model_path: Path, wanted=_wanted) -> dict[str, mx.array]:
    """Read every wanted tensor by its byte range (no pass over the expert bytes)."""
    out: dict[str, mx.array] = {}
    for shard in sorted(model_path.glob("model-*.safetensors")):
        header, data_start = read_safetensors_header(shard)
        names = [n for n in header if n != "__metadata__" and wanted(n)]
        if not names:
            continue
        with open(shard, "rb", buffering=0) as f:
            for name in names:
                meta = header[name]
                a, b = meta["data_offsets"]
                f.seek(data_start + a)
                buf = f.read(b - a)
                arr = np.frombuffer(buf, dtype=_NP_DTYPE[meta["dtype"]]).reshape(meta["shape"])
                value = mx.array(arr)
                if meta["dtype"] == "BF16":
                    value = value.view(mx.bfloat16)
                out[name] = value
    return out


def _remap(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """mlx-vlm Model.sanitize's key renames for the text side."""
    out = {}
    for key, value in weights.items():
        if key.startswith("model.language_model."):
            key = "language_model.model." + key[len("model.language_model."):]
        elif key.startswith("lm_head."):
            key = "language_model." + key
        out[key] = value
    return out


MEMORY_FIT_EVERY = int(os.environ.get("CACHALOT_MEMORY_FIT_EVERY", "512"))
MEMORY_FIT_MIN_SLOTS = 8
# HANDOFF 18.15 (S2): the expert capacity may only grow while macOS reports normal memory pressure and at least this
# much stays available (kern.memorystatus_level: free plus reclaimable) after the growth; at warning pressure, or with
# less than half of it available, a slab is given back. Free pages alone sit near 0.5 GiB at normal pressure.
HOST_AVAILABLE_FLOOR = int(float(os.environ.get("CACHALOT_HOST_AVAILABLE_FLOOR_GIB", "8")) * 1024**3)

_SYSCTL = None


def _sysctl(name: str) -> int | None:
    global _SYSCTL
    if _SYSCTL is None:
        import ctypes
        import ctypes.util

        try:
            _SYSCTL = ctypes, ctypes.CDLL(ctypes.util.find_library("c")).sysctlbyname
        except (OSError, AttributeError):
            _SYSCTL = False
    if not _SYSCTL:
        return None
    ctypes, fn = _SYSCTL
    value, size = ctypes.c_uint64(0), ctypes.c_size_t(8)
    if fn(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        return None
    return value.value if size.value == 8 else value.value & 0xFFFFFFFF


def host_memory() -> tuple[int, int]:
    """(free bytes, kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical); (-1, 0) where the
    kernel does not say (not macOS)."""
    free, page, level = _sysctl("vm.page_free_count"), _sysctl("hw.pagesize"), _sysctl("kern.memorystatus_vm_pressure_level")
    if free is None or page is None:
        return -1, 0
    return free * page, level or 0


def host_available() -> int:
    """Bytes macOS considers available to allocate (kern.memorystatus_level percent of RAM); -1 when unknown."""
    level, total = _sysctl("kern.memorystatus_level"), _sysctl("hw.memsize")
    if level is None or total is None:
        return -1
    return total * level // 100


# HANDOFF 18.24: with other apps holding host memory, pressure flickers between normal and critical within seconds
# and kern.memorystatus_level swings 6-44 % as the machine swaps. The fits only look at the moment they run (a
# prefill's start, decode tokens 1, 513, ...), so a fit right after a prefill freed its buffers read a high moment
# and grew the capacity by 13.6 GiB at once; swap then grew 6 -> 16 GiB and 150-400-token follow-ups took 33-41 s
# instead of 6-9. A watcher samples both every HOST_WATCH_EVERY seconds: the capacity grows only after
# HOST_GROW_QUIET seconds without warning pressure (or less than half the floor available), by the lowest
# availability seen in that window; each such event gives back a slab at the next fit (a decode token), at most one
# every HOST_SHRINK_EVERY seconds. HOST_GROW_QUIET <= 0 turns the watcher off (the 0.43 rule).
HOST_GROW_QUIET = float(os.environ.get("CACHALOT_HOST_GROW_QUIET_S", "60"))
HOST_WATCH_EVERY = 0.5
HOST_SHRINK_EVERY = float(os.environ.get("CACHALOT_HOST_SHRINK_EVERY_S", "10"))


class HostWatch:
    """Memory pressure and availability sampled in a daemon thread (two sysctls, a few µs each)."""

    def __init__(self, floor: int, quiet: float = HOST_GROW_QUIET, every: float = HOST_WATCH_EVERY,
                 clock=time.monotonic, start: bool = True):
        self.floor, self.quiet_s, self.every, self.clock = floor, quiet, every, clock
        self.last_event = float("-inf")
        self.events = 0
        self._samples: list[tuple[float, int]] = []
        self._lock = threading.Lock()
        if start:
            threading.Thread(target=self._run, daemon=True, name="host-watch").start()

    def sample(self, level: int, available: int) -> None:
        t = self.clock()
        with self._lock:
            if level >= 2 or 0 <= available < self.floor // 2:
                self.last_event = t
                self.events += 1
            self._samples.append((t, available))
            cut = t - self.quiet_s
            while self._samples and self._samples[0][0] < cut:
                self._samples.pop(0)

    def _run(self) -> None:
        while True:
            self.sample(host_memory()[1], host_available())
            time.sleep(self.every)

    def quiet(self) -> bool:
        """No pressure event in the last `quiet_s` seconds."""
        with self._lock:
            return self.clock() - self.last_event >= self.quiet_s

    def min_available(self) -> int:
        with self._lock:
            return min((a for _, a in self._samples if a >= 0), default=-1)


# HANDOFF 18.17: the GPU's memory is shared by every process. With 62 GiB of MiniMax slots, 1 GiB more of anyone's
# GPU memory (48 more slots, or another process holding 2 GiB) slowed decode 6-25 % and short prefills up to 2.3x,
# the same tokens: the driver pages (its "In use system memory" stays pinned near 74.3 GiB while "Alloc system
# memory", every process's GPU allocations, grows). 62 GiB alone allocates 78.1-78.7 GiB, 63 GiB 79.0-79.2. The
# capacity may only grow, and gives back slots, so that the allocated total stays under Metal's recommended working
# set plus this slack. Negative turns it off.
GPU_ALLOC_SLACK = int(float(os.environ.get("CACHALOT_GPU_ALLOC_SLACK_GIB", "1.0")) * 1024**3)

_GPU_STATS = None


def gpu_allocated() -> int:
    """Bytes of GPU system memory allocated by all processes (the AGX driver's PerformanceStatistics "Alloc system
    memory"); -1 when the driver does not say. ~20 µs a read."""
    global _GPU_STATS
    if _GPU_STATS is None:
        _GPU_STATS = False
        try:
            import ctypes
            import ctypes.util

            iokit = ctypes.cdll.LoadLibrary(ctypes.util.find_library("IOKit"))
            cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
            iokit.IOServiceMatching.restype = ctypes.c_void_p
            iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
            iokit.IOServiceGetMatchingService.restype = ctypes.c_uint
            iokit.IOServiceGetMatchingService.argtypes = [ctypes.c_uint, ctypes.c_void_p]
            iokit.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
            iokit.IORegistryEntryCreateCFProperty.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                                                              ctypes.c_uint]
            cf.CFStringCreateWithCString.restype = ctypes.c_void_p
            cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
            cf.CFDictionaryGetValue.restype = ctypes.c_void_p
            cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            cf.CFNumberGetValue.restype = ctypes.c_bool
            cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            cf.CFRelease.argtypes = [ctypes.c_void_p]
            service = iokit.IOServiceGetMatchingService(0, iokit.IOServiceMatching(b"AGXAccelerator"))
            if service:
                utf8 = 0x08000100
                keys = (cf.CFStringCreateWithCString(None, b"PerformanceStatistics", utf8),
                        cf.CFStringCreateWithCString(None, b"Alloc system memory", utf8))
                _GPU_STATS = ctypes, iokit, cf, service, keys
        except (OSError, AttributeError):
            pass
    if not _GPU_STATS:
        return -1
    ctypes, iokit, cf, service, (k_stats, k_alloc) = _GPU_STATS
    stats = iokit.IORegistryEntryCreateCFProperty(service, k_stats, None, 0)
    if not stats:
        return -1
    try:
        value = cf.CFDictionaryGetValue(stats, k_alloc)
        out = ctypes.c_int64(-1)
        if not value or not cf.CFNumberGetValue(value, 4, ctypes.byref(out)):  # kCFNumberSInt64Type
            return -1
        return out.value
    finally:
        cf.CFRelease(stats)


def gpu_ceiling() -> int:
    """The GPU memory all processes together may allocate before the governor gives back slots; -1 when off."""
    if GPU_ALLOC_SLACK < 0:
        return -1
    try:
        recommended = int(mx.device_info()["max_recommended_working_set_size"])
    except (AttributeError, KeyError, TypeError):
        return -1
    return recommended + GPU_ALLOC_SLACK


class _NoProjectedCache(KVCache):
    """Stands in for the MLA layers' projected prefill cache, which mlx-vlm keeps per head:
    ~720 KB per token over the 11 MLA layers (64 heads x 256 x K and V), ~14 GB at Hermes's
    20k-token prompts, on top of the expert cache. Reporting size -1 makes every prefill chunk
    project from the compact latent cache instead (language.py, Glm5NextAttention._attend)."""

    def size(self):
        return -1


def _clone(obj):
    """A snapshot of a cache tree that later in-place updates cannot reach.

    MLX slice assignment updates an array object in place, so a snapshot that shares array
    objects with the live cache would change under it; every array is re-wrapped instead
    (mx.array(x) is a new object over the same data, no copy until one side is updated)."""
    if isinstance(obj, mx.array):
        return mx.array(obj)
    if isinstance(obj, list):
        return [_clone(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone(v) for v in obj)
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and type(obj).__module__.endswith("models.cache") or isinstance(obj, _NoProjectedCache):
        new = copy.copy(obj)
        new.__dict__ = {k: _clone(v) for k, v in obj.__dict__.items()}
        return new
    return obj


@dataclass
class Snapshot:
    tokens: tuple[int, ...]
    cache: list
    logits: mx.array | None
    # HANDOFF 18.6 (CONSUME_SNAPSHOTS): a conversation's own snapshots, taken at the end of a request, are
    # handed to the next request of that conversation instead of copied; snapshots of one request share
    # `group` (the same buffers), so they are counted once and consumed together
    consumable: bool = False
    group: object | None = None

    @property
    def nbytes(self) -> int:
        total = 0

        def walk(o):
            nonlocal total
            if isinstance(o, mx.array):
                total += o.nbytes
            elif isinstance(o, (list, tuple)):
                for v in o:
                    walk(v)
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif hasattr(o, "__dict__"):
                for v in o.__dict__.values():
                    walk(v)

        walk(self.cache)
        return total


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    prefill_seconds: float = 0.0
    completion_tokens: int = 0
    decode_seconds: float = 0.0
    misses: int = 0
    hits: int = 0
    read_bytes: int = 0

    @property
    def decode_tok_s(self) -> float:
        return self.completion_tokens / self.decode_seconds if self.decode_seconds else 0.0


class GlmModel:
    """GLM-5.3-Flash with routed experts streamed from SSD."""

    PREFILL_CHUNK = int(os.environ.get("CACHALOT_GLM_PREFILL_CHUNK", "2048"))

    def __init__(
        self,
        model_path,
        *,
        expert_budget_gib: float = 52.0,
        wired_limit_gib: float | None = None,
        load_workers: int = 8,
        heartbeat_seconds: float = 0.5,
        verbose: bool = True,
    ) -> None:
        t0 = time.perf_counter()
        self.model_path = Path(model_path)
        config = json.loads((self.model_path / "config.json").read_text())
        self.config = TextConfig.from_dict(config["text_config"])
        self._raw_config = config
        self._vision_tower_obj = None
        quant = config.get("quantization") or config.get("quantization_config") or {}

        self.expert_format, index = build_glm_expert_index(self.model_path)
        from cachalot.glm import bank as glm_bank

        bank_path = glm_bank.bank_dir()
        if bank_path is not None:
            index = glm_bank.apply_bank(self.expert_format, index, bank_path)
            print(f"GLM expert bank: {bank_path}", flush=True)
        # layer 45 is the MTP block's MoE, not loaded
        self.expert_index = {k: v for k, v in index.items() if k[0] < self.config.num_hidden_layers}
        sizes = tensor_sizes(self.expert_format)
        expert_bytes = sum(sizes.values())
        budget = int(expert_budget_gib * 1024**3)

        # MLX keeps freed buffers for reuse; unbounded, a long prefill's activations pile up past the wired
        # set and macOS swaps them (HANDOFF 17.1). DeepSeek's runtime caps it at 2 GiB (config.py) as well.
        mx.set_cache_limit(int(float(os.environ.get("CACHALOT_GLM_MLX_CACHE_GIB", "2")) * 1024**3))
        if wired_limit_gib is None:
            wired_limit_gib = float(os.environ.get("CACHALOT_MLX_WIRED_LIMIT_GIB", "80"))
        if wired_limit_gib > 0:
            from cachalot.config import device_memory

            _, recommended = device_memory()  # Metal refuses a limit above its working set
            mx.set_wired_limit(int(min(wired_limit_gib * 1024**3, recommended)))

        self.store = ResidentExpertStore(
            budget,
            ExpertReader(),
            tensor_sizes=sizes,
            # one prefill layer's misses plus the next layer read early (experts.PREFILL_SCAN)
            transient_slots=2 * self.config.n_routed_experts + 16,
            load_workers=load_workers,
            verbose=verbose,
        )
        self.store.format = self.expert_format

        self.model = LanguageModel(self.config)
        n_moe = 0
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer.mlp, Glm5NextMoE):
                activation = layer.mlp.switch_mlp.activation
                # replaced before anything evaluates: the stacked expert parameters are never allocated
                layer.mlp.switch_mlp = StreamingSwitchGLU(
                    i, self.store, self.expert_index, self.expert_format, activation
                )
                n_moe += 1

        self._install_predictors()

        sidecar = self.model_path / NONEXPERT_SIDECAR
        if sidecar.exists():
            # the non-expert weights of another quantisation, in mlx-lm's layout (HANDOFF 18.49)
            weights = self.model.sanitize({_sidecar_key(k): v for k, v in mx.load(str(sidecar)).items() if k.startswith("language_model.")})
        else:
            weights = self.model.sanitize(_remap(load_non_expert_weights(self.model_path)))
        weights = {k[len("language_model."):]: v for k, v in weights.items() if k.startswith("language_model.")}
        group_size, bits, mode = int(quant.get("group_size", 64)), int(quant.get("bits", 4)), quant.get("mode", "affine")

        def quantized(path, module):
            if not (hasattr(module, "to_quantized") and f"{path}.scales" in weights):
                return False
            # a mixed-precision checkpoint's width is read off the tensors (packed columns per scale group), not off the
            # config, whose per-module map is keyed by the unfused names the fused modules were built from
            module_bits = weights[f"{path}.weight"].shape[-1] * 32 // (weights[f"{path}.scales"].shape[-1] * group_size)
            return True if module_bits == bits else {"group_size": group_size, "bits": module_bits, "mode": mode}

        nn.quantize(self.model, group_size=group_size, bits=bits, mode=mode, class_predicate=quantized)
        params = dict(nn.utils.tree_flatten(self.model.parameters()))
        missing = sorted(set(params) - set(weights))
        unexpected = sorted(set(weights) - set(params))
        if missing:
            raise ValueError(f"{len(missing)} model parameters have no weight, e.g. {missing[:5]}")
        self.model.load_weights(list((k, v) for k, v in weights.items() if k in params))
        mx.eval(self.model.parameters())
        self.model.eval()
        self.unused_weights = unexpected
        self.trunk_bytes = sum(v.nbytes for v in params.values())

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.eos_ids = set(config["text_config"].get("eos_token_id") or [])

        self.prefix: list[Snapshot] = []
        self.disk = None  # snapshots.GlmSnapshotStore, attach_snapshot_store()
        self.max_seq_len = int(os.environ.get("CACHALOT_GLM_MAX_SEQ_LEN", "131072"))
        self._lock = threading.RLock()
        self._busy = False
        self._idle_since = time.perf_counter()
        self._stop = threading.Event()
        if heartbeat_seconds > 0:
            threading.Thread(target=self._heartbeat, args=(heartbeat_seconds,), daemon=True, name="glm-heartbeat").start()

        if verbose:
            print(
                f"GLM-5.3-Flash: {len(self.model.layers)} layers ({n_moe} MoE), trunk {self.trunk_bytes / 1024**3:.1f} GiB, "
                f"{len(self.expert_index)} experts x {expert_bytes / 2**20:.2f} MiB, "
                f"{self.store.capacity} resident slots ({self.store.budget_bytes / 1024**3:.1f} GiB), "
                f"loaded in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    def _install_predictors(self) -> None:
        """Per MoE layer, the next MoE layer's router applied to this layer's MoE input (HANDOFF 18.35)."""
        from cachalot.third_party.mlx_vlm.models.glm5_next.language import _expert_select

        layers = self.model.layers

        def make(nxt_moe):
            def predict(x, k):
                gate = nxt_moe.gate
                logits = x.astype(mx.float32) @ gate.weight.T
                return _expert_select(
                    logits, gate.e_score_correction_bias, k, gate.n_group, gate.topk_group,
                    gate.routed_scaling_factor, gate.norm_topk_prob,
                )
            return predict

        for i, layer in enumerate(layers[:-1]):
            nxt = layers[i + 1]
            if isinstance(layer.mlp, Glm5NextMoE) and isinstance(nxt.mlp, Glm5NextMoE):
                layer.mlp.switch_mlp.predict = make(nxt.mlp)

    # -- keep the wired set wired while idle (macOS un-wires an idle Metal queue) -------------------------
    def _heartbeat(self, period: float) -> None:
        probe = mx.zeros((1,))
        while not self._stop.wait(period):
            if self._busy or time.perf_counter() - self._idle_since < period:
                continue
            if self._lock.acquire(blocking=False):
                try:
                    if not self._busy:
                        mx.eval(probe + 1)
                finally:
                    self._lock.release()

    # Messages from background threads (the warm set read back, idle warming). None prints them; the terminal
    # chat collects them instead and shows them in the next turn's summary, not in the line being typed.
    notice = None

    def _notify(self, message: str) -> None:
        if self.notice is not None:
            self.notice(message)
        else:
            print(message, flush=True)

    def generation_defaults(self) -> dict:
        """The checkpoint's `generation_config.json` sampling defaults (temperature, top_p) that it sets."""
        try:
            config = json.loads((Path(self.model_path) / "generation_config.json").read_text())
        except (OSError, ValueError, TypeError):
            return {}
        return {k: float(config[k]) for k in ("temperature", "top_p") if isinstance(config.get(k), (int, float))}

    def close(self) -> None:
        self._stop.set()
        self._stop_idle_warm()
        self.store.close()

    # -- text ---------------------------------------------------------------------------------------------
    # -- the model family's chat format (GlmEngine and `cachalot chat` go through these) -------------------
    @property
    def splitter_cls(self):
        from cachalot.glm.engine import _GlmSplitter

        return _GlmSplitter

    def render_chat(self, messages, *, tools=None, thinking=False, effort=None, add_generation_prompt=True) -> str:
        """The prompt text. GLM's template always opens `<think>`; thinking off closes it at once."""
        from cachalot.glm.engine import THINK_END, _effort

        kwargs = {}
        effort = _effort(effort)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        text = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False, **kwargs
        )
        if add_generation_prompt and not thinking:
            text += THINK_END
        return text

    def parse_tool_calls(self, text: str, tools=None):
        from cachalot.glm.engine import parse_tool_calls

        return parse_tool_calls(text, tools)

    def encode_chat(self, messages, *, tools=None, reasoning_effort=None, add_generation_prompt=True) -> list[int]:
        kwargs = {}
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        text = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False, **kwargs
        )
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def new_cache(self):
        caches = self.model.make_cache()
        out = []
        for layer, cache in zip(self.model.layers, caches):
            if isinstance(cache, CacheList):
                parts = list(cache.caches)
                # the projected cache is the last KVCache of an MLA layer's list (language.py make_cache)
                parts[-1] = _NoProjectedCache()
                cache = CacheList(*parts)
            out.append(cache)
        return out

    @staticmethod
    def snapshot(tokens, cache, logits=None) -> Snapshot:
        return Snapshot(tuple(tokens), _clone(cache), logits)

    # MiniMax (plain KVCache per layer) sets it: see Snapshot.consumable. GLM's caches keep the copying path.
    CONSUME_SNAPSHOTS = False

    @classmethod
    def snapshot_at(cls, tokens, cache, logits=None) -> Snapshot:
        """A snapshot of the first len(tokens) cached positions, over the cache's own buffers (KVCache only:
        positions past `offset` are never read, and a later write to a shared buffer copies it)."""
        snap = cls.snapshot(tokens, cache, logits)
        n = len(snap.tokens)
        for c in snap.cache:
            if not isinstance(c, KVCache) or c.offset < n:
                raise TypeError("snapshot_at needs KVCache layers holding at least the snapshot's tokens")
            c.offset = n
        return snap

    def _consume(self, snap: Snapshot) -> None:
        """Hand a conversation snapshot (and the others over its buffers) to the request that continues it."""
        self.prefix = [p for p in self.prefix
                       if p is not snap and (snap.group is None or p.group is not snap.group)]

    def _prefix_bytes(self) -> int:
        total, seen = 0, set()
        for p in self.prefix:
            if p.group is not None:
                if id(p.group) in seen:
                    continue
                seen.add(id(p.group))
            total += p.nbytes
        return total

    @staticmethod
    def restore(snap: Snapshot) -> list:
        return _clone(snap.cache)

    def set_tracer(self, tracer) -> int:
        """Install (or remove, with None) a routing tracer on every streaming MoE layer; the layers installed.
        MiniMax's GPU-select decode keeps its own routing loop and takes the tracer there too."""
        self._tracer = tracer
        n = 0
        for module in self.model.modules():
            if isinstance(module, StreamingSwitchGLU):
                module.tracer = tracer
                n += 1
        decoder = getattr(self, "gpu_decoder", None)
        if decoder is not None:
            decoder.tracer = tracer
        return n

    def _forward(self, tokens: list[int], cache) -> mx.array:
        out = self.model(mx.array(tokens, dtype=mx.int32)[None], cache=cache)
        return out.logits[:, -1, :]

    # -- vision (HANDOFF 18.39) -------------------------------------------------------------------------------------
    def vision_config(self):
        from cachalot.glm import vision

        return vision.ImageProcessorConfig.from_model(self.model_path)

    def has_vision(self) -> bool:
        # MiniMaxModel subclasses this class without running its __init__: no config, no vision
        return bool(getattr(self, "_raw_config", {}).get("vision_config"))

    def _vision_tower(self):
        if self._vision_tower_obj is None:
            from types import SimpleNamespace

            from cachalot.glm.vision import VisionTower
            from cachalot.third_party.mlx_vlm.models.glm5_next.config import VisionConfig

            cfg = SimpleNamespace(vision_config=VisionConfig.from_dict(self._raw_config["vision_config"]))
            self._vision_tower_obj = VisionTower(self.model_path, cfg)
        return self._vision_tower_obj

    def _forward_span(self, tokens: list[int], start: int, cache, spans) -> mx.array:
        """`_forward` for the chunk of the prompt at [start, start + len(tokens)): where it holds image tokens, the
        embedding rows are the vision tower's (one row per token of the image, in reading order)."""
        end = start + len(tokens)
        hit = [sp for sp in spans if sp.start < end and sp.start + sp.length > start]
        if not hit:
            return self._forward(tokens, cache)
        ids = mx.array(tokens, dtype=mx.int32)[None]
        embeds = self.model.model.embed_tokens(ids)
        tower = self._vision_tower()
        for sp in hit:
            a, b = max(sp.start, start), min(sp.start + sp.length, end)
            rows = tower.features(sp)[a - sp.start:b - sp.start].astype(embeds.dtype)
            embeds[:, a - start:b - start, :] = rows[None]
        out = self.model(None, cache=cache, inputs_embeds=embeds)
        return out.logits[:, -1, :]

    def prefill(self, tokens: list[int], cache) -> mx.array:
        logits = None
        # decode's finished wrong predictions (MiniMax, HANDOFF 18.9) give their transient slots back first
        self.store.expire_predictions()
        self._fit_prefill(len(tokens))
        tracer = getattr(self, "_tracer", None)
        if tracer is not None:
            tracer.forced_phase = "prefill"
        try:
            for start in range(0, len(tokens), self.PREFILL_CHUNK):
                logits = self._forward(tokens[start:start + self.PREFILL_CHUNK], cache)
                mx.eval(logits)
        finally:
            if tracer is not None:
                tracer.forced_phase = None
            self.store.release_prefill()
        return logits

    @staticmethod
    def _sample(logits: mx.array, temperature: float, top_p: float) -> int:
        if temperature <= 0:
            return int(mx.argmax(logits, axis=-1).item())
        probs = mx.softmax(logits.astype(mx.float32) / temperature, axis=-1)[0]
        p = np.array(probs, dtype=np.float64)
        if top_p < 1.0:
            order = np.argsort(-p)
            keep = np.cumsum(p[order]) <= top_p
            keep[0] = True
            mask = np.zeros_like(p, dtype=bool)
            mask[order[keep]] = True
            p = np.where(mask, p, 0.0)
        p /= p.sum()
        return int(np.random.choice(len(p), p=p))

    # -- memory fit (HANDOFF 18.6) ------------------------------------------------------------------------
    # A long context's KV cache (and the snapshot copy of it) comes on top of the expert cache; MiniMax's
    # ~120 KB per token is 2.7 GiB per copy at 25k. At the memory ceiling, with the display on, decode stalls
    # (0.9 tok/s instead of 3.3 at 25k). `_memory_target` is MLX's active memory right after loading plus an
    # allowance; after the first decode token of a request (when the KV copies exist) and every
    # MEMORY_FIT_EVERY tokens, the expert capacity gives back or takes back whole slots to stay at it.
    # None (GLM) disables it.
    _memory_target: int | None = None
    # HANDOFF 18.15 (S2): the expert bytes allowed while a chunk of PREFILL_CHUNK tokens prefills (its activations
    # need the memory); None (GLM) keeps the capacity through a prefill. Chunks of PREFILL_FULL_TOKENS or fewer keep
    # the full capacity, longer ones scale down to this linearly.
    _prefill_budget: int | None = None
    PREFILL_FULL_TOKENS = 2048
    # HANDOFF 18.24: the host watcher (started by the first _host_capacity), the event count the last fit saw, and
    # when it last gave back a slab for one
    _host_watch: HostWatch | None = None
    _watch_seen = 0
    _watch_shrunk = float("-inf")

    def _watch_pending(self) -> bool:
        """A pressure event the fits have not answered yet, and a slab may be given back for it now."""
        w = self._host_watch
        return (w is not None and w.events != self._watch_seen
                and time.monotonic() - self._watch_shrunk >= HOST_SHRINK_EVERY)

    def _host_capacity(self, capacity: int) -> int:
        """The capacity the machine's memory allows (S2): a slab less at warning pressure or under half the floor
        available, else whatever keeps HOST_AVAILABLE_FLOOR available; and (18.17) at most what keeps every
        process's GPU allocations under gpu_ceiling()."""
        _, level = host_memory()
        available = host_available()
        if available < 0:
            return capacity
        pool = self.store.pool
        step = getattr(pool, "slab_slots", MEMORY_FIT_MIN_SLOTS)
        watch = self._host_watch
        if watch is None and HOST_GROW_QUIET > 0:
            watch = self._host_watch = HostWatch(HOST_AVAILABLE_FLOOR)
        now = time.monotonic()
        if level >= 2 or available < HOST_AVAILABLE_FLOOR // 2:
            if watch is not None:
                self._watch_seen, self._watch_shrunk = watch.events, now
            return capacity - step
        if watch is not None:
            if watch.events != self._watch_seen and now - self._watch_shrunk >= HOST_SHRINK_EVERY:
                # warning pressure since the last fit (HANDOFF 18.24): give back a slab
                self._watch_seen, self._watch_shrunk = watch.events, now
                return capacity - step
            if not watch.quiet():
                available = min(available, HOST_AVAILABLE_FLOOR)  # no growth until it has been quiet a while
            else:
                low = watch.min_available()
                if low >= 0:
                    available = min(available, low)
        allowed = capacity + max(0, available - HOST_AVAILABLE_FLOOR) // self.store.expert_bytes
        ceiling = gpu_ceiling()
        if ceiling > 0:
            mx.clear_cache()  # MLX's cached free buffers count in the driver's total, and are not needed
        allocated = gpu_allocated()
        if ceiling > 0 and allocated >= 0:
            ours = mx.get_active_memory() + mx.get_cache_memory()
            # floor division: a negative room gives back whole slots rounded up
            gpu = capacity + (ceiling - max(allocated, ours)) // self.store.expert_bytes
            if gpu < capacity:
                # HANDOFF 18.18 item 8: any overshoot pages (0.3-1.2 GiB over held decode at 2.9-3.5 tok/s), and a
                # slab pool keeps up to a quarter slab of excess, so the GPU term asks for at least a whole slab
                gpu = min(gpu, capacity - step)
            allowed = min(allowed, gpu)
        return allowed

    def _fit_prefill(self, n: int) -> None:
        """Before prefilling `n` tokens (S2): give back the slots a long chunk's activations need; the first decode
        token's _fit_memory takes them back."""
        if self._prefill_budget is None or n <= 0:
            return
        store = self.store
        full = getattr(store, "_full_capacity", store.capacity)
        low = self._prefill_budget // store.expert_bytes
        m = min(n, self.PREFILL_CHUNK)
        f = min(1.0, max(0.0, (m - self.PREFILL_FULL_TOKENS) / max(1, self.PREFILL_CHUNK - self.PREFILL_FULL_TOKENS)))
        want = min(int(full - f * max(0, full - low)), self._host_capacity(store.capacity))
        if want >= store.capacity:
            return
        before = store.capacity
        after = store.set_capacity(want)
        mx.clear_cache()
        if after != before:
            print(f"memory fit: expert slots {before} -> {after} for a {n}-token prefill", flush=True)

    def _fit_memory(self) -> None:
        target = self._memory_target
        if target is None:
            return
        store = self.store
        slot = store.expert_bytes
        if self._prefill_budget is not None:
            mx.clear_cache()  # the prefill's cached buffers are not what the host check should count
        excess = mx.get_active_memory() - target
        full = getattr(store, "_full_capacity", store.capacity)
        # shrink by whole slots rounded up, grow by whole slots rounded down
        want = store.capacity - int(-(-excess // slot)) if excess > 0 else store.capacity + int(-excess // slot)
        want = min(full, want)
        if self._prefill_budget is not None:
            want = min(want, self._host_capacity(store.capacity))
        if abs(want - store.capacity) < MEMORY_FIT_MIN_SLOTS:
            return
        before = store.capacity
        after = store.set_capacity(want)
        mx.clear_cache()
        if after != before:
            print(f"memory fit: expert slots {before} -> {after} "
                  f"(MLX active {mx.get_active_memory() / 1024**3:.1f} GiB, target {target / 1024**3:.1f})", flush=True)

    # -- prefix cache ---------------------------------------------------------------------------------------
    PREFIX_BYTES = int(float(os.environ.get("CACHALOT_GLM_PREFIX_GIB", "3")) * 1024**3)

    def attach_snapshot_store(self, directory, preload: int = 4) -> str:
        """Keep system-block snapshots in `directory` across restarts (HANDOFF 17.1); returns a status line.
        `preload` snapshots are read into memory now, the others when a prompt starts with them (the terminal
        chat passes 0: an agent's 2 GB blocks in that directory would only take slots from the expert cache)."""
        from cachalot.glm.snapshots import GlmSnapshotStore, glm_identity

        t0 = time.perf_counter()
        # files kept on disk; MiniMax's full-attention cache is ~120 KB per token (~2.4 GB at 20k), GLM's ~12 KB
        keep = int(os.environ.get("CACHALOT_SNAPSHOT_KEEP", "32"))
        store = GlmSnapshotStore(directory, glm_identity(self.model_path, self.max_seq_len, self.PREFILL_CHUNK,
                                                          self.NUMERICS_TAG), keep=keep, preload=preload)
        loaded = store.load_all()
        for snap in loaded:
            self._add_prefix(snap)
        self.disk = store
        if loaded and self._prefill_budget is not None:
            # HANDOFF 18.18 item 8: the preloaded snapshots count against the GPU ceiling before the first request
            # would fit the capacity (0.37.1 at startup: all processes' GPU memory 91.3 GiB, swap +2.6 GiB)
            self._fit_memory()
        warm = self.start_warm_set(Path(directory) / "resident-set.json") if WARM_SET else "warm set off"
        return (f"prefix snapshots: {len(loaded)} loaded from {directory} "
                f"({', '.join(str(len(s.tokens)) for s in loaded) + ' tokens' if loaded else 'none'}), "
                f"{len(store.tokens) - len(loaded)} more on disk, in {time.perf_counter() - t0:.2f}s; {warm}")

    # -- warm restart (HANDOFF 18.2) ---------------------------------------------------------------------
    # The resident expert set is written after every request and read back into the cache at startup, in the
    # background, so the first turn after a restart decodes at the last session's hit rate, not a cold one.
    NUMERICS_TAG = ""

    def _warm_identity(self) -> dict:
        return {"model": str(self.model_path), "experts": len(self.expert_index), "format": repr(self.expert_format)}

    def start_warm_set(self, path) -> str:
        self._warm_path = Path(path)
        try:
            data = json.loads(self._warm_path.read_text())
        except (OSError, ValueError):
            return f"warm set: none at {path}"
        if data.get("identity") != self._warm_identity():
            return f"warm set: {path} is another model's, ignored"
        keys = [tuple(k) for k in data.get("keys", []) if tuple(k) in self.expert_index]
        # oldest first, so the preload keeps their recency order; only the newest that fit
        keys = keys[-self.store.capacity:]
        entries = [self.expert_index[k] for k in keys]
        self._warm_pending = entries
        cancel = threading.Event()

        def run():
            t0 = time.perf_counter()
            n = self._read_warm_set(cancel)
            self._notify(f"warm set: {n} experts ({n * self.store.expert_bytes / 2**30:.1f} GiB) "
                         f"read back in {time.perf_counter() - t0:.1f}s"
                         f"{'; the rest after the request' if cancel.is_set() else ''}")

        self._warm_cancel = cancel
        self._warm_thread = threading.Thread(target=run, daemon=True, name="warm-set")
        self._warm_thread.start()
        return f"warm set: reading {len(entries)} experts back in the background"

    def _read_warm_set(self, cancel: threading.Event) -> int:
        """Preload what is left of the warm set into free slots; it is done unless `cancel` stopped it."""
        pending = getattr(self, "_warm_pending", None)
        if not pending:
            return 0
        n = self.store.preload(pending, reserve_fraction=0.0, cancel=cancel)
        if not cancel.is_set():
            self._warm_pending = None
        return n

    # -- idle-time warming (HANDOFF 18.8) ------------------------------------------------------------------
    # After a request, while the server waits for the next one (an agent's tool, a user typing), the resident
    # set is moved towards the experts this process has requested most, so the next turn's prefill and first
    # decode tokens miss less. The next request cancels it (it waits for one batch of reads at most).
    IDLE_WARM = False
    IDLE_WARM_DELAY = 0.2  # seconds after a request before warming starts

    def warm_now(self, cancel: threading.Event | None = None) -> tuple[int, int]:
        """Warm the resident set towards the most-requested experts; (read, evicted)."""
        counts = self.store.use_counts
        ranked = sorted((k for k in counts if k in self.expert_index), key=lambda k: -counts[k])
        return self.store.warm([self.expert_index[k] for k in ranked], cancel)

    def _start_idle_warm(self) -> None:
        if not self.IDLE_WARM and not getattr(self, "_warm_pending", None):
            return
        cancel = threading.Event()

        def run():
            if cancel.wait(self.IDLE_WARM_DELAY):
                return
            if getattr(self, "_warm_pending", None):
                # M27: the warm set a request interrupted, first
                t0 = time.perf_counter()
                n = self._read_warm_set(cancel)
                if n:
                    self._notify(f"warm set: {n} more experts in {time.perf_counter() - t0:.1f}s"
                                 f"{' (cancelled)' if cancel.is_set() else ''}")
                if cancel.is_set() or not self.IDLE_WARM:
                    return
            t0 = time.perf_counter()
            read, _ = self.warm_now(cancel)
            if read:
                self._notify(f"idle warm: {read} experts in {time.perf_counter() - t0:.1f}s"
                             f"{' (cancelled)' if cancel.is_set() else ''}")

        self._idle_warm = (cancel, threading.Thread(target=run, daemon=True, name="idle-warm"))
        self._idle_warm[1].start()

    def _stop_idle_warm(self) -> None:
        pending = getattr(self, "_idle_warm", None)
        if pending is not None:
            pending[0].set()
            pending[1].join()
            self._idle_warm = None

    def _wait_warm_set(self) -> None:
        """Before a request: stop the startup warm set's reads (M27; WARM_SET_YIELD=0 waits for all of them)."""
        thread = getattr(self, "_warm_thread", None)
        if thread is not None:
            if WARM_SET_YIELD:
                self._warm_cancel.set()
            thread.join()
            self._warm_thread = None

    def _save_warm_set(self) -> None:
        path = getattr(self, "_warm_path", None)
        if path is None:
            return
        try:
            data = {"identity": self._warm_identity(), "keys": [list(k) for k in self.store.resident_keys()]}
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(path)
        except OSError as exc:
            print(f"warm set not saved: {exc}", flush=True)

    def _persist(self, snap: Snapshot) -> None:
        if self.disk is None:
            return
        try:
            self.disk.persist(snap)
        except Exception as exc:  # a full disk must not fail the request
            print(f"prefix snapshot not saved: {exc}", flush=True)

    def _find_prefix(self, tokens: tuple[int, ...]) -> Snapshot | None:
        best = None
        for snap in self.prefix:
            n = len(snap.tokens)
            if n > len(tokens) or tokens[:n] != snap.tokens:
                continue
            if n == len(tokens) and snap.logits is None:
                continue
            if best is None or n > len(best.tokens):
                best = snap
        if self.disk is not None:
            try:
                fetched = self.disk.fetch(tokens, len(best.tokens) if best is not None else 0)
                if fetched is not None and len(fetched.tokens) < len(tokens):
                    self._add_prefix(fetched)
                    best = fetched
                self.disk.on_find(tokens, [p for p in self.prefix if tokens[:len(p.tokens)] == p.tokens])
            except Exception as exc:
                print(f"prefix snapshot lookup failed: {exc}", flush=True)
        if best is not None:  # most recently used last
            self.prefix.remove(best)
            self.prefix.append(best)
        return best

    # HANDOFF 18.22: MLX's buffer cache limit while a reply decodes (None keeps the prefill's). The memory governor
    # sizes the expert cache with the buffer cache emptied; decode then refilled it to its 2 GiB cap, which put all
    # processes' GPU memory above the working set, and the next short prefill ran 3-5x slower in 12 of 35 turns (0 of
    # 20 capped).
    DECODE_CACHE_BYTES: int | None = None

    # HANDOFF 18.22: a system block the disk store holds is dropped from memory after each request (the next prompt
    # that starts with it loads it back, ~0.4 s); in memory it held ~2.4 GiB (~115 of MiniMax's expert slots) through
    # every decode token of an agent session. Off for GLM (its blocks are ~0.25 GiB).
    SPILL_PERSISTED = False

    def _spill_persisted(self) -> None:
        if not self.SPILL_PERSISTED or self.disk is None:
            return
        on_disk = set(self.disk.tokens.values())
        kept = [p for p in self.prefix if p.tokens not in on_disk]
        if len(kept) != len(self.prefix):
            self.prefix = kept
            mx.clear_cache()

    def _add_prefix(self, snap: Snapshot) -> None:
        self.prefix = [p for p in self.prefix if p.tokens != snap.tokens]
        self.prefix.append(snap)
        while len(self.prefix) > 1 and self._prefix_bytes() > self.PREFIX_BYTES:
            self.prefix.pop(0)

    def stream(
        self,
        prompt_tokens: list[int],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        cancel: threading.Event | None = None,
        boundary: int = 0,
        images=None,
    ):
        """Yields ("prefill", reused, seconds), ("token", id) ..., ("done", finish, decode_seconds).

        `images` (cachalot.glm.vision.ImageSpan list) names the runs of image tokens in `prompt_tokens`; saved
        prefixes are keyed on a copy whose image tokens are pseudo tokens of the image's hash (HANDOFF 18.39)."""
        with self._lock:
            self._busy = True
            restore_cache = None
            try:
                self._stop_idle_warm()
                self._wait_warm_set()
                spans = list(images or [])
                if spans:
                    from cachalot.glm import vision as _vision

                    tokens = tuple(_vision.key_tokens(list(prompt_tokens), spans))
                else:
                    tokens = tuple(prompt_tokens)
                real = list(prompt_tokens)
                t0 = time.perf_counter()
                snap = self._find_prefix(tokens)
                if snap is not None:
                    cache, reused, logits = self.restore(snap), len(snap.tokens), snap.logits
                    if self.CONSUME_SNAPSHOTS and snap.consumable:
                        self._consume(snap)
                    snap = None  # a reference kept here would make the first write copy the whole cache
                else:
                    cache, reused, logits = self.new_cache(), 0, None
                pos = reused
                self._fit_prefill(len(tokens) - reused)
                cuts = sorted({c for c in (boundary,) if reused < c < len(tokens)} | {len(tokens)})
                for cut in cuts:
                    while pos < cut:
                        if cancel is not None and cancel.is_set():
                            if self.CONSUME_SNAPSHOTS and pos > 0:
                                # what was consumed or prefilled so far stays reusable
                                kept = self.snapshot_at(tokens[:pos], cache, logits if pos == reused else None)
                                kept.consumable = True
                                self._add_prefix(kept)
                            yield ("done", "cancel", 0.0)
                            return
                        end = min(pos + self.PREFILL_CHUNK, cut)
                        logits = self._forward_span(real[pos:end], pos, cache, spans) if spans \
                            else self._forward(real[pos:end], cache)
                        mx.eval(logits)
                        pos = end
                    if cut < len(tokens):
                        block = self.snapshot(tokens[:cut], cache, None)
                        self._add_prefix(block)
                        if cut == boundary:
                            self._persist(block)
                self.store.release_prefill()
                prompt_logits = logits
                if reused < len(tokens) and not self.CONSUME_SNAPSHOTS:
                    self._add_prefix(self.snapshot(tokens, cache, logits))
                yield ("prefill", reused, time.perf_counter() - t0)
                if self.DECODE_CACHE_BYTES is not None:
                    restore_cache = mx.set_cache_limit(self.DECODE_CACHE_BYTES)
                t1 = time.perf_counter()
                out: list[int] = []
                finish = "length"
                for _ in range(max_new_tokens):
                    if cancel is not None and cancel.is_set():
                        finish = "cancel"
                        break
                    token = self._sample(logits, temperature, top_p)
                    out.append(token)
                    if token in self.eos_ids:
                        finish = "stop"
                        yield ("token", token)
                        break
                    yield ("token", token)
                    logits = self._forward([token], cache)
                    mx.eval(logits)
                    if len(out) % MEMORY_FIT_EVERY == 1 or self._watch_pending():
                        self._fit_memory()
                if not self.CONSUME_SNAPSHOTS:
                    if out and finish != "cancel":
                        # prompt + reply without the final token, which was never fed: the next turn's
                        # prompt repeats the reply and continues from it
                        fed = tokens + tuple(out[:-1]) if finish == "stop" else tokens + tuple(out)
                        self._add_prefix(self.snapshot(fed, cache, None if finish == "stop" else logits))
                else:
                    # after the reply, not before it: a prompt snapshot held through decode would make the
                    # first decode write copy the whole KV cache (HANDOFF 18.6). The prompt snapshot (for a
                    # retry) is the same buffers trimmed to the prompt; the reply snapshot continues the turn.
                    group = object()
                    prompt_snap = self.snapshot_at(tokens, cache, prompt_logits)
                    prompt_snap.consumable, prompt_snap.group = True, group
                    self._add_prefix(prompt_snap)
                    if out and finish != "cancel":
                        fed = tokens + tuple(out[:-1]) if finish == "stop" else tokens + tuple(out)
                        reply = self.snapshot_at(fed, cache, None if finish == "stop" else logits)
                        reply.consumable, reply.group = True, group
                        self._add_prefix(reply)
                yield ("done", finish, time.perf_counter() - t1)
            finally:
                if restore_cache is not None:
                    mx.set_cache_limit(restore_cache)
                self.store.release_prefill()
                self._spill_persisted()
                self._save_warm_set()
                self._idle_since = time.perf_counter()
                self._busy = False
                self._start_idle_warm()

    def generate(
        self,
        prompt_tokens: list[int],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        on_token=None,
        stats: GenerationStats | None = None,
    ) -> list[int]:
        stats = stats if stats is not None else GenerationStats()
        with self._lock:
            self._busy = True
            try:
                self._stop_idle_warm()
                self._wait_warm_set()
                s0 = self.store.stats()
                cache = self.new_cache()
                t0 = time.perf_counter()
                logits = self.prefill(prompt_tokens, cache)
                stats.prompt_tokens = len(prompt_tokens)
                stats.prefill_seconds = time.perf_counter() - t0
                out: list[int] = []
                t1 = time.perf_counter()
                for _ in range(max_new_tokens):
                    token = self._sample(logits, temperature, top_p)
                    out.append(token)
                    if on_token is not None:
                        on_token(token)
                    if token in self.eos_ids:
                        break
                    logits = self._forward([token], cache)
                    mx.eval(logits)
                stats.completion_tokens = len(out)
                stats.decode_seconds = time.perf_counter() - t1
                s1 = self.store.stats()
                stats.hits = s1.cache_hits - s0.cache_hits
                stats.misses = s1.cache_misses - s0.cache_misses
                stats.read_bytes = s1.ssd_bytes_read - s0.ssd_bytes_read
                return out
            finally:
                self._idle_since = time.perf_counter()
                self._busy = False
                self._start_idle_warm()
