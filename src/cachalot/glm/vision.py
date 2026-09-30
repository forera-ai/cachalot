"""
GLM-5.3-Flash vision (HANDOFF 18.39): image preprocessing, the vision tower, and the token expansion.

The checkpoint carries a 24-block ViT (`model.visual.*`, bf16, ~0.9 GiB). The chat template writes one
`<|begin_of_image|><|image|><|end_of_image|>` per image part; each `<|image|>` becomes one token per merged 2x2
patch group (`grid_h * grid_w / 4` of them), and the tower's output rows replace those tokens' embeddings. The language
model has no positional encoding of its own (NoPE attention, linear-attention layers), so nothing else changes.

The preprocessing is a port of mlx-vlm's `Glm5NextImageProcessor` (commit ad4a3cc, MIT) without its Transformers
dependencies: an aligned canvas under a token budget (`smart_resize`), PIL bicubic resize, zero padding, rescale and
normalise, then patches ordered by 2x2 merge group with the temporal axis repeated. Videos are not supported.

A prompt's cache key replaces every image token by a pseudo token derived from the image's content hash and the
token's index in its span (`key_tokens`), so two different images of one size never share a saved prefix while the same
image in a later turn does.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

IMAGE_TOKEN_ID = 154854
FEATURE_CACHE = 8  # images whose tower output is kept (a few MiB each)


def smart_resize(num_frames, height, width, temporal_factor=2, factor=28, min_pixels=16, max_pixels=8000):
    """The aligned canvas (height, width) for an image under a token budget (mlx-vlm's `smart_resize`)."""
    if min(num_frames, height, width, temporal_factor, factor) <= 0:
        raise ValueError("Image dimensions and alignment factors must be positive.")
    if min_pixels <= 0 or max_pixels <= 0 or min_pixels > max_pixels:
        raise ValueError("Expected 0 < min_pixels <= max_pixels.")
    per_token = temporal_factor * factor**2
    min_pixels *= per_token
    max_pixels *= per_token

    def align(value, f):
        return math.ceil(value / f) * f

    frames = max(temporal_factor, round(num_frames / temporal_factor) * temporal_factor)

    def fit_within_budget():
        minimum = frames * factor**2
        if max_pixels < minimum:
            raise ValueError(f"max_pixels={max_pixels} is too small; at least {minimum} is required.")
        low, high, best = 1, height, (factor, factor)
        while low <= high:
            content_h = (low + high) // 2
            content_w = max(1, math.floor(width * content_h / height))
            candidate = (align(content_h, factor), align(content_w, factor))
            if frames * candidate[0] * candidate[1] <= max_pixels:
                best = candidate
                low = content_h + 1
            else:
                high = content_h - 1
        return best

    aligned_h, aligned_w = align(height, factor), align(width, factor)
    pixels = frames * aligned_h * aligned_w
    if pixels < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_h = align(max(1, math.ceil(height * scale)), factor)
        aligned_w = align(max(1, math.ceil(width * scale)), factor)
        pixels = frames * aligned_h * aligned_w
    if pixels > max_pixels:
        aligned_h, aligned_w = fit_within_budget()
    return aligned_h, aligned_w


@dataclass(frozen=True)
class ImageProcessorConfig:
    patch_size: int = 14
    temporal_patch_size: int = 2
    merge_size: int = 2
    patch_expand_factor: int = 1
    min_image_tokens: int = 16
    max_image_tokens: int = 8000
    image_mean: tuple = (0.48145466, 0.4578275, 0.40821073)
    image_std: tuple = (0.26862954, 0.26130258, 0.27577711)

    @classmethod
    def from_model(cls, model_path) -> ImageProcessorConfig:
        path = Path(model_path) / "processor_config.json"
        if not path.exists():
            return cls()
        raw = dict(json.loads(path.read_text()).get("image_processor", {}))
        fields = cls.__dataclass_fields__
        values = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in fields}
        return cls(**values)


def _geometry(height, width, cfg: ImageProcessorConfig):
    factor = cfg.patch_size * cfg.merge_size * cfg.patch_expand_factor
    per_token = cfg.temporal_patch_size * factor**2
    target_h, target_w = smart_resize(
        cfg.temporal_patch_size, height, width, temporal_factor=cfg.temporal_patch_size, factor=factor,
        min_pixels=cfg.min_image_tokens, max_pixels=cfg.max_image_tokens,
    )
    scale = min(target_h / height, target_w / width)
    if cfg.temporal_patch_size * height * width >= cfg.min_image_tokens * per_token:
        scale = min(1.0, scale)
    content_h = max(1, min(target_h, math.floor(height * scale)))
    content_w = max(1, min(target_w, math.floor(width * scale)))
    return target_h, target_w, content_h, content_w


def image_tokens(height, width, cfg: ImageProcessorConfig) -> int:
    """How many `<|image|>` tokens an image of this size occupies."""
    target_h, target_w, _, _ = _geometry(height, width, cfg)
    return (target_h // cfg.patch_size) * (target_w // cfg.patch_size) // cfg.merge_size**2


def preprocess(image, cfg: ImageProcessorConfig) -> tuple[np.ndarray, list[int]]:
    """(patches float32 [grid_h * grid_w, 3 * temporal * patch * patch], [1, grid_h, grid_w]) for a PIL image."""
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size
    target_h, target_w, content_h, content_w = _geometry(height, width, cfg)
    if (content_h, content_w) != (height, width):
        image = image.resize((content_w, content_h), Image.BICUBIC)
    array = np.asarray(image).transpose(2, 0, 1)  # CHW uint8
    array = np.pad(array, ((0, 0), (0, target_h - content_h), (0, target_w - content_w)))
    array = array.astype(np.float32) * (1 / 255.0)
    mean = np.asarray(cfg.image_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(cfg.image_std, dtype=np.float32)[:, None, None]
    array = (array - mean) / std
    channels, patch, merge, temporal = 3, cfg.patch_size, cfg.merge_size, cfg.temporal_patch_size
    grid_h, grid_w = target_h // patch, target_w // patch
    patches = array.reshape(channels, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
    patches = np.broadcast_to(patches[:, :, :, :, :, None], (*patches.shape[:5], temporal, *patches.shape[5:]))
    return patches.reshape(grid_h * grid_w, channels * temporal * patch * patch), [1, grid_h, grid_w]


@dataclass
class ImageInput:
    """One image of a prompt, before its tokens are placed."""

    digest: str
    patches: np.ndarray
    grid: list[int]
    tokens: int


@dataclass
class ImageSpan:
    """One image's run of `<|image|>` tokens in the expanded prompt."""

    start: int
    length: int
    digest: str
    patches: np.ndarray = field(repr=False)
    grid: list[int] = field(default_factory=list)


def load_inputs(records: list[dict], cfg: ImageProcessorConfig) -> list[ImageInput]:
    import io

    from PIL import Image

    from cachalot.model.image_processor_mlx import load_image_bytes

    out = []
    for record in records:
        data = load_image_bytes(record)
        with Image.open(io.BytesIO(data)) as source:
            patches, grid = preprocess(source, cfg)
        out.append(ImageInput(hashlib.sha256(data).hexdigest(), patches, grid, grid[1] * grid[2] // cfg.merge_size**2))
    return out


def image_records(messages: list[dict]) -> list[dict]:
    """The image parts of OpenAI-style messages in prompt order, as records `load_image_bytes` reads."""
    records = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in ("image_url", "image", "input_image"):
                continue
            url = part.get("image_url", part.get("url"))
            if isinstance(url, dict):
                url = url.get("url")
            if isinstance(url, str) and url:
                records.append({"url": url})
            elif isinstance(part.get("source"), dict) or part.get("data") is not None:
                records.append({k: part[k] for k in ("source", "data") if k in part})
            else:
                raise ValueError("an image part carries no url or data")
    return records


def expand(tokens: list[int], inputs: list[ImageInput]) -> tuple[list[int], list[ImageSpan]]:
    """Replace each `<|image|>` token by its image's run of them; the spans in the expanded prompt."""
    slots = [i for i, t in enumerate(tokens) if t == IMAGE_TOKEN_ID]
    if len(slots) != len(inputs):
        raise ValueError(f"the prompt has {len(slots)} image markers for {len(inputs)} images")
    out, spans, prev = [], [], 0
    for slot, image in zip(slots, inputs, strict=True):
        out.extend(tokens[prev:slot])
        spans.append(ImageSpan(len(out), image.tokens, image.digest, image.patches, image.grid))
        out.extend([IMAGE_TOKEN_ID] * image.tokens)
        prev = slot + 1
    out.extend(tokens[prev:])
    return out, spans


def key_tokens(tokens: list[int], spans: list[ImageSpan]) -> list[int]:
    """`tokens` with every image token replaced by a negative pseudo token of its image and index (cache keys only)."""
    key = list(tokens)
    for span in spans:
        base = int(span.digest[:12], 16)
        for j in range(span.length):
            key[span.start + j] = -1 - ((base + j * 1_000_003) % (1 << 30))
    return key


class VisionTower:
    """The tower, loaded on the first image, and a small cache of its outputs by image hash."""

    def __init__(self, model_path, config) -> None:
        self.model_path, self.config = Path(model_path), config
        self._tower = None
        self._lock = threading.Lock()
        self._features: OrderedDict[str, mx.array] = OrderedDict()

    def _load(self):
        from cachalot.glm.model import load_non_expert_weights
        from cachalot.third_party.mlx_vlm.models.glm5_next.vision import VisionModel

        tower = VisionModel(self.config.vision_config)
        weights = load_non_expert_weights(self.model_path, lambda n: n.startswith("model.visual."))
        weights = tower.sanitize({k[len("model.visual."):]: v for k, v in weights.items()})
        tower.load_weights(list(weights.items()))
        mx.eval(tower.parameters())
        tower.eval()
        return tower

    def features(self, span: ImageSpan) -> mx.array:
        """The tower's rows [span.length, hidden] for an image, computed once per image."""
        with self._lock:
            cached = self._features.get(span.digest)
            if cached is not None:
                self._features.move_to_end(span.digest)
                return cached
            if self._tower is None:
                self._tower = self._load()
            dtype = self._tower.patch_embed.proj.weight.dtype
            rows = self._tower(mx.array(span.patches).astype(dtype), mx.array([span.grid], dtype=mx.int64))
            mx.eval(rows)
            if rows.shape[0] != span.length:
                raise ValueError(f"the tower gave {rows.shape[0]} rows for {span.length} image tokens")
            self._features[span.digest] = rows
            while len(self._features) > FEATURE_CACHE:
                self._features.popitem(last=False)
            return rows
