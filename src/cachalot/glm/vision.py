"""
GLM-5.3-Flash vision (HANDOFF 18.39): image preprocessing, the vision tower, and the token expansion.

The checkpoint carries a 24-block ViT (`model.visual.*`, bf16, ~0.9 GiB). The chat template writes one
`<|begin_of_image|><|image|><|end_of_image|>` per image part; each `<|image|>` becomes one token per merged 2x2
patch group (`grid_h * grid_w / 4` of them), and the tower's output rows replace those tokens' embeddings. The language
model has no positional encoding of its own (NoPE attention, linear-attention layers), so nothing else changes.

The preprocessing is a port of mlx-vlm's `Glm5NextImageProcessor` (commit ad4a3cc, MIT) without its Transformers
dependencies: an aligned canvas under a token budget (`smart_resize`), PIL bicubic resize, zero padding, rescale and
normalise, then patches ordered by 2x2 merge group with the temporal axis repeated.

Video (HANDOFF 18.48) is a port of `Glm5NextVideoProcessor`: frames sampled at the checkpoint's 2 fps (`sample_indices`), decoded with ffmpeg,
resized under a token budget for the whole clip, and cut into temporal steps of two frames. Each step is one image-like span through the same tower
(mlx-vlm gives the tower one `[1, h, w]` grid per step), and the `<|video|>` marker expands to
`<|begin_of_image|>` + that step's `<|image|>` run + `<|end_of_image|>` + the step's `"<seconds> seconds"` text, one block per step.

A prompt's cache key replaces every image token by a pseudo token derived from the image's content hash and the
token's index in its span (`key_tokens`), so two different images of one size never share a saved prefix while the same
image in a later turn does.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

IMAGE_TOKEN_ID = 154854
VIDEO_TOKEN_ID = 154855
FEATURE_CACHE = 192  # image and video-step tower outputs kept (about 1 MiB each at 256 tokens)
VIDEO_MAX_TOKENS = int(os.environ.get("CACHALOT_GLM_VIDEO_MAX_TOKENS", "4000"))  # `<|image|>` tokens of a whole clip (checkpoint ceiling 240,000)
VIDEO_MAX_FRAMES = int(os.environ.get("CACHALOT_GLM_VIDEO_MAX_FRAMES", "128"))  # sampled frames (mlx-vlm's ceiling 2048)


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


def _geometry(height, width, cfg: ImageProcessorConfig, num_frames: int | None = None):
    num_frames = cfg.temporal_patch_size if num_frames is None else num_frames
    factor = cfg.patch_size * cfg.merge_size * cfg.patch_expand_factor
    per_token = cfg.temporal_patch_size * factor**2
    target_h, target_w = smart_resize(
        num_frames, height, width, temporal_factor=cfg.temporal_patch_size, factor=factor,
        min_pixels=cfg.min_image_tokens, max_pixels=cfg.max_image_tokens,
    )
    scale = min(target_h / height, target_w / width)
    if num_frames * height * width >= cfg.min_image_tokens * per_token:
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


@dataclass
class VideoInput:
    """One video of a prompt: its temporal steps, each an image-like input, and the second each step starts at."""

    steps: list
    timestamps: list


def sample_indices(source_fps: float, total_frames: int, fps: float, max_frames: int, duration: float | None = None) -> np.ndarray:
    """The frame numbers `Glm5NextVideoProcessor.sample_frames` picks (an even count, duplicates removed); `duration` is the loader's
    frames / fps when known (what Transformers' metadata carries), else the processor's own estimate."""
    duration = duration or round((total_frames - 1) / source_fps) + 1
    count = min(int(duration * fps), max_frames)
    if total_frames < count:
        indices = np.linspace(0, total_frames - 1, count, dtype=int).tolist()
    else:
        indices, current, interval = [], 0.0, 1 / fps
        for index in range(total_frames):
            if index / source_fps >= current:
                current += interval
                indices.append(index)
                if current >= int(duration):
                    break
    if len(indices) < count:
        start = indices[0] if indices else 0
        end = indices[-1] if indices else max(total_frames - 1, 0)
        indices = np.linspace(start, end, count, dtype=int).tolist()
    elif len(indices) > count:
        indices = np.linspace(0, total_frames - 1, count, dtype=int).tolist()
    indices = np.asarray(list(dict.fromkeys(indices)), dtype=int)
    if len(indices) == 0:  # a clip under half a second: the processor would sample nothing
        indices = np.asarray([0], dtype=int)
    if len(indices) & 1:
        indices = np.append(indices, indices[-1])
    return indices


def _ffmpeg(name: str) -> str:
    path = shutil.which(name) or f"/opt/homebrew/bin/{name}"
    if not os.path.exists(path):
        raise ValueError(f"video needs {name} on the PATH (brew install ffmpeg)")
    return path


def decode_frames(path: str, fps_target: float, max_frames: int):
    """(frames uint8 [n, h, w, 3], timestamps in seconds) of the sampled frames of a video file, through ffmpeg."""
    from PIL import Image

    probe = json.loads(subprocess.run(
        [_ffmpeg("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,nb_frames,duration:format=duration",
         "-of", "json", path], capture_output=True, check=True, text=True).stdout)
    stream = (probe.get("streams") or [{}])[0]
    num, _, den = str(stream.get("r_frame_rate", "0/1")).partition("/")
    source_fps = float(num) / float(den or 1)
    if source_fps <= 0:
        raise ValueError("the video has no frame rate")
    duration = float(stream.get("duration") or probe.get("format", {}).get("duration") or 0)
    nb = stream.get("nb_frames")
    total = int(nb) if nb not in (None, "N/A") and str(nb).isdigit() else int(round(duration * source_fps))
    if total <= 0:
        raise ValueError("the video has no frames")
    indices = sample_indices(source_fps, total, fps_target, max_frames, total / source_fps)
    wanted = sorted(set(int(i) for i in indices))
    select = "+".join(f"eq(n\\,{i})" for i in wanted)
    with tempfile.TemporaryDirectory(prefix="cachalot-video-") as tmp:
        subprocess.run([_ffmpeg("ffmpeg"), "-v", "error", "-i", path, "-vf", f"select='{select}'", "-fps_mode", "passthrough",
                        os.path.join(tmp, "f_%06d.png")], check=True, capture_output=True)
        files = sorted(os.listdir(tmp))
        if len(files) != len(wanted):
            raise ValueError(f"ffmpeg gave {len(files)} frames for {len(wanted)} sampled")
        by_index = {}
        for index, name in zip(wanted, files, strict=True):
            with Image.open(os.path.join(tmp, name)) as frame:
                by_index[index] = np.asarray(frame.convert("RGB"))
    frames = np.stack([by_index[int(i)] for i in indices])
    return frames, [float(i) / source_fps for i in indices]


def preprocess_video(frames: np.ndarray, cfg: ImageProcessorConfig):
    """(patches [grid_t * grid_h * grid_w, 3 * temporal * patch * patch], [grid_t, grid_h, grid_w]) for uint8 frames [n, h, w, 3]."""
    from PIL import Image

    num_frames, height, width, _ = frames.shape
    target_h, target_w, content_h, content_w = _geometry(height, width, cfg, num_frames)
    if (content_h, content_w) != (height, width):
        frames = np.stack([np.asarray(Image.fromarray(f).resize((content_w, content_h), Image.BICUBIC)) for f in frames])
    array = frames.transpose(0, 3, 1, 2)
    array = np.pad(array, ((0, 0), (0, 0), (0, target_h - content_h), (0, target_w - content_w)))
    array = array.astype(np.float32) * (1 / 255.0)
    mean = np.asarray(cfg.image_mean, dtype=np.float32)[None, :, None, None]
    std = np.asarray(cfg.image_std, dtype=np.float32)[None, :, None, None]
    array = (array - mean) / std
    channels, patch, merge, temporal = 3, cfg.patch_size, cfg.merge_size, cfg.temporal_patch_size
    if pad := -num_frames % temporal:
        array = np.concatenate([array, np.repeat(array[-1:], pad, axis=0)])
        num_frames += pad
    grid_t, grid_h, grid_w = num_frames // temporal, target_h // patch, target_w // patch
    patches = array.reshape(grid_t, temporal, channels, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(grid_t * grid_h * grid_w, channels * temporal * patch * patch), [grid_t, grid_h, grid_w]


def load_video(data: bytes, cfg: ImageProcessorConfig, fps: float = 2.0) -> VideoInput:
    import dataclasses

    digest = hashlib.sha256(data).hexdigest()
    with tempfile.NamedTemporaryFile(prefix="cachalot-video-", suffix=".bin") as handle:
        handle.write(data)
        handle.flush()
        frames, stamps = decode_frames(handle.name, fps, VIDEO_MAX_FRAMES)
    patches, (grid_t, grid_h, grid_w) = preprocess_video(frames, dataclasses.replace(cfg, max_image_tokens=VIDEO_MAX_TOKENS))
    rows, per_frame = grid_h * grid_w, grid_h * grid_w // cfg.merge_size**2
    key = f"{VIDEO_MAX_TOKENS}/{VIDEO_MAX_FRAMES}/{fps}"
    steps = [ImageInput(hashlib.sha256(f"{digest}/{key}/{k}".encode()).hexdigest(), patches[k * rows:(k + 1) * rows], [1, grid_h, grid_w], per_frame)
             for k in range(grid_t)]
    stamps = list(stamps[::cfg.temporal_patch_size])
    stamps = (stamps + [stamps[-1]] * grid_t)[:grid_t]
    return VideoInput(steps, stamps)


def load_inputs(records: list[dict], cfg: ImageProcessorConfig) -> list:
    import io

    from PIL import Image

    from cachalot.model.image_processor_mlx import load_image_bytes

    out = []
    for record in records:
        data = load_image_bytes(record)
        if record.get("kind") == "video":
            out.append(load_video(data, cfg))
            continue
        with Image.open(io.BytesIO(data)) as source:
            patches, grid = preprocess(source, cfg)
        out.append(ImageInput(hashlib.sha256(data).hexdigest(), patches, grid, grid[1] * grid[2] // cfg.merge_size**2))
    return out


VIDEO_PART_TYPES = ("video_url", "video", "input_video")


def image_records(messages: list[dict]) -> list[dict]:
    """The image parts of OpenAI-style messages in prompt order, as records `load_image_bytes` reads."""
    records = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind in VIDEO_PART_TYPES:
                url = part.get("video_url", part.get("url"))
                if isinstance(url, dict):
                    url = url.get("url")
                if isinstance(url, str) and url:
                    records.append({"url": url, "kind": "video"})
                elif isinstance(part.get("source"), dict) or part.get("data") is not None:
                    records.append({"kind": "video", **{k: part[k] for k in ("source", "data") if k in part}})
                else:
                    raise ValueError("a video part carries no url or data")
                continue
            if kind not in ("image_url", "image", "input_image"):
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


def expand(tokens: list[int], inputs: list, tokenizer=None) -> tuple[list[int], list[ImageSpan]]:
    """Replace each `<|image|>` token by its image's run of them and each `<|video|>` token by its steps' blocks; the spans in the expanded prompt."""
    slots = [i for i, t in enumerate(tokens) if t in (IMAGE_TOKEN_ID, VIDEO_TOKEN_ID)]
    if len(slots) != len(inputs):
        raise ValueError(f"the prompt has {len(slots)} image and video markers for {len(inputs)} inputs")
    out, spans, prev = [], [], 0
    for slot, item in zip(slots, inputs, strict=True):
        is_video = isinstance(item, VideoInput)
        if is_video != (tokens[slot] == VIDEO_TOKEN_ID):
            raise ValueError("an image part sits at a video marker or the reverse")
        out.extend(tokens[prev:slot])
        if is_video:
            if tokenizer is None:
                raise ValueError("expanding a video needs the tokenizer")
            begin = tokenizer.convert_tokens_to_ids("<|begin_of_image|>")
            end = tokenizer.convert_tokens_to_ids("<|end_of_image|>")
            for step, stamp in zip(item.steps, item.timestamps, strict=True):
                out.append(begin)
                spans.append(ImageSpan(len(out), step.tokens, step.digest, step.patches, step.grid))
                out.extend([IMAGE_TOKEN_ID] * step.tokens)
                out.append(end)
                out.extend(tokenizer.encode(f"{stamp:.1f} seconds", add_special_tokens=False))
        else:
            spans.append(ImageSpan(len(out), item.tokens, item.digest, item.patches, item.grid))
            out.extend([IMAGE_TOKEN_ID] * item.tokens)
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
