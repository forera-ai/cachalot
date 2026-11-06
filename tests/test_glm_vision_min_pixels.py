"""CACHALOT_GLM_VISION_MIN_PIXELS: off by default, the upscale it plans for still images, video untouched, and the digest it keys rows by
(no tower, no model)."""
from __future__ import annotations

import base64
import hashlib
import io

from PIL import Image

from cachalot.glm import vision

ON = vision.ImageProcessorConfig(upscale_min_pixels=810000)
OFF = vision.ImageProcessorConfig()


def test_off_by_default_and_when_empty_or_zero(monkeypatch, tmp_path):
    monkeypatch.delenv("CACHALOT_GLM_VISION_MIN_PIXELS", raising=False)
    assert vision.min_pixels_from_env() is None
    assert vision.ImageProcessorConfig.from_model(tmp_path) == OFF
    for off in ("", " ", "0"):
        monkeypatch.setenv("CACHALOT_GLM_VISION_MIN_PIXELS", off)
        assert vision.min_pixels_from_env() is None


def test_from_model_reads_the_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHALOT_GLM_VISION_MIN_PIXELS", "810000")
    assert vision.ImageProcessorConfig.from_model(tmp_path).upscale_min_pixels == 810000
    (tmp_path / "processor_config.json").write_text('{"image_processor": {"patch_size": 14}}')
    assert vision.ImageProcessorConfig.from_model(tmp_path).upscale_min_pixels == 810000


def test_the_tested_screenshot_gets_the_tested_token_count():
    assert vision.image_tokens(512, 492, OFF) == 342
    assert vision.image_tokens(512, 492, ON) == 1056  # the padded copy of HANDOFF 18.269
    patches, grid = vision.preprocess(Image.new("RGB", (492, 512), "white"), ON)
    assert grid[1] * grid[2] // ON.merge_size**2 == 1056
    assert patches.shape[0] == grid[1] * grid[2]


def test_an_image_above_the_threshold_is_planned_as_the_checkpoint_plans_it():
    assert vision._geometry(943, 906, ON) == vision._geometry(943, 906, OFF)
    assert vision._geometry(2000, 3000, ON) == vision._geometry(2000, 3000, OFF)


def test_the_token_ceiling_still_applies():
    huge = vision.ImageProcessorConfig(upscale_min_pixels=10**10)
    assert vision.image_tokens(512, 492, huge) <= huge.max_image_tokens


def test_video_keeps_the_checkpoint_floor():
    assert vision._geometry(108, 192, ON, 8) == vision._geometry(108, 192, OFF, 8)


def test_digest_is_unchanged_off_and_differs_on():
    out = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(out, "PNG")
    data = out.getvalue()
    record = {"url": "data:image/png;base64," + base64.b64encode(data).decode()}
    assert vision.load_inputs([record], OFF)[0].digest == hashlib.sha256(data).hexdigest()
    assert vision.load_inputs([record], ON)[0].digest != hashlib.sha256(data).hexdigest()
    assert vision.load_inputs([record], vision.ImageProcessorConfig(upscale_min_pixels=1000000))[0].digest != vision.load_inputs([record], ON)[0].digest
