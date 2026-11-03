"""CACHALOT_VISION_MIN_PIXELS: off by default, the upscale it plans, and the digest it keys rows by (no tower, no model)."""
from __future__ import annotations

from cachalot.model import image_processor_mlx as ip
from cachalot.model.vision_prompt import VisionEncoder, image_digest


def test_off_by_default_and_at_or_under_the_official_minimum(monkeypatch):
    monkeypatch.delenv("CACHALOT_VISION_MIN_PIXELS", raising=False)
    assert ip.min_pixels_from_env() is None
    assert ip.image_config_from_env() == ip.ImageProcessorConfig()
    for off in ("", "0", str(ip.OFFICIAL_MIN_PIXELS)):
        monkeypatch.setenv("CACHALOT_VISION_MIN_PIXELS", off)
        assert ip.min_pixels_from_env() is None


def test_small_screenshot_grows_and_a_large_one_does_not(monkeypatch):
    monkeypatch.setenv("CACHALOT_VISION_MIN_PIXELS", "810000")
    cfg = ip.image_config_from_env()
    assert cfg.min_pixels == 810000
    small = ip.plan_image_grid(492, 512, cfg)
    assert ip.num_image_tokens(small[0], small[1]) == 486
    assert ip.num_image_tokens(*ip.plan_image_grid(492, 512, ip.ImageProcessorConfig())[:2]) == 198
    # an image already above the threshold is planned exactly as the checkpoint plans it
    assert ip.plan_image_grid(906, 943, cfg) == ip.plan_image_grid(906, 943, ip.ImageProcessorConfig())


def test_digest_is_unchanged_at_the_official_setting_and_differs_otherwise():
    raw = b"not really a png"
    import hashlib

    assert image_digest(raw, ip.ImageProcessorConfig()) == hashlib.sha256(raw).hexdigest()[:32]
    assert image_digest(raw, ip.ImageProcessorConfig(min_pixels=810000)) != image_digest(raw, ip.ImageProcessorConfig())


def test_encoder_takes_the_setting_from_the_environment(monkeypatch):
    monkeypatch.setenv("CACHALOT_VISION_MIN_PIXELS", "810000")
    assert VisionEncoder({}).image_cfg.min_pixels == 810000
    monkeypatch.delenv("CACHALOT_VISION_MIN_PIXELS")
    assert VisionEncoder({}).image_cfg == ip.ImageProcessorConfig()
