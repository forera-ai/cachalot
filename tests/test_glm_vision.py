import numpy as np
import pytest
from PIL import Image

from cachalot.glm import vision


CFG = vision.ImageProcessorConfig()


def test_smart_resize_aligns_to_the_merge_grid():
    h, w = vision.smart_resize(2, 360, 640, temporal_factor=2, factor=28, min_pixels=16, max_pixels=8000)
    assert h % 28 == 0 and w % 28 == 0
    assert (h, w) == (364, 644)


def test_smart_resize_respects_the_token_budget():
    h, w = vision.smart_resize(2, 4000, 6000, temporal_factor=2, factor=28, min_pixels=16, max_pixels=8000)
    assert (h // 14) * (w // 14) // 4 <= 8000


def test_image_tokens_match_the_patch_grid():
    patches, grid = vision.preprocess(Image.new("RGB", (640, 360), "white"), CFG)
    assert grid[0] == 1
    assert patches.shape == (grid[1] * grid[2], 3 * CFG.temporal_patch_size * CFG.patch_size**2)
    assert vision.image_tokens(360, 640, CFG) == grid[1] * grid[2] // 4 == 26 * 46 // 4


def test_preprocess_normalises_and_repeats_the_temporal_axis():
    patches, _ = vision.preprocess(Image.new("RGB", (28, 28), (255, 255, 255)), CFG)
    per_frame = 3 * CFG.patch_size**2
    p = patches[0].reshape(3, CFG.temporal_patch_size, CFG.patch_size, CFG.patch_size)
    assert np.allclose(p[:, 0], p[:, 1])
    expected = (1.0 - np.asarray(CFG.image_mean)) / np.asarray(CFG.image_std)
    assert np.allclose(p[:, 0, 0, 0], expected, atol=1e-5)
    assert patches.shape[1] == 2 * per_frame


def test_image_records_reads_openai_parts_in_order():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "a"},
                                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "image_url", "image_url": "file:///tmp/x.png"}]},
    ]
    assert vision.image_records(messages) == [{"url": "data:image/png;base64,AAAA"}, {"url": "file:///tmp/x.png"}]


def test_expand_and_key_tokens():
    img = lambda n, d: vision.ImageInput(d, np.zeros((n * 4, 1176), np.float32), [1, 2, 2 * n], n)
    tokens = [1, 2, vision.IMAGE_TOKEN_ID, 3, vision.IMAGE_TOKEN_ID, 4]
    out, spans = vision.expand(tokens, [img(3, "a" * 64), img(2, "b" * 64)])
    assert out == [1, 2] + [vision.IMAGE_TOKEN_ID] * 3 + [3] + [vision.IMAGE_TOKEN_ID] * 2 + [4]
    assert [(s.start, s.length) for s in spans] == [(2, 3), (6, 2)]
    key = vision.key_tokens(out, spans)
    assert key[:2] == [1, 2] and key[5] == 3 and key[-1] == 4
    assert all(k < 0 for k in key[2:5] + key[6:8]) and len(set(key[2:5])) == 3
    other, other_spans = vision.expand(tokens, [img(3, "c" * 64), img(2, "b" * 64)])
    assert vision.key_tokens(other, other_spans)[2:5] != key[2:5]  # another image, another key
    assert vision.key_tokens(other, other_spans)[6:8] == key[6:8]  # the same image, the same key


def test_expand_rejects_a_marker_mismatch():
    with pytest.raises(ValueError):
        vision.expand([1, vision.IMAGE_TOKEN_ID, vision.IMAGE_TOKEN_ID], [vision.ImageInput("d", np.zeros((4, 4)), [1, 2, 2], 1)])


def test_a_subclass_that_skips_init_has_no_vision():
    # MiniMaxModel subclasses GlmModel without calling its __init__ (no _raw_config): the shared engine must see False
    from cachalot.glm.model import GlmModel
    from cachalot.minimax.model import MiniMaxModel

    assert MiniMaxModel.__new__(MiniMaxModel).has_vision() is False
    assert GlmModel.__new__(GlmModel).has_vision() is False
