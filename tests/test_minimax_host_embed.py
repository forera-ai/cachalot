"""HANDOFF 18.18: MiniMax's quantized embedding table in host memory gives the same rows as on the GPU."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from cachalot.minimax.language import HostEmbedding


def test_host_embedding_is_bit_identical_to_quantized_embedding():
    mx.random.seed(0)
    emb = nn.Embedding(1000, 256)
    emb.weight = (mx.random.normal((1000, 256)) * 0.02).astype(mx.bfloat16)
    q = nn.QuantizedEmbedding.from_embedding(emb, group_size=64, bits=3)
    host = HostEmbedding(q)
    assert not dict(nn.utils.tree_flatten(host.parameters()))  # nothing of it is on the GPU
    for ids in ([[7]], [[0, 999, 7, 7, 500]]):
        x = mx.array(ids, dtype=mx.int32)
        want, got = q(x), host(x)
        assert got.dtype == want.dtype and got.shape == want.shape
        assert np.array_equal(np.array(got.view(mx.uint16)), np.array(want.view(mx.uint16)))


def test_host_embedding_uses_a_file_map_only_when_it_holds_the_same_table(tmp_path):
    mx.random.seed(1)
    emb = nn.Embedding(300, 128)
    emb.weight = (mx.random.normal((300, 128)) * 0.02).astype(mx.bfloat16)
    q = nn.QuantizedEmbedding.from_embedding(emb, group_size=64, bits=3)
    maps = {}
    for name in ("weight", "scales", "biases"):
        a = np.array(q[name]) if name == "weight" else np.array(q[name].view(mx.uint16))
        path = tmp_path / name
        a.tofile(path)
        maps[name] = np.memmap(path, dtype=a.dtype, mode="r", shape=a.shape)
    host = HostEmbedding(q, maps)
    assert host.mapped
    x = mx.array([[3, 299, 0]], dtype=mx.int32)
    assert np.array_equal(np.array(host(x).view(mx.uint16)), np.array(q(x).view(mx.uint16)))
    wrong = dict(maps, scales=np.zeros_like(np.array(maps["scales"])))
    assert not HostEmbedding(q, wrong).mapped
