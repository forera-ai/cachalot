"""GLM's in-block pins: which chunk boundaries are kept, that a pin past the store's old cap reaches the disk and survives a restart, and that MiniMax stays off."""
from __future__ import annotations

import mlx.core as mx

from cachalot.glm.model import GlmModel, Snapshot
from cachalot.glm.snapshots import GlmSnapshotStore
from cachalot.minimax.model import MiniMaxModel
from cachalot.third_party.mlx_vlm.models.cache import KVCache


def _due(boundary):
    return [p for p in range(0, boundary + 1) if GlmModel._block_pin_due(p, boundary)]


def test_the_last_three_boundaries_inside_a_block():
    assert GlmModel.PREFILL_CHUNK == 2048 and GlmModel.BLOCK_PINS == 3
    assert _due(15564) == [10240, 12288, 14336]
    assert _due(4096) == [2048]  # a boundary at the block's end is the block itself
    assert _due(4097) == [2048, 4096]
    assert _due(2048) == []


def test_a_position_off_the_chunk_grid_is_never_pinned():
    assert not GlmModel._block_pin_due(14337, 15564)


def test_minimax_keeps_no_pins():
    assert MiniMaxModel.BLOCK_PINS == 0
    assert _due(15564) and not any(MiniMaxModel._block_pin_due(p, 40000) for p in range(0, 40001, 8192))


def _snap(n):
    c = KVCache()
    c.update_and_fetch(mx.zeros((1, 2, 4, 2)), mx.zeros((1, 2, 4, 2)))
    mx.eval(c.keys, c.values)
    return Snapshot(tuple(range(n)), [c], None)


def test_a_long_pin_survives_a_restart_and_serves_a_changed_block(tmp_path):
    store = GlmSnapshotStore(tmp_path, "id", keep=4)
    store.PIN_MAX_TOKENS = GlmModel.PIN_MAX_TOKENS
    store.persist_pin(_snap(14336))
    again = GlmSnapshotStore(tmp_path, "id", keep=4)
    again.PIN_MAX_TOKENS = GlmModel.PIN_MAX_TOKENS
    assert again.load_all() == []  # pins are indexed, not preloaded
    changed = tuple(range(15000)) + (7,)
    assert len(again.fetch(changed, 0).tokens) == 14336
