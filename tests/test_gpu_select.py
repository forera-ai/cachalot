"""GPU-side expert selection for MiniMax decode (HANDOFF 18.12): slab slots, the slot table, the slab kernels."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from cachalot.cache.resident import ResidentExpert
from cachalot.cache.resident_store import ResidentExpertStore
from cachalot.cache.slots import SlabSlotPool
from cachalot.minimax import codes_qmv
from cachalot.minimax.gpu_select import SlabExperts

D, INTER = 512, 512


def _sizes():
    sizes = {}
    for p, (out, inp) in {"w1": (INTER, D), "w3": (INTER, D), "w2": (D, INTER)}.items():
        sizes[f"{p}.weight"] = out * inp * 3 // 8
        sizes[f"{p}.scales"] = out * inp // 64 * 2
        sizes[f"{p}.codes"] = out * inp // 64 // 2
    return dict(sorted(sizes.items()))


def _fmt():
    return SimpleNamespace(shapes={"w1.weight": (INTER, D * 3 // 32), "w2.weight": (D, INTER * 3 // 32)})


def _fill(pool, index, rng):
    """Random weights, small positive bf16 scales, random nibble codes k + 7 with k in -7..-3."""
    slot = pool._slots[index]
    for name, view in slot.views.items():
        if name.endswith(".weight"):
            view[:] = rng.integers(0, 256, view.size, dtype=np.uint8)
        elif name.endswith(".scales"):
            s = rng.uniform(1e-3, 2e-2, view.size // 2).astype(np.float32)
            view[:] = (s.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
        else:
            c = rng.integers(0, 5, view.size * 2).astype(np.uint8)
            view[:] = c[0::2] | (c[1::2] << 4)


def _typed(slot, p, out, inp):
    a = slot.arrays
    return (a[f"{p}.weight"].view(mx.uint32).reshape(out, inp * 3 // 32),
            a[f"{p}.scales"].view(mx.bfloat16).reshape(out, inp // 64),
            a[f"{p}.codes"])


def test_slab_views_alias_the_slab_and_park_by_slab():
    pool = SlabSlotPool(_sizes(), 10, slab_slots=4)
    assert len(pool.slabs) == 3 and pool.capacity == 10
    rng = np.random.default_rng(0)
    _fill(pool, 5, rng)
    slot = pool._slots[5]
    assert np.array_equal(np.array(slot.arrays["w2.weight"]), slot.views["w2.weight"])
    row = np.array(pool.slabs[1])[1]
    assert np.array_equal(row[pool.offsets["w2.weight"]:][:slot.views["w2.weight"].size], slot.views["w2.weight"])
    held = [pool.try_acquire() for _ in range(10)]
    with pytest.raises(RuntimeError):
        pool.park_slab(2)
    pool.release_many(held)
    assert pool.park_slab(2) == 2 and pool.free_count == 8 and pool.slabs[2] is None
    assert pool.kernel_slabs()[2].shape == (1, 4096)
    assert pool.unpark_slab(2) == 2 and pool.free_count == 10


def test_slab_kernels_match_codes_qmv_bit_for_bit():
    pool = SlabSlotPool(_sizes(), 6, slab_slots=4)
    rng = np.random.default_rng(1)
    for i in range(6):
        _fill(pool, i, rng)
    act = lambda up, gate: up * mx.sigmoid(gate)  # noqa: E731
    x = mx.array(rng.standard_normal((1, D)).astype(np.float32)).astype(mx.bfloat16)
    slots = [5, 0, 3, 4]
    got = SlabExperts(pool, _fmt())(x, mx.array(np.array(slots, np.int32)), act)
    for j, i in enumerate(slots):
        s = pool._slots[i]
        g, u = codes_qmv.gate_up(x, _typed(s, "w1", INTER, D), _typed(s, "w3", INTER, D))
        want = codes_qmv.qmv(act(u, g), *_typed(s, "w2", D, INTER))
        assert np.array_equal(np.array(got[j:j + 1].view(mx.uint16)), np.array(want.view(mx.uint16))), j
    # a parked slab (its placeholder in the kernel's inputs) changes nothing for the others
    pool.park_slab(1)
    again = SlabExperts(pool, _fmt())(x, mx.array(np.array([0, 3], np.int32)), act)
    assert np.array_equal(np.array(again.view(mx.uint16)), np.array(got[1:3].view(mx.uint16)))


def _store(n_slots=12, slab=4, capacity_budget=8):
    pool = SlabSlotPool(_sizes(), n_slots, slab_slots=slab)
    store = ResidentExpertStore(capacity_budget * pool.slot_bytes, slot_pool=pool)
    table = store.track_slots(2, 16)
    return store, pool, table


def _admit(store, key):
    slot = store.pool.try_acquire()
    store._items[key] = ResidentExpert(key[0], key[1], slot)
    return slot


def test_the_slot_table_follows_every_change():
    store, pool, table = _store()
    v0 = store.slot_table_version
    slot = _admit(store, (1, 3))
    assert table[1, 3] == slot.index and store.slot_table_version > v0
    store._items.move_to_end((1, 3))
    assert table[1, 3] == slot.index
    with store._lock:
        store._evict_lru_locked()
    assert table[1, 3] == -1 and (table == -1).all()


def test_shrinking_parks_whole_slabs_and_keeps_the_survivors_data():
    store, pool, table = _store()
    assert store.capacity == 8
    rng = np.random.default_rng(2)
    keys = [(0, e) for e in range(8)]
    for k in keys:
        _admit(store, k)
    for i in range(12):
        if pool._slots[i].views:
            _fill(pool, i, rng)
    before = {k: pool._slots[table[k]].views["w1.weight"].copy() for k in keys}
    # three slots over a quarter slab: one slab (4 slots) is parked; the 4 least recently used go
    assert store.set_capacity(5) == 4
    assert pool.slabs[2] is None or pool.slabs[1] is None
    alive = [k for k in keys if table[k] >= 0]
    assert alive == keys[4:]
    for k in alive:
        assert pool.slabs[pool.slab_of(int(table[k]))] is not None
        assert np.array_equal(pool._slots[table[k]].views["w1.weight"], before[k])
    # room for a slab but not a quarter more does not unpark; a slab and a quarter does
    assert store.set_capacity(8) == 4
    store._full_capacity = 9
    assert store.set_capacity(9) == 8
