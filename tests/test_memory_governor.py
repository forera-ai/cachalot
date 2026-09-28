"""The memory governor (HANDOFF 18.15, S2): a long prefill chunk gives back expert slots, macOS's memory state caps
growth and forces a slab back under pressure."""
from types import SimpleNamespace

import pytest

from cachalot.glm import model as glm_model
from cachalot.glm.model import GlmModel

GIB = 1024**3


class FakeStore:
    def __init__(self, capacity, expert_bytes, slab_slots=128):
        self.capacity = capacity
        self._full_capacity = capacity
        self.expert_bytes = expert_bytes
        self.pool = SimpleNamespace(slab_slots=slab_slots)
        self.calls = []

    def set_capacity(self, target):
        self.calls.append(target)
        self.capacity = max(1, min(target, self._full_capacity))
        return self.capacity


def governed(capacity=2718, slot=21 * 2**20, prefill_gib=52):
    m = object.__new__(GlmModel)
    m.store = FakeStore(capacity, slot)
    m._prefill_budget = prefill_gib * GIB
    m.PREFILL_CHUNK = 8192
    return m


@pytest.fixture
def host(monkeypatch):
    state = {"level": 1, "available": 20 * GIB, "gpu": -1, "ours": 0}
    monkeypatch.setattr(glm_model, "host_memory", lambda: (1 * GIB, state["level"]))
    monkeypatch.setattr(glm_model, "host_available", lambda: state["available"])
    monkeypatch.setattr(glm_model, "gpu_allocated", lambda: state["gpu"])
    monkeypatch.setattr(glm_model, "gpu_ceiling", lambda: 78 * GIB)
    monkeypatch.setattr(glm_model.mx, "get_active_memory", lambda: state["ours"])
    monkeypatch.setattr(glm_model.mx, "get_cache_memory", lambda: 0)
    monkeypatch.setattr(glm_model.mx, "clear_cache", lambda: None)
    return state


def test_short_prefill_keeps_the_capacity(host):
    m = governed()
    m._fit_prefill(2048)
    assert m.store.calls == []


def test_full_chunk_prefills_at_the_prefill_budget(host):
    m = governed()
    m._fit_prefill(20000)
    assert m.store.capacity == 52 * GIB // (21 * 2**20)


def test_mid_chunk_scales_linearly(host):
    m = governed()
    full, low = 2718, 52 * GIB // (21 * 2**20)
    m._fit_prefill(5120)  # halfway between 2,048 and 8,192
    assert m.store.capacity == int(full - 0.5 * (full - low))


def test_warning_pressure_gives_back_a_slab_even_for_a_short_prefill(host):
    m = governed()
    host["level"] = 2
    m._fit_prefill(30)
    assert m.store.capacity == 2718 - 128


def test_little_available_caps_growth(host):
    m = governed()
    m.store.capacity = 2500
    host["available"] = glm_model.HOST_AVAILABLE_FLOOR + 10 * m.store.expert_bytes
    assert m._host_capacity(2500) == 2510
    host["available"] = glm_model.HOST_AVAILABLE_FLOOR // 2 - 1
    assert m._host_capacity(2500) == 2500 - 128


def test_ungoverned_model_is_untouched(host):
    m = governed()
    m._prefill_budget = None
    host["level"] = 4
    m._fit_prefill(8192)
    assert m.store.calls == []


def test_host_readers_answer_on_macos():
    free, level = glm_model.host_memory()
    if free < 0:
        pytest.skip("not macOS")
    assert free > 0 and level in (1, 2, 4)
    assert glm_model.host_available() > 0


def test_other_processes_gpu_memory_gives_back_slots(host):
    # HANDOFF 18.17: 1 GiB allocated above the GPU ceiling, whoever holds it, costs ceil(1 GiB / slot) slots,
    # rounded up to a whole slab since 18.18 item 8
    m, slot = governed(), 21 * 2**20
    host["ours"], host["gpu"] = 72 * GIB, 79 * GIB
    assert -(-GIB // slot) < 128
    assert m._host_capacity(2718) == 2718 - 128
    host["gpu"] = 78 * GIB + 3 * GIB  # a larger overshoot: its own count, more than a slab
    assert m._host_capacity(2718) == 2718 - -(-3 * GIB // slot)


def test_gpu_room_caps_growth(host):
    m, slot = governed(), 21 * 2**20
    host["ours"], host["gpu"] = 70 * GIB, 77 * GIB  # 1 GiB under the ceiling, 12 GiB available
    assert m._host_capacity(2718) == 2718 + GIB // slot


def test_unknown_gpu_memory_leaves_the_host_rule(host):
    m, slot = governed(), 21 * 2**20
    host["gpu"] = -1
    assert m._host_capacity(2718) == 2718 + 12 * GIB // slot


def test_a_small_gpu_overshoot_still_gives_back_a_whole_slab(host):
    """HANDOFF 18.18 item 8: 0.5 GiB over the ceiling is ~24 slots, under a slab pool's quarter-slab tolerance;
    the GPU term asks for a whole slab so the pool actually parks one."""
    m = governed(capacity=2920)
    host["gpu"] = 78 * GIB + GIB // 2
    assert m._host_capacity(2920) <= 2920 - 128
