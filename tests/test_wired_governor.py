from types import SimpleNamespace

import pytest

from cachalot.cache import wired_governor as wg
from cachalot.cache.wired_governor import GIB, WiredGovernor

SLOT = 10 * 1024 * 1024  # 10 MiB


def make(ceiling_gib=74.0, quiet=60.0, clock=None):
    t = clock or [0.0]
    gov = WiredGovernor(int(ceiling_gib * GIB), SLOT, hysteresis=int(1.5 * GIB), grow_quiet=quiet, clock=lambda: t[0])
    return gov, t


def test_shrinks_by_the_excess_rounded_up():
    gov, _ = make()
    excess = 3 * GIB + 1  # a byte over three GiB needs one slot more than 307.2 -> 308... whole slots, rounded up
    want = gov.target(5000, 5000, int(74 * GIB) + excess)
    assert want == 5000 - -(-excess // SLOT)
    assert gov.shrinks == 1


def test_holds_inside_the_band():
    gov, _ = make()
    assert gov.target(4000, 5000, int(73.0 * GIB)) == 4000  # under the ceiling, inside the hysteresis
    assert gov.grows == 0 and gov.shrinks == 0


def test_grows_only_after_the_quiet_period_and_into_the_room():
    gov, t = make(quiet=60.0)
    gov.target(5000, 5000, int(75 * GIB))  # shrink at t=0
    t[0] = 30.0
    assert gov.target(4800, 5000, int(60 * GIB)) == 4800  # too soon after a shrink
    t[0] = 61.0
    room = int(74 * GIB) - int(1.5 * GIB) - int(60 * GIB)
    assert gov.target(4800, 5000, int(60 * GIB)) == min(5000, 4800 + room // SLOT)
    assert gov.grows == 1


def test_growth_is_capped_at_the_built_capacity():
    gov, t = make()
    t[0] = 1e6
    assert gov.target(4990, 5000, int(40 * GIB)) == 5000


def test_a_tiny_room_does_not_grow():
    gov, t = make()
    t[0] = 1e6
    wired = int(74 * GIB) - int(1.5 * GIB) - 5 * SLOT  # five slots of room: under MIN_GROW_SLOTS
    assert gov.target(4000, 5000, wired) == 4000


def test_never_below_one_slot():
    gov, _ = make()
    assert gov.target(3, 5000, int(500 * GIB)) == 1


def test_off_when_disabled_or_unknown():
    assert WiredGovernor(-1, SLOT).target(4000, 5000, int(90 * GIB)) == 4000
    gov, _ = make()
    assert gov.target(4000, 5000, -1) == 4000


def test_ceiling_from_the_environment(monkeypatch):
    monkeypatch.setenv("CACHALOT_WIRED_CEILING_GIB", "70")
    assert wg.default_ceiling_bytes() == 70 * GIB
    monkeypatch.setenv("CACHALOT_WIRED_CEILING_GIB", "0")
    assert wg.default_ceiling_bytes() == -1
    monkeypatch.delenv("CACHALOT_WIRED_CEILING_GIB")
    monkeypatch.setattr(wg, "_sysctl", lambda name: 96 * GIB if name == "hw.memsize" else None)
    assert wg.default_ceiling_bytes() == int(96 * GIB * 0.76)


@pytest.mark.parametrize("pages,page,expect", [(1000, 16384, 1000 * 16384), (None, 16384, -1), (1000, None, -1)])
def test_system_wired_bytes(monkeypatch, pages, page, expect):
    values = {"vm.page_wired_count": pages, "hw.pagesize": page}
    monkeypatch.setattr(wg, "_sysctl", lambda name: values[name])
    assert wg.system_wired_bytes() == expect


def test_runtime_fit_applies_the_governor_between_tokens(monkeypatch):
    from cachalot.model import text_decode_runtime as rt

    calls = []

    class Store:
        capacity = 5000
        _full_capacity = 5000

        def set_capacity(self, want):
            calls.append(want)
            self.capacity = want
            return want

    gov, _ = make()
    fake = SimpleNamespace(wired_governor=gov, _wired_tokens=0, expert_store=Store())
    monkeypatch.setattr(rt, "system_wired_bytes", lambda: int(76 * GIB))

    rt.TextDecodeRuntime._wired_fit(fake)  # first token: always checked
    assert calls == [5000 - -(-(2 * GIB) // SLOT)]
    for _ in range(rt.WIRED_CHECK_EVERY_TOKENS - 1):  # tokens 2..16 are not checked
        rt.TextDecodeRuntime._wired_fit(fake)
    assert len(calls) == 1
    rt.TextDecodeRuntime._wired_fit(fake)  # token 17 is
    assert len(calls) == 2
