"""HANDOFF 18.22: after a request, a system block the disk store holds leaves memory; the conversation stays."""

from types import SimpleNamespace

from cachalot.glm.model import GlmModel


def _server(spill, disk=True):
    block = SimpleNamespace(tokens=tuple(range(100)), group=None)
    group = object()
    prompt = SimpleNamespace(tokens=tuple(range(150)), group=group)
    reply = SimpleNamespace(tokens=tuple(range(180)), group=group)
    side = SimpleNamespace(tokens=("aux",) * 20, group=None)
    s = SimpleNamespace(prefix=[block, prompt, reply, side], SPILL_PERSISTED=spill,
                        disk=SimpleNamespace(tokens={"a.safetensors": tuple(range(100))}) if disk else None)
    return s, block


def test_a_persisted_block_leaves_memory_and_the_rest_stays():
    s, block = _server(True)
    GlmModel._spill_persisted(s)
    assert block not in s.prefix
    assert [len(p.tokens) for p in s.prefix] == [150, 180, 20]


def test_off_or_without_a_disk_store_nothing_changes():
    for spill, disk in ((False, True), (True, False)):
        s, block = _server(spill, disk)
        GlmModel._spill_persisted(s)
        assert len(s.prefix) == 4


def test_minimax_defaults_spill_blocks_and_cap_the_decode_buffer_cache():
    from cachalot.minimax.model import MiniMaxModel

    assert MiniMaxModel.SPILL_PERSISTED is True
    assert MiniMaxModel.DECODE_CACHE_BYTES == 1024**3 // 4
    assert GlmModel.SPILL_PERSISTED is False and GlmModel.DECODE_CACHE_BYTES is None
