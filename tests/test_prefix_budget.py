"""HANDOFF 18.18 item 7: an agent conversation's snapshot must survive a short side request (Hermes's auxiliary
tasks) in the MiniMax server's prefix budget. MiniMax's KV cache is ~120 KiB a token."""

from types import SimpleNamespace

from cachalot.glm.model import GlmModel

KV = 120 * 1024


def _snap(n, group=None):
    return SimpleNamespace(tokens=tuple(range(n)) if group is None else tuple(range(n)) + (-1,), nbytes=n * KV,
                           group=group)


def _server(gib):
    s = SimpleNamespace(prefix=[], PREFIX_BYTES=int(gib * 1024**3))
    s._prefix_bytes = lambda: GlmModel._prefix_bytes(s)
    return s


def _after_session(gib):
    """The 2026-09-28 session: the system block from disk, a 45k-token turn (prompt and reply snapshots, one
    group), then a 1,871-token side request."""
    s = _server(gib)
    GlmModel._add_prefix(s, _snap(21318))
    group = object()
    GlmModel._add_prefix(s, _snap(45318, group))
    GlmModel._add_prefix(s, _snap(45906, group))
    GlmModel._add_prefix(s, SimpleNamespace(tokens=("aux",) * 1871, nbytes=1871 * KV, group=None))
    return [len(p.tokens) for p in s.prefix]


def test_a_side_request_evicts_a_45k_conversation_at_5_gib():
    assert max(_after_session(5)) == 1871


def test_the_serve_script_budget_keeps_it():
    budget = float(open("serve-minimax.sh").read().split("CACHALOT_GLM_PREFIX_GIB:-")[1].split("}")[0])
    assert 45907 in _after_session(budget)
