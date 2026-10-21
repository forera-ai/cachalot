"""DI-1 knob in prepare_prompt (Mac only; docs/DI1-BUILD-RECORD.md): the call order and the snapshot policy against a fake runtime. The model is never loaded."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cachalot.model import generation  # noqa: E402
from cachalot.model.generation import DecodeResult, prepare_prompt  # noqa: E402


class FakeCache:
    def __init__(self):
        self.added = []

    def find(self, tokens, span_keys=()):
        return None

    def add(self, snap, boundary=False, pin=False):
        self.added.append((snap["position"], boundary, pin))


class FakeRuntime:
    def __init__(self, with_finish=True):
        self.prefix_cache = FakeCache()
        self.calls, self.position, self.pending = [], 0, False
        if with_finish:
            self.finish_decoder = self._finish

    def reset(self):
        self.position, self.pending = 0, False

    def prefill_tokens(self, ids, next_token_ids=None, encoder_only=False, **_):
        if not encoder_only and self.pending:
            raise RuntimeError("full prefill while a decoder half is pending")
        self.calls.append(("enc" if encoder_only else "full", len(ids)))
        self.position += len(ids)
        self.pending = self.pending or encoder_only
        return DecodeResult(logits=None if encoder_only else "logits", hidden=None, routes=(), position=self.position - 1)

    def _finish(self):
        assert self.pending
        self.calls.append(("finish", self.position))
        self.pending = False
        return DecodeResult(logits="logits", hidden=None, routes=(), position=self.position - 1)

    def snapshot(self, logits=None):
        assert not self.pending, "a snapshot with a decoder half pending"
        return {"position": self.position, "logits": logits}


def run(monkeypatch, knob, n, boundaries=(), with_finish=True, chunk=4096):
    monkeypatch.setenv("CACHALOT_PREFILL_ENCODER_ONLY", "1" if knob else "0")
    monkeypatch.setattr(generation, "PREFILL_CHUNK", chunk)
    rt = FakeRuntime(with_finish)
    result, reused = prepare_prompt(rt, list(range(n)), boundaries=boundaries)
    return rt, result


def test_knob_off_is_the_old_path(monkeypatch):
    rt, result = run(monkeypatch, False, 10000)
    assert [c[0] for c in rt.calls] == ["full"] * 3 and result.logits == "logits"
    assert [a[0] for a in rt.prefix_cache.added] == [4096, 8192, 10000]


def test_knob_on_runs_the_encoder_everywhere_and_the_decoder_once(monkeypatch):
    rt, result = run(monkeypatch, True, 10000)
    assert rt.calls == [("enc", 4096), ("enc", 4096), ("enc", 1808), ("finish", 10000)] and result.logits == "logits"
    assert [a[0] for a in rt.prefix_cache.added] == [10000]                  # unpinned intermediate snapshots are not taken


def test_a_boundary_snapshot_finishes_the_decoder_there_and_again_at_the_end(monkeypatch):
    rt, result = run(monkeypatch, True, 10000, boundaries=(4096,))
    assert rt.calls == [("enc", 4096), ("finish", 4096), ("enc", 4096), ("enc", 1808), ("finish", 10000)]
    assert [(a[0], a[1]) for a in rt.prefix_cache.added] == [(4096, True), (10000, False)]


def test_a_single_call_prompt_and_a_runtime_without_finish_keep_the_old_path(monkeypatch):
    rt, _ = run(monkeypatch, True, 3000)
    assert rt.calls == [("full", 3000)]
    rt, _ = run(monkeypatch, True, 10000, with_finish=False)
    assert [c[0] for c in rt.calls] == ["full"] * 3
