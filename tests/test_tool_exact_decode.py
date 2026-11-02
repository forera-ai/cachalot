"""CACHALOT_DECODE_TOOL_EXACT (default off): once a reply opens a tool-call block the decode miss budget is lifted for
the rest of the reply and restored afterwards, however the request ends (docs/v1017-tool-exact/)."""

import pytest

import cachalot.server.engine as engine_mod
from cachalot.model.generation import SamplingParams
from cachalot.server.engine import ChatRequest, Engine
from fake_model import FakeEncoding, FakeModel, ScriptedRuntime

OPENER = "\n\n<CALLS"  # an ASCII stand-in for the DSML opener the fake tokenizer cannot spell


class BudgetRecordingRuntime(ScriptedRuntime):
    """Records the miss budget each decode step runs under."""

    def __init__(self, reply, budget):
        super().__init__(reply=reply)
        self.expert_store.decode_miss_budget = budget
        self.budgets: list = []

    def decode_token(self, token_id):
        self.budgets.append(self.expert_store.decode_miss_budget)
        return super().decode_token(token_id)


@pytest.fixture(autouse=True)
def ascii_opener(monkeypatch):
    monkeypatch.setattr(engine_mod, "TOOL_CALLS_START", OPENER)


def _run(reply, budget, knob, monkeypatch):
    monkeypatch.setenv("CACHALOT_DECODE_TOOL_EXACT", "1" if knob else "0")
    rt = BudgetRecordingRuntime(reply, budget)
    engine = Engine(FakeModel(rt), model_id="t", encoding=FakeEncoding())
    req = ChatRequest(messages=[{"role": "user", "content": "go"}], params=SamplingParams(temperature=0.0, max_new_tokens=64))
    stream = engine.stream_chat(req)
    for _ in stream:
        pass
    return rt


def test_knob_lifts_the_budget_from_the_opener_to_the_end_and_restores_it(monkeypatch):
    rt = _run("ok" + OPENER + " x=1>", 0, True, monkeypatch)
    n_before = len("ok" + OPENER)  # decodes fed with the tokens up to and including the opener's last character
    assert rt.budgets[: n_before - 1] == [0] * (n_before - 1)
    assert rt.budgets[n_before - 1:] and all(b is None for b in rt.budgets[n_before - 1:])
    assert rt.expert_store.decode_miss_budget == 0


def test_knob_off_keeps_the_budget(monkeypatch):
    rt = _run("ok" + OPENER + " x=1>", 0, False, monkeypatch)
    assert set(rt.budgets) == {0}


def test_a_reply_without_a_tool_call_keeps_the_budget(monkeypatch):
    rt = _run("plain text reply", 0, True, monkeypatch)
    assert set(rt.budgets) == {0}


def test_the_exact_path_is_left_alone(monkeypatch):
    rt = _run("ok" + OPENER + " x=1>", None, True, monkeypatch)
    assert set(rt.budgets) == {None}


def test_the_budget_is_restored_when_the_client_leaves_mid_block(monkeypatch):
    # The content before the opener is released on the step that completes the opener, the step that lifts the
    # budget; closing the stream on that delta leaves the request inside the block.
    monkeypatch.setenv("CACHALOT_DECODE_TOOL_EXACT", "1")
    rt = BudgetRecordingRuntime("ok" + OPENER + " x=1 and more arguments>", 2)
    engine = Engine(FakeModel(rt), model_id="t", encoding=FakeEncoding())
    req = ChatRequest(messages=[{"role": "user", "content": "go"}], params=SamplingParams(temperature=0.0, max_new_tokens=64))
    stream = engine.stream_chat(req)
    for delta in stream:
        if delta.content and rt.expert_store.decode_miss_budget is None:  # the tail before the opener, budget lifted
            stream.close()
            break
    else:
        pytest.fail("the stream never paused inside the block")
    assert rt.expert_store.decode_miss_budget == 2


def test_the_knob_defaults_off(monkeypatch):
    monkeypatch.delenv("CACHALOT_DECODE_TOOL_EXACT", raising=False)
    engine = Engine(FakeModel(ScriptedRuntime()), model_id="t", encoding=FakeEncoding())
    assert engine.tool_exact_decode is False
