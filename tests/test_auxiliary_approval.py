"""The approve-only review decision: request recognition and the exact-APPROVE rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cachalot.auxiliary import errors as E
from cachalot.auxiliary.approval import is_command_review, review
from cachalot.auxiliary.provider import GenerationResult

FIX = Path(__file__).resolve().parents[1] / "benchmarks" / "apple_fm"


def reviewer(command: str = "ls", category: str = "recursive delete") -> list[dict]:
    p = json.loads((FIX / "hermes_reviewer_prompt.json").read_text())
    return [{"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user_template"].format(category=category, command=command)}]


class Replies:
    name = "replies"

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, prompt, *, instructions=None, params=None, cancel=None):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return GenerationResult(text=self.reply, latency_s=0.01)


def test_recognizes_only_the_reviewer_shape():
    assert is_command_review(reviewer())
    msgs = reviewer()
    assert not is_command_review(msgs, tools=[{"type": "function"}])
    assert not is_command_review(msgs, n=2)
    assert not is_command_review(msgs + [{"role": "assistant", "content": "APPROVE"}])
    assert not is_command_review([{"role": "user", "content": msgs[1]["content"]}])
    other = [{"role": "system", "content": "You are a helpful assistant."}, msgs[1]]
    assert not is_command_review(other)


@pytest.mark.parametrize("reply,answered,verdict", [
    ("APPROVE", True, "APPROVE"), (" approve.\n", True, "APPROVE"),
    ("DENY", False, "DENY"), ("ESCALATE", False, "ESCALATE"),
    ("APPROVE, but be careful", False, "OTHER"), ("", False, "OTHER"),
])
def test_only_an_exact_approve_is_answered(reply, answered, verdict):
    d = review(Replies(reply), reviewer())
    assert d.answered is answered and d.verdict == verdict


@pytest.mark.parametrize("exc", [E.AuxTimeout("slow"), E.AuxRefused("no"), E.AuxUnavailable("off")])
def test_errors_fall_through(exc):
    d = review(Replies(exc), reviewer())
    assert not d.answered and d.verdict == exc.kind


def test_gate_fixture_is_well_formed():
    items = [json.loads(line) for line in (FIX / "approval_gate.jsonl").read_text().splitlines() if line.strip()]
    assert len(items) == 36 and len({i["id"] for i in items}) == 36
    for it in items:
        assert is_command_review(reviewer(it["command"], it["category"]))
