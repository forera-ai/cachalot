import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import quality_blind_ab as q  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": {"required": ["command"]}}}]


def call(name, args):
    return {"function": {"name": name, "arguments": args}}


def test_tool_call_problems():
    assert q.tool_call_problems([call("terminal", '{"command": "ls"}')], TOOLS) == []
    assert q.tool_call_problems(None, TOOLS) == ["no tool call"]
    assert "unknown tool 'rm'" in q.tool_call_problems([call("rm", "{}")], TOOLS)[0]
    assert "not JSON" in q.tool_call_problems([call("terminal", "{oops")], TOOLS)[0]
    assert "required 'command'" in q.tool_call_problems([call("terminal", "{}")], TOOLS)[0]
    assert q.tool_call_problems([call("terminal", {"command": "ls"})], TOOLS) == []


def test_repetition_share():
    prose = " ".join(f"word{i}" for i in range(60))
    assert q.repetition_share(prose) == 0.0
    assert q.repetition_share("loop and loop and loop " * 30) > 0.8
    assert q.repetition_share("short") == 0.0


def test_fisher_matches_known_values():
    assert q.fisher_two_sided(0, 10, 0, 10) == pytest.approx(1.0)
    assert q.fisher_two_sided(3, 10, 3, 10) == pytest.approx(1.0)
    assert q.fisher_two_sided(0, 10, 10, 10) < 1e-4
    # 1 of 8 against 6 of 8: two-sided p = 0.0406 (hypergeometric)
    assert q.fisher_two_sided(1, 8, 6, 8) == pytest.approx(0.0406, abs=1e-3)
