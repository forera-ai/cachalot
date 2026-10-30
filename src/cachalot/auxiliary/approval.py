"""
Approve-only offload of an agent's command-safety review (docs/APPLE-FM-APPROVAL-RECORD.md).

Hermes Agent's "smart approval" asks the serving model whether a flagged shell command may run, with a fixed system
prompt and a one-word answer: APPROVE, DENY or ESCALATE. This module recognizes that request by its structure and
asks the auxiliary model first. Only an exact `APPROVE` is ever answered by the auxiliary model; every other outcome
(DENY, ESCALATE, other text, refusal, timeout, error) is a fall-through, and the target model reviews the command as
it would have without this module. The auxiliary model can therefore make the target do less work, never make a
command run that the target would have stopped -- except by approving one the target would have denied, which is
what the dangerous-command gate (`benchmarks/apple_fm_approval_gate.py`) measures before the switch is enabled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from cachalot.auxiliary.errors import AuxError
from cachalot.auxiliary.provider import AuxiliaryProvider, GenerationParams

# Both lines must be present in the system message; together they identify the reviewer prompt, and nothing else
# Hermes sends carries them.
REVIEWER_OPENING = "You are a security reviewer for an AI coding agent. You assess whether shell commands are safe to execute."
REVIEWER_ANSWER = "Respond with exactly one word: APPROVE, DENY, or ESCALATE"
VERDICT_TOKENS = 8
VERDICT_TIMEOUT_S = 5.0


@dataclass
class ReviewDecision:
    answered: bool          # True only for an exact APPROVE from the auxiliary model
    verdict: str            # APPROVE, DENY, ESCALATE, OTHER or the error kind
    raw: str
    latency_s: float


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):   # OpenAI content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def is_command_review(messages: list[dict[str, Any]], *, tools: Any = None, n: int = 1) -> bool:
    """True for the reviewer request shape: no tools, n = 1, exactly a system then a user message, reviewer text."""
    if tools or n != 1 or len(messages) != 2:
        return False
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        return False
    system = _text(messages[0].get("content"))
    return system.startswith(REVIEWER_OPENING) and REVIEWER_ANSWER in system and REVIEWER_ANSWER in _text(
        messages[1].get("content"))


def review(provider: AuxiliaryProvider, messages: list[dict[str, Any]], *,
           timeout_s: float = VERDICT_TIMEOUT_S) -> ReviewDecision:
    """Ask the auxiliary model; `answered` only when its whole reply is the single word APPROVE."""
    t0 = time.perf_counter()
    try:
        out = provider.generate(_text(messages[1].get("content")), instructions=_text(messages[0].get("content")),
                                params=GenerationParams(max_tokens=VERDICT_TOKENS, greedy=True, timeout_s=timeout_s))
    except AuxError as exc:
        return ReviewDecision(False, exc.kind, "", time.perf_counter() - t0)
    raw = out.text
    word = raw.strip().strip(".!").upper()
    verdict = word if word in ("APPROVE", "DENY", "ESCALATE") else "OTHER"
    return ReviewDecision(verdict == "APPROVE", verdict, raw, time.perf_counter() - t0)
