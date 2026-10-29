"""
Error taxonomy of the auxiliary inference layer.

Every provider maps its own failures onto these classes, so a caller handles
one vocabulary whatever model sits behind it. Each class carries a stable
`kind` string, which is what metrics and benchmark rows record.
"""

from __future__ import annotations


class AuxError(Exception):
    """Base class. Never raised for a failure of the primary (target) model."""

    kind = "error"


class AuxUnavailable(AuxError):
    """The provider cannot run here: SDK missing, OS too old, Apple Intelligence off, assets not ready."""

    kind = "unavailable"

    def __init__(self, message: str, reason: str = "unknown"):
        super().__init__(message)
        self.reason = reason


class AuxDisabled(AuxUnavailable):
    """Auxiliary inference, or the requested feature, is switched off by configuration."""

    kind = "disabled"


class AuxTimeout(AuxError):
    kind = "timeout"


class AuxCancelled(AuxError):
    kind = "cancelled"


class AuxBusy(AuxError):
    """Admission refused: the bounded request queue is full."""

    kind = "busy"


class AuxContextOverflow(AuxError):
    """The request does not fit the provider's context window."""

    kind = "context_overflow"

    def __init__(self, message: str, needed: int | None = None, limit: int | None = None):
        super().__init__(message)
        self.needed = needed
        self.limit = limit


class AuxRefused(AuxError):
    """The provider's guardrails or the model itself declined the request."""

    kind = "refused"


class AuxInvalidOutput(AuxError):
    """Output failed deterministic validation (bad JSON, schema mismatch, ungrounded content)."""

    kind = "invalid_output"

    def __init__(self, message: str, problems: list[str] | None = None, raw: str | None = None):
        super().__init__(message)
        self.problems = list(problems or [])
        self.raw = raw


class AuxProviderError(AuxError):
    """Any other provider failure, including a crashed or terminated backend."""

    kind = "provider_error"
