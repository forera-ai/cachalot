"""
A stand-in for the `apple_fm_sdk` module: the classes and call shapes the provider uses,
with scripted behaviour, so the provider's error handling is tested without Apple's model.
"""

from __future__ import annotations

import asyncio
import json
from enum import IntEnum
from types import SimpleNamespace


class FoundationModelsError(Exception):
    pass


class GenerationError(FoundationModelsError):
    pass


class ExceededContextWindowSizeError(GenerationError):
    pass


class AssetsUnavailableError(GenerationError):
    pass


class GuardrailViolationError(GenerationError):
    pass


class RefusalError(GenerationError):
    pass


class RateLimitedError(GenerationError):
    pass


class ConcurrentRequestsError(GenerationError):
    pass


class DecodingFailureError(GenerationError):
    pass


class InvalidGenerationSchemaError(FoundationModelsError):
    pass


class SystemLanguageModelUnavailableReason(IntEnum):
    APPLE_INTELLIGENCE_NOT_ENABLED = 0
    DEVICE_NOT_ELIGIBLE = 1
    MODEL_NOT_READY = 2
    UNKNOWN = 0xFF


class SystemLanguageModelUseCase(IntEnum):
    GENERAL = 0
    CONTENT_TAGGING = 1


class SamplingMode:
    @classmethod
    def greedy(cls):
        return "greedy"


class GenerationOptions:
    def __init__(self, sampling=None, temperature=None, maximum_response_tokens=None):
        self.sampling, self.temperature, self.maximum_response_tokens = sampling, temperature, maximum_response_tokens


class _Content:
    def __init__(self, value):
        self._value = value

    def to_json(self):
        return self._value if isinstance(self._value, str) else json.dumps(self._value)


def make_sdk(*, available=True, reason=None, context_size=8192, reply="ok", structured=None,
             delay_s=0.0, raise_exc=None, snapshots=None, tokens_per_char=0.25):
    """Build a fake module. `structured` may be a dict, a raw string, or a callable(prompt)."""
    calls = SimpleNamespace(respond=0, stream=0, count=0, cancelled=0, sessions=0)

    class SystemLanguageModel:
        def __init__(self, use_case=SystemLanguageModelUseCase.GENERAL, guardrails=None):
            self.use_case = use_case
            self.context_size = context_size

        def is_available(self):
            return (True, None) if available else (False, reason)

        async def token_count(self, value=None, *, instructions=None):
            calls.count += 1
            text = value if value is not None else instructions
            return max(1, int(len(text) * tokens_per_char))

    class LanguageModelSession:
        def __init__(self, instructions=None, model=None, tools=None):
            calls.sessions += 1
            self.instructions = instructions

        async def respond(self, prompt, generating=None, *, schema=None, json_schema=None, options=None):
            calls.respond += 1
            try:
                if delay_s:
                    await asyncio.sleep(delay_s)
            except asyncio.CancelledError:
                calls.cancelled += 1
                raise
            if raise_exc is not None:
                raise raise_exc
            if json_schema is not None:
                value = structured(prompt) if callable(structured) else structured
                return _Content(value)
            return reply

        async def stream_response(self, prompt, options=None):
            calls.stream += 1
            for snap in snapshots or ["o", "ok"]:
                try:
                    await asyncio.sleep(delay_s or 0)
                except asyncio.CancelledError:
                    calls.cancelled += 1
                    raise
                yield snap
            if raise_exc is not None:
                raise raise_exc

    return SimpleNamespace(
        SystemLanguageModel=SystemLanguageModel, LanguageModelSession=LanguageModelSession,
        SystemLanguageModelUseCase=SystemLanguageModelUseCase,
        SystemLanguageModelUnavailableReason=SystemLanguageModelUnavailableReason,
        SamplingMode=SamplingMode, GenerationOptions=GenerationOptions,
        FoundationModelsError=FoundationModelsError, GenerationError=GenerationError,
        ExceededContextWindowSizeError=ExceededContextWindowSizeError,
        AssetsUnavailableError=AssetsUnavailableError, GuardrailViolationError=GuardrailViolationError,
        RefusalError=RefusalError, RateLimitedError=RateLimitedError,
        ConcurrentRequestsError=ConcurrentRequestsError, DecodingFailureError=DecodingFailureError,
        InvalidGenerationSchemaError=InvalidGenerationSchemaError, calls=calls,
    )
