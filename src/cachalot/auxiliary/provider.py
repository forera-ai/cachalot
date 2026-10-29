"""
The provider interface: one small model behind a capability-described, blocking API.

Cachalot's runtime is thread-based (the server runs `Engine.stream_chat` in a
worker thread and cancels it through a `threading.Event`), so providers expose
blocking calls that take the same kind of cancel event. A provider that is
asynchronous inside, like the Apple Foundation Models SDK, owns its event loop.

A provider never touches the target model, its tokenizer, its KV cache or its
memory budget. What it can do is described by `Capabilities`, and every field
there is something the provider verified or something its API states; nothing
is inferred from a model's marketing name.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from cachalot.auxiliary.errors import AuxUnavailable


@dataclass(frozen=True)
class Capabilities:
    provider: str
    model: str
    available: bool
    unavailable_reason: str | None = None
    local_only: bool = True               # never sends a prompt off the machine
    context_tokens: int | None = None     # the provider's own token count, not the target's
    text_generation: bool = False
    streaming: bool = False
    structured_output: bool = False       # constrained decoding against a JSON schema
    token_counting: bool = False
    cancellation: bool = False
    greedy_sampling: bool = False
    tool_calling: bool = False            # supported by the backend; not used by any task yet
    # What speculative decoding would need. The Apple SDK exposes none of these.
    logits: bool = False
    tokenizer_access: bool = False
    kv_access: bool = False
    max_concurrent_requests: int = 1
    versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int = 256
    greedy: bool = True                   # deterministic by default: tasks are not creative writing
    temperature: float | None = None      # only read when greedy is False
    timeout_s: float | None = None        # None: the provider's configured default


@dataclass
class GenerationResult:
    text: str
    latency_s: float                      # submit to final byte, queue wait included
    queue_s: float = 0.0
    first_chunk_s: float | None = None    # streaming only
    prompt_tokens: int | None = None      # provider tokens, when the provider can count them
    output_tokens: int | None = None
    structured: dict[str, Any] | None = None


@runtime_checkable
class AuxiliaryProvider(Protocol):
    name: str

    def capabilities(self) -> Capabilities: ...

    def is_available(self) -> bool: ...

    def generate(self, prompt: str, *, instructions: str | None = None,
                 params: GenerationParams | None = None,
                 cancel: threading.Event | None = None) -> GenerationResult: ...

    def generate_stream(self, prompt: str, *, instructions: str | None = None,
                        params: GenerationParams | None = None,
                        cancel: threading.Event | None = None) -> Iterator[str]: ...

    def generate_structured(self, prompt: str, json_schema: dict[str, Any], *,
                            instructions: str | None = None,
                            params: GenerationParams | None = None,
                            cancel: threading.Event | None = None) -> GenerationResult: ...

    def count_tokens(self, text: str, *, instructions: str | None = None) -> int: ...

    def close(self) -> None: ...


class UnavailableProvider:
    """Stands in when auxiliary inference is disabled or its backend cannot load."""

    def __init__(self, name: str = "none", reason: str = "disabled", message: str = "auxiliary inference is disabled"):
        self.name = name
        self._reason = reason
        self._message = message

    def capabilities(self) -> Capabilities:
        return Capabilities(provider=self.name, model="none", available=False, unavailable_reason=self._reason)

    def is_available(self) -> bool:
        return False

    def _fail(self, *_, **__):
        raise AuxUnavailable(self._message, reason=self._reason)

    generate = generate_stream = generate_structured = count_tokens = _fail

    def close(self) -> None:
        pass
