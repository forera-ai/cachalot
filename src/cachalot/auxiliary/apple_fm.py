"""
Apple Foundation Models provider: the on-device system model through Apple's Python SDK.

Backend choice (docs/APPLE-FM-AUXILIARY.md section 2): `apple-fm-sdk` (Apple, Apache-2.0,
ctypes over Swift C bindings of FoundationModels.framework). It covers availability,
instructions, greedy sampling, streaming, JSON-schema guided generation, token counting,
the context size and cancellation, and it has no Private Cloud Compute path at all: the
only model it constructs is `SystemLanguageModel`, the on-device model. The `fm` CLI is
kept as a diagnostic, not a backend (a subprocess per call, text parsing, and a `pcc`
model one flag away).

Inference itself runs in a system service (TGOnDeviceInferenceProviderService), not in
this process; the SDK is an IPC client. Measured on macOS 27.2 (2026-10-29): two
sessions in parallel take as long as the two run back to back, so the system model
serializes requests and `max_concurrent_requests` above 1 buys queueing, not throughput.

Facts this class relies on, each checked against the installed SDK 0.2.1:
  * `SystemLanguageModel.is_available()` -> (bool, SystemLanguageModelUnavailableReason | None)
  * `SystemLanguageModel.context_size` (8192 on this machine) and async `token_count(...)`
  * `LanguageModelSession(instructions=, model=)`, async `respond(prompt, json_schema=, options=)`,
    async generator `stream_response(prompt, options=)` yielding cumulative snapshots
  * cancelling the awaiting asyncio task calls FMTaskCancel; the session is reusable after
  * a prompt over the context window fails as a generic GenerationError (status 255), not
    ExceededContextWindowSizeError, so requests are checked against `context_size` first.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import platform
import queue
import sys
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from typing import Any

from cachalot.auxiliary.errors import (
    AuxBusy,
    AuxCancelled,
    AuxContextOverflow,
    AuxError,
    AuxInvalidOutput,
    AuxProviderError,
    AuxRefused,
    AuxTimeout,
    AuxUnavailable,
)
from cachalot.auxiliary.provider import Capabilities, GenerationParams, GenerationResult

MIN_MACOS_MAJOR = 26
CANCEL_POLL_S = 0.02
CANCEL_GRACE_S = 2.0
# Tokens kept free beside the counted prompt, instructions, schema and response cap:
# the session template and schema framing are not in the separate counts.
CONTEXT_MARGIN_TOKENS = 64


class _LoopThread:
    """A private asyncio loop on a daemon thread; the SDK's futures live and die on it."""

    def __init__(self, name: str):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro: Coroutine) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        if self.loop.is_closed():
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            self.loop.close()


def _macos_major() -> int | None:
    release = platform.mac_ver()[0]
    try:
        return int(release.split(".")[0])
    except (ValueError, IndexError):
        return None


class AppleFoundationModelsProvider:
    name = "apple_foundation_models"

    def __init__(
        self,
        *,
        timeout_s: float = 10.0,
        max_concurrent_requests: int = 1,
        queue_limit: int = 4,
        use_case: str = "general",
        precheck_context: bool = True,
        sdk: Any = None,
    ):
        self.timeout_s = float(timeout_s)
        self.max_concurrent = int(max_concurrent_requests)
        self.queue_limit = int(queue_limit)
        self.use_case = use_case
        self.precheck_context = precheck_context
        self._sdk = sdk                     # injected by tests; imported lazily otherwise
        self._model = None
        self._context_size: int | None = None
        self._loop: _LoopThread | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._state = threading.Lock()
        self._pending = 0
        self._closed = False
        self._unavailable: tuple[str, str] | None = None   # (reason, message)

    # ------------------------------------------------------------------ setup
    def _import_sdk(self):
        if self._sdk is not None:
            return self._sdk
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise AuxUnavailable("Apple Foundation Models needs an Apple Silicon Mac", reason="unsupported_hardware")
        major = _macos_major()
        if major is None or major < MIN_MACOS_MAJOR:
            raise AuxUnavailable(f"Apple Foundation Models needs macOS {MIN_MACOS_MAJOR} or later", reason="unsupported_os")
        try:
            import apple_fm_sdk
        except ImportError as exc:
            raise AuxUnavailable(
                "apple-fm-sdk is not installed (pip install 'cachalot[apple-fm]')", reason="sdk_missing"
            ) from exc
        except OSError as exc:  # the bundled dylib failed to load
            raise AuxUnavailable(f"apple-fm-sdk failed to load: {exc}", reason="runtime_incompatible") from exc
        self._sdk = apple_fm_sdk
        return apple_fm_sdk

    def _ensure(self):
        """Load SDK, model and loop once; raise AuxUnavailable with a reason otherwise."""
        with self._state:
            if self._closed:
                raise AuxUnavailable("provider is closed", reason="closed")
            if self._model is not None:
                return self._model
            sdk = self._import_sdk()
            use_case = {
                "general": sdk.SystemLanguageModelUseCase.GENERAL,
                "content_tagging": sdk.SystemLanguageModelUseCase.CONTENT_TAGGING,
            }[self.use_case]
            try:
                model = sdk.SystemLanguageModel(use_case=use_case)
            except Exception as exc:
                raise AuxUnavailable(f"system model initialization failed: {exc}", reason="init_failed") from exc
            ok, why = model.is_available()
            if not ok:
                reason = why.name.lower() if why is not None else "unknown"
                self._unavailable = (reason, f"system model unavailable: {reason}")
                raise AuxUnavailable(self._unavailable[1], reason=reason)
            self._model = model
            self._context_size = int(model.context_size)
            self._loop = _LoopThread("cachalot-aux-apple-fm")
            self._semaphore = None   # built on the loop, on first use
            self._unavailable = None
            return model

    # ------------------------------------------------------------ capability
    def is_available(self) -> bool:
        try:
            self._ensure()
            return True
        except AuxUnavailable as exc:
            self._unavailable = (exc.reason, str(exc))
            return False

    def capabilities(self) -> Capabilities:
        versions = {"macos": platform.mac_ver()[0], "python": platform.python_version()}
        try:
            import importlib.metadata as metadata

            versions["apple_fm_sdk"] = metadata.version("apple-fm-sdk")
        except Exception:
            pass
        available = self.is_available()
        return Capabilities(
            provider=self.name,
            model="system",
            available=available,
            unavailable_reason=None if available else (self._unavailable or ("unknown", ""))[0],
            local_only=True,
            context_tokens=self._context_size,
            text_generation=available,
            streaming=available,
            structured_output=available,
            token_counting=available,
            cancellation=available,
            greedy_sampling=available,
            tool_calling=available,
            max_concurrent_requests=self.max_concurrent,
            versions=versions,
        )

    # ------------------------------------------------------------ execution
    def _admit(self) -> None:
        with self._state:
            if self._pending >= self.max_concurrent + self.queue_limit:
                raise AuxBusy(f"{self._pending} auxiliary requests in flight or queued (limit "
                              f"{self.max_concurrent + self.queue_limit})")
            self._pending += 1

    def _release(self) -> None:
        with self._state:
            self._pending -= 1

    def _options(self, params: GenerationParams):
        sdk = self._sdk
        if params.greedy:
            return sdk.GenerationOptions(sampling=sdk.SamplingMode.greedy(), maximum_response_tokens=params.max_tokens)
        return sdk.GenerationOptions(temperature=params.temperature, maximum_response_tokens=params.max_tokens)

    def _map_error(self, exc: BaseException) -> AuxError:
        if isinstance(exc, AuxError):
            return exc
        sdk = self._sdk

        def is_a(name: str) -> bool:
            cls = getattr(sdk, name, None)
            return isinstance(cls, type) and isinstance(exc, cls)

        message = f"{type(exc).__name__}: {exc}"
        if is_a("ExceededContextWindowSizeError"):
            return AuxContextOverflow(message, limit=self._context_size)
        if is_a("AssetsUnavailableError"):
            return AuxUnavailable(message, reason="assets_unavailable")
        if is_a("GuardrailViolationError") or is_a("RefusalError") or "refused to answer" in message:
            # SDK 0.2.1 surfaces a refusal as a generic GenerationError (status 255) whose text reads
            # "LanguageModelError:3 - The model refused to answer." (measured 2026-10-29)
            return AuxRefused(message)
        if is_a("RateLimitedError") or is_a("ConcurrentRequestsError"):
            return AuxBusy(message)
        if is_a("DecodingFailureError") or is_a("InvalidGenerationSchemaError"):
            return AuxInvalidOutput(message)
        return AuxProviderError(message)

    async def _guarded(self, work: Callable[[], Coroutine], timeout_s: float, t_submit: float):
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run():
            async with self._semaphore:
                queue_s = time.perf_counter() - t_submit
                return queue_s, await work()

        try:
            return await asyncio.wait_for(run(), timeout_s)
        except TimeoutError as exc:
            raise AuxTimeout(f"auxiliary request exceeded {timeout_s:.1f} s") from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self._map_error(exc) from exc

    def _call(self, work: Callable[[], Coroutine], timeout_s: float | None,
              cancel: threading.Event | None):
        self._ensure()
        self._admit()
        t_submit = time.perf_counter()
        try:
            if cancel is not None and cancel.is_set():
                raise AuxCancelled("cancelled before start")
            loop = self._loop
            if loop is None:   # closed between _ensure and here
                raise AuxUnavailable("provider is closed", reason="closed")
            fut = loop.submit(self._guarded(work, timeout_s or self.timeout_s, t_submit))
            while True:
                try:
                    queue_s, value = fut.result(timeout=CANCEL_POLL_S)
                    return time.perf_counter() - t_submit, queue_s, value
                except concurrent.futures.TimeoutError:
                    if cancel is not None and cancel.is_set():
                        fut.cancel()   # cancels the asyncio task; the SDK then calls FMTaskCancel
                        concurrent.futures.wait([fut], timeout=CANCEL_GRACE_S)
                        raise AuxCancelled("cancelled by caller") from None
                except concurrent.futures.CancelledError:
                    raise AuxCancelled("cancelled") from None
        finally:
            self._release()

    async def _count(self, text: str | None = None, instructions: str | None = None) -> int:
        if instructions is not None and text is None:
            return int(await self._model.token_count(instructions=instructions))
        return int(await self._model.token_count(text))

    async def _precheck(self, prompt: str, instructions: str | None, max_tokens: int,
                        schema: dict | None) -> int | None:
        if not self.precheck_context:
            return None
        n = await self._count(prompt)
        if instructions:
            n += await self._count(instructions=instructions)
        if schema is not None:
            n += await self._count(json.dumps(schema))   # an estimate of the schema's framing
        limit = self._context_size or 0
        if n + max_tokens + CONTEXT_MARGIN_TOKENS > limit:
            raise AuxContextOverflow(
                f"request needs ~{n} prompt + {max_tokens} response tokens, context is {limit}",
                needed=n + max_tokens, limit=limit,
            )
        return n

    def _session(self, instructions: str | None):
        return self._sdk.LanguageModelSession(instructions=instructions, model=self._model)

    def generate(self, prompt: str, *, instructions: str | None = None,
                 params: GenerationParams | None = None,
                 cancel: threading.Event | None = None) -> GenerationResult:
        params = params or GenerationParams()

        async def work():
            n = await self._precheck(prompt, instructions, params.max_tokens, None)
            text = await self._session(instructions).respond(prompt, options=self._options(params))
            return n, text

        latency, queue_s, (n, text) = self._call(work, params.timeout_s, cancel)
        return GenerationResult(text=str(text), latency_s=latency, queue_s=queue_s, prompt_tokens=n)

    def generate_structured(self, prompt: str, json_schema: dict[str, Any], *,
                            instructions: str | None = None,
                            params: GenerationParams | None = None,
                            cancel: threading.Event | None = None) -> GenerationResult:
        params = params or GenerationParams()

        async def work():
            n = await self._precheck(prompt, instructions, params.max_tokens, json_schema)
            content = await self._session(instructions).respond(
                prompt, json_schema=json_schema, options=self._options(params)
            )
            return n, content.to_json()

        latency, queue_s, (n, raw) = self._call(work, params.timeout_s, cancel)
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise AuxInvalidOutput(f"structured output is not JSON: {exc}", raw=str(raw)) from exc
        if not isinstance(value, dict):
            raise AuxInvalidOutput("structured output is not a JSON object", raw=str(raw))
        return GenerationResult(text=raw, latency_s=latency, queue_s=queue_s, prompt_tokens=n, structured=value)

    def generate_stream(self, prompt: str, *, instructions: str | None = None,
                        params: GenerationParams | None = None,
                        cancel: threading.Event | None = None) -> Iterator[str]:
        """Yield text deltas. Closing the iterator early cancels the generation."""
        params = params or GenerationParams()
        chunks: queue.Queue = queue.Queue()
        done = object()

        async def work():
            await self._precheck(prompt, instructions, params.max_tokens, None)
            sent = ""
            async for snapshot in self._session(instructions).stream_response(prompt, options=self._options(params)):
                snapshot = str(snapshot)
                if snapshot.startswith(sent):
                    delta = snapshot[len(sent):]
                else:   # the SDK yields cumulative snapshots; tolerate a rewrite by resending
                    delta = snapshot
                sent = snapshot
                if delta:
                    chunks.put(delta)
            return None

        stop = threading.Event()

        def run():
            try:
                self._call(work, params.timeout_s, _AnyEvent(cancel, stop))
            except BaseException as exc:   # handed to the consumer below
                chunks.put(exc)
            finally:
                chunks.put(done)

        worker = threading.Thread(target=run, name="cachalot-aux-stream", daemon=True)
        worker.start()
        try:
            while True:
                item = chunks.get()
                if item is done:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            worker.join(timeout=CANCEL_GRACE_S + 1)

    def count_tokens(self, text: str, *, instructions: str | None = None) -> int:
        async def work():
            n = await self._count(text) if text else 0
            if instructions:
                n += await self._count(instructions=instructions)
            return n

        _, _, n = self._call(work, None, None)
        return n

    def close(self) -> None:
        with self._state:
            self._closed = True
            loop, self._loop = self._loop, None
            self._model = None
        if loop is not None:
            loop.stop()


class _AnyEvent:
    """Quacks like threading.Event for `is_set`: set when either event is set."""

    def __init__(self, *events: threading.Event | None):
        self._events = [e for e in events if e is not None]

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._events)
