"""
Parent side of the auxiliary worker: the same provider API, one child process behind it.

At most one child exists per IsolatedProvider. It starts on the first call, not at
import, and is restarted when it dies, at most `max_restarts` times within
`restart_window_s`; past that the provider reports itself unavailable
(`crash_loop`) rather than respawning forever. The parent enforces its own deadline
(the request timeout plus a grace period): a child that does not answer by then is
killed, so a wedged backend can never hold a caller. `close()` closes the child's
stdin, waits briefly and kills it.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

from cachalot.auxiliary import errors as E
from cachalot.auxiliary.provider import Capabilities, GenerationParams, GenerationResult

READY_TIMEOUT_S = 30.0
DEADLINE_GRACE_S = 3.0


def _raise_payload(err: dict[str, Any]) -> None:
    kind = err.get("kind", "provider_error")
    message = err.get("message", "auxiliary worker error")
    if kind in ("unavailable", "disabled"):
        cls = E.AuxDisabled if kind == "disabled" else E.AuxUnavailable
        raise cls(message, reason=err.get("reason", "unknown"))
    if kind == "context_overflow":
        raise E.AuxContextOverflow(message, needed=err.get("needed"), limit=err.get("limit"))
    if kind == "invalid_output":
        raise E.AuxInvalidOutput(message, problems=err.get("problems"), raw=err.get("raw"))
    cls = {
        "timeout": E.AuxTimeout, "cancelled": E.AuxCancelled, "busy": E.AuxBusy, "refused": E.AuxRefused,
    }.get(kind, E.AuxProviderError)
    raise cls(message)


class IsolatedProvider:
    def __init__(self, name: str, *, timeout_s: float, env: dict[str, str] | None = None,
                 max_restarts: int = 3, restart_window_s: float = 60.0,
                 command: list[str] | None = None):
        self.name = name
        self.timeout_s = float(timeout_s)
        self._env = env
        self._command = command or [sys.executable, "-m", "cachalot.auxiliary.worker"]
        self.max_restarts = max_restarts
        self.restart_window_s = restart_window_s
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._inbox: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._next_id = 1
        self._starts: list[float] = []
        self._closed = False
        self.worker_pid: int | None = None
        self.restarts = 0

    # ---------------------------------------------------------- child process
    def _spawn(self) -> None:
        now = time.monotonic()
        self._starts = [t for t in self._starts if now - t < self.restart_window_s]
        if len(self._starts) > self.max_restarts:
            raise E.AuxUnavailable(
                f"auxiliary worker died {len(self._starts)} times in {self.restart_window_s:.0f} s",
                reason="crash_loop",
            )
        if self._starts:
            self.restarts += 1
        self._starts.append(now)
        env = dict(os.environ if self._env is None else self._env)
        src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.Popen(
            self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, text=True, bufsize=1,
        )
        ready: queue.Queue = queue.Queue()
        self._inbox = {0: ready}
        self._proc = proc
        self._reader = threading.Thread(target=self._read, args=(proc,), name="cachalot-aux-reader", daemon=True)
        self._reader.start()
        try:
            msg = ready.get(timeout=READY_TIMEOUT_S)
        except queue.Empty:
            self._kill(proc)
            raise E.AuxUnavailable("auxiliary worker did not start", reason="worker_start_failed") from None
        if msg.get("_dead"):
            raise E.AuxUnavailable("auxiliary worker exited at start", reason="worker_start_failed")
        self.worker_pid = int(msg.get("pid", proc.pid))

    def _read(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            box = self._inbox.get(int(msg.get("id", -1)))
            if box is not None:
                box.put(msg)
        # EOF: the child is gone; wake every waiter
        for box in list(self._inbox.values()):
            box.put({"_dead": True})

    def _kill(self, proc: subprocess.Popen | None) -> None:
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _send(self, obj: dict[str, Any]) -> None:
        with self._write_lock:
            try:
                self._proc.stdin.write(json.dumps(obj) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError, AttributeError) as exc:
                raise E.AuxProviderError(f"auxiliary worker pipe closed: {exc}") from exc

    def _request(self, op: str, args: dict[str, Any], timeout_s: float | None) -> tuple[int, queue.Queue]:
        with self._lock:
            if self._closed:
                raise E.AuxUnavailable("provider is closed", reason="closed")
            if not self._alive():
                self._spawn()
            req_id = self._next_id
            self._next_id += 1
            box: queue.Queue = queue.Queue()
            self._inbox[req_id] = box
        self._send({"id": req_id, "op": op, "args": args})
        return req_id, box

    def _wait(self, req_id: int, box: queue.Queue, deadline_s: float,
              cancel: threading.Event | None, on_chunk=None) -> dict[str, Any]:
        t_end = time.monotonic() + deadline_s
        cancelled_at = None
        try:
            while True:
                if cancel is not None and cancel.is_set() and cancelled_at is None:
                    cancelled_at = time.monotonic()
                    self._send({"id": req_id, "op": "cancel"})
                    t_end = min(t_end, cancelled_at + DEADLINE_GRACE_S)
                remaining = t_end - time.monotonic()
                if remaining <= 0:
                    proc = self._proc
                    self._kill(proc)   # a child that misses its deadline is not trusted with the next request
                    if cancelled_at is not None:
                        raise E.AuxCancelled("cancelled (worker killed after grace period)")
                    raise E.AuxTimeout(f"auxiliary worker missed its {deadline_s:.1f} s deadline; killed")
                try:
                    msg = box.get(timeout=min(remaining, 0.05))
                except queue.Empty:
                    continue
                if msg.get("_dead"):
                    raise E.AuxProviderError("auxiliary worker terminated unexpectedly")
                if "chunk" in msg:
                    if on_chunk is not None:
                        on_chunk(msg["chunk"])
                    continue
                if msg.get("ok"):
                    return msg.get("result") or {}
                _raise_payload(msg.get("error") or {})
        finally:
            self._inbox.pop(req_id, None)

    def _deadline(self, params: GenerationParams | None) -> float:
        timeout = (params.timeout_s if params and params.timeout_s else None) or self.timeout_s
        return timeout + DEADLINE_GRACE_S

    @staticmethod
    def _params(params: GenerationParams | None) -> dict[str, Any]:
        p = params or GenerationParams()
        return {"max_tokens": p.max_tokens, "greedy": p.greedy, "temperature": p.temperature,
                "timeout_s": p.timeout_s}

    # ------------------------------------------------------------ provider API
    def capabilities(self) -> Capabilities:
        try:
            req_id, box = self._request("capabilities", {}, None)
            return Capabilities(**self._wait(req_id, box, READY_TIMEOUT_S, None))
        except E.AuxError as exc:
            return Capabilities(provider=self.name, model="system", available=False,
                                unavailable_reason=getattr(exc, "reason", exc.kind))

    def is_available(self) -> bool:
        return self.capabilities().available

    def generate(self, prompt: str, *, instructions: str | None = None,
                 params: GenerationParams | None = None,
                 cancel: threading.Event | None = None) -> GenerationResult:
        req_id, box = self._request("generate", {"prompt": prompt, "instructions": instructions,
                                                 "params": self._params(params)}, None)
        return GenerationResult(**self._wait(req_id, box, self._deadline(params), cancel))

    def generate_structured(self, prompt: str, json_schema: dict[str, Any], *,
                            instructions: str | None = None,
                            params: GenerationParams | None = None,
                            cancel: threading.Event | None = None) -> GenerationResult:
        req_id, box = self._request("structured", {"prompt": prompt, "schema": json_schema,
                                                   "instructions": instructions,
                                                   "params": self._params(params)}, None)
        return GenerationResult(**self._wait(req_id, box, self._deadline(params), cancel))

    def generate_stream(self, prompt: str, *, instructions: str | None = None,
                        params: GenerationParams | None = None,
                        cancel: threading.Event | None = None) -> Iterator[str]:
        req_id, box = self._request("stream", {"prompt": prompt, "instructions": instructions,
                                               "params": self._params(params)}, None)
        chunks: queue.Queue = queue.Queue()
        done = object()
        stop = threading.Event()

        class _Either:
            def is_set(self_inner) -> bool:
                return stop.is_set() or (cancel is not None and cancel.is_set())

        def run():
            try:
                self._wait(req_id, box, self._deadline(params), _Either(), on_chunk=chunks.put)
            except BaseException as exc:
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
            worker.join(timeout=DEADLINE_GRACE_S + 1)

    def count_tokens(self, text: str, *, instructions: str | None = None) -> int:
        req_id, box = self._request("count", {"text": text, "instructions": instructions}, None)
        return int(self._wait(req_id, box, self.timeout_s + DEADLINE_GRACE_S, None)["tokens"])

    def close(self) -> None:
        with self._lock:
            self._closed = True
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:
            self._kill(proc)
