"""
Child process that hosts an auxiliary provider behind JSON lines on stdin/stdout.

    python -m cachalot.auxiliary.worker      (started by IsolatedProvider; not by hand)

Why a child: the Apple SDK is ctypes over Swift bindings loaded into the calling
process. A fault in that layer would otherwise take the Cachalot server and its
loaded model down with it; in a child it costs one restart. The child exits when
its stdin closes, so a parent that dies cannot leave it behind.

Protocol, one JSON object a line:
    parent -> child  {"id": 7, "op": "generate"|"structured"|"stream"|"count"|"capabilities", "args": {...}}
                     {"id": 7, "op": "cancel"}
    child -> parent  {"id": 7, "chunk": "..."}                      (stream only, zero or more)
                     {"id": 7, "ok": true, "result": {...}}
                     {"id": 7, "ok": false, "error": {"kind": ..., "message": ..., ...}}
                     {"id": 0, "ready": true, "pid": ...}           (once, at start)
"""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import asdict
from typing import Any

from cachalot.auxiliary.errors import AuxError
from cachalot.auxiliary.provider import GenerationParams


def error_payload(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": getattr(exc, "kind", "provider_error"), "message": str(exc),
                               "type": type(exc).__name__}
    for attr in ("reason", "needed", "limit", "problems", "raw"):
        if hasattr(exc, attr):
            payload[attr] = getattr(exc, attr)
    return payload


def _make_provider():
    from cachalot.auxiliary.config import load_aux_config
    from cachalot.auxiliary.engine import build_provider

    return build_provider(load_aux_config(), isolation="inline")


def main() -> int:
    out_lock = threading.Lock()
    real_stdout = sys.stdout
    sys.stdout = sys.stderr   # nothing but protocol lines may reach the parent's pipe

    def send(obj: dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with out_lock:
            real_stdout.write(line + "\n")
            real_stdout.flush()

    provider = _make_provider()
    cancels: dict[int, threading.Event] = {}

    def handle(req_id: int, op: str, args: dict[str, Any]) -> None:
        cancel = cancels[req_id]
        try:
            params = GenerationParams(**args.pop("params", {}))
            if op == "capabilities":
                result: dict[str, Any] = provider.capabilities().to_dict()
            elif op == "count":
                result = {"tokens": provider.count_tokens(args["text"], instructions=args.get("instructions"))}
            elif op == "generate":
                result = asdict(provider.generate(args["prompt"], instructions=args.get("instructions"),
                                                  params=params, cancel=cancel))
            elif op == "structured":
                result = asdict(provider.generate_structured(args["prompt"], args["schema"],
                                                             instructions=args.get("instructions"),
                                                             params=params, cancel=cancel))
            elif op == "stream":
                for chunk in provider.generate_stream(args["prompt"], instructions=args.get("instructions"),
                                                      params=params, cancel=cancel):
                    send({"id": req_id, "chunk": chunk})
                result = {}
            else:
                raise AuxError(f"unknown op {op!r}")
            send({"id": req_id, "ok": True, "result": result})
        except BaseException as exc:
            send({"id": req_id, "ok": False, "error": error_payload(exc)})
        finally:
            cancels.pop(req_id, None)

    send({"id": 0, "ready": True, "pid": os.getpid()})
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        req_id, op = int(msg.get("id", 0)), msg.get("op")
        if op == "cancel":
            event = cancels.get(req_id)
            if event is not None:
                event.set()
            continue
        cancels[req_id] = threading.Event()
        threading.Thread(target=handle, args=(req_id, op, dict(msg.get("args") or {})),
                         name=f"aux-req-{req_id}", daemon=True).start()
    for event in list(cancels.values()):
        event.set()
    provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
