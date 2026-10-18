"""
CUDA research lane: manifests, result rows and the redaction contract (docs/CUDA-RESEARCH-PLAN.md section 7).

Standard library only. Nothing here imports MLX, torch or `cachalot`, so it runs on a Mac with no NVIDIA stack.

Two documents, schema `cuda-manifest` v1:

* the *machine manifest* (once per instance session) and
* the *run manifest* (once per benchmark run, naming the machine manifest by id).

Result rows follow plan section 7.4. A field that was not measured is absent, never zero; a quantity the platform
cannot report is a row tagged `unavailable` with a reason and no value.

Redaction is an allow-list for environment variables plus a scan that refuses any document containing an address,
token, UUID, MAC, e-mail or home-directory-shaped string (plan section 7.3). `assert_publishable` is the gate every
writer calls before a file is saved.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from typing import Any

SCHEMA_NAME = "cuda-manifest"
SCHEMA_VERSION = 1
PLATFORM = "cuda"
TAGS = ("measured", "derived", "predicted", "hypothesis", "unavailable")
RUN_MODES = ("timing", "profiling", "correctness")

# Environment variables that may enter a manifest. Everything else is dropped, never scrubbed.
ENV_ALLOW_PREFIXES = ("CUDA_", "NCCL_", "CACHALOT_")
ENV_ALLOW_EXACT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF")
# ... and these never enter, even under an allowed prefix (credentials a prefix could cover).
ENV_DENY_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASS", "CRED", "AUTH", "CODE", "PROMO")

MACHINE_REQUIRED = (
    "schema", "schema_version", "platform", "manifest_id", "kind", "created_utc", "code", "gpu", "software",
    "host", "interconnect", "profiler", "provenance",
)
RUN_REQUIRED = (
    "schema", "schema_version", "platform", "manifest_id", "kind", "created_utc", "machine_manifest_id",
    "run_id", "instrument", "code", "workload", "mode", "protocol", "limits", "gates", "status",
)
ROW_REQUIRED = ("run", "arm", "field", "tag", "unit", "source")

# Shapes that must never appear in a publishable file.
_FORBIDDEN = {
    "ipv4": re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"),
    "ipv6": re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{1,4}:){4,7}[0-9A-Fa-f]{1,4}(?![0-9A-Fa-f:])"),
    "mac": re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
    "uuid": re.compile(r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b"),
    "gpu_uuid": re.compile(r"\bGPU-[0-9A-Fa-f]{8}-"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "home_path": re.compile(r"(?:/Users/|/home/|C:\\Users\\)[^/\\\s\"']+"),
    "api_token": re.compile(r"\b(?:rift_|sk-|ghp_|hf_|AKIA)[A-Za-z0-9_-]{8,}"),
    "long_secret": re.compile(r"\b[A-Za-z0-9+/_-]{40,}={0,2}(?![A-Za-z0-9+/_-])"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----|ssh-(?:ed25519|rsa) AAAA"),
}
# A sha256 digest is 64 hex characters and legitimately appears as an id or hash; allow it explicitly.
_HEX_DIGEST = re.compile(r"^[0-9a-f]{16,64}$")


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_id(doc: dict) -> str:
    """Id = first 16 hex of sha256 over the canonical JSON without the id field."""
    body = {k: v for k, v in doc.items() if k != "manifest_id"}
    return hashlib.sha256(canonical(body).encode()).hexdigest()[:16]


def seal(doc: dict) -> dict:
    doc = dict(doc)
    doc["manifest_id"] = content_id(doc)
    return doc


def filter_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Allow-listed environment only. Values are kept as given (they are then scanned like everything else)."""
    env = os.environ if environ is None else environ
    kept = {}
    for name in sorted(env):
        up = name.upper()
        allowed = name in ENV_ALLOW_EXACT or up.startswith(ENV_ALLOW_PREFIXES)
        if allowed and not any(s in up for s in ENV_DENY_SUBSTRINGS):
            kept[name] = env[name]
    return kept


def _strings(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(k, f"{path}/<key>")
            yield from _strings(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def find_forbidden(obj: Any) -> list[tuple[str, str]]:
    """(json path, kind) for every string that has a forbidden shape. Never returns the offending text."""
    hits = []
    for path, text in _strings(obj):
        if _HEX_DIGEST.match(text):
            continue
        for kind, rx in _FORBIDDEN.items():
            if rx.search(text):
                hits.append((path, kind))
    return hits


class RedactionError(ValueError):
    pass


def assert_publishable(obj: Any) -> None:
    hits = find_forbidden(obj)
    if hits:
        shown = ", ".join(f"{p} ({k})" for p, k in hits[:8])
        raise RedactionError(f"refusing to write: forbidden content at {shown}")


def _check_required(doc: dict, required: tuple[str, ...], what: str) -> list[str]:
    problems = [f"{what}: missing '{k}'" for k in required if k not in doc]
    if doc.get("schema") != SCHEMA_NAME:
        problems.append(f"{what}: schema must be '{SCHEMA_NAME}'")
    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"{what}: schema_version must be {SCHEMA_VERSION}")
    if doc.get("platform") != PLATFORM:
        problems.append(f"{what}: platform must be '{PLATFORM}'")
    if "manifest_id" in doc and doc["manifest_id"] != content_id(doc):
        problems.append(f"{what}: manifest_id does not match content")
    return problems


def validate_machine(doc: dict) -> list[str]:
    problems = _check_required(doc, MACHINE_REQUIRED, "machine manifest")
    if doc.get("kind") != "machine":
        problems.append("machine manifest: kind must be 'machine'")
    return problems + [f"redaction: {p} ({k})" for p, k in find_forbidden(doc)]


def validate_run(doc: dict) -> list[str]:
    problems = _check_required(doc, RUN_REQUIRED, "run manifest")
    if doc.get("kind") != "run":
        problems.append("run manifest: kind must be 'run'")
    if doc.get("mode") not in RUN_MODES:
        problems.append(f"run manifest: mode must be one of {RUN_MODES}")
    if doc.get("status") not in ("complete", "truncated", "skipped", "failed", "dry-run"):
        problems.append("run manifest: unknown status")
    return problems + [f"redaction: {p} ({k})" for p, k in find_forbidden(doc)]


def validate_row(row: dict) -> list[str]:
    problems = [f"row: missing '{k}'" for k in ROW_REQUIRED if k not in row]
    tag = row.get("tag")
    if tag not in TAGS:
        problems.append(f"row: tag {tag!r} not in {TAGS}")
    if tag == "unavailable":
        if "value" in row:
            problems.append("row: an unavailable row must not carry a value")
        if not row.get("note"):
            problems.append("row: an unavailable row needs a reason in 'note'")
    elif "value" not in row:
        problems.append("row: missing 'value'")
    if "value" in row and not isinstance(row["value"], (int, float, bool, str)):
        problems.append("row: value must be a scalar")
    return problems + [f"redaction: {p} ({k})" for p, k in find_forbidden(row)]


def row(
    run_id: str, arm: str, field: str, value: float | int | str | None, unit: str, tag: str, *,
    source: str, n: int | None = None, p50: float | None = None, p95: float | None = None,
    ci_low: float | None = None, ci_high: float | None = None, ci_method: str | None = None,
    overhead_ms: float | None = None, note: str | None = None,
) -> dict:
    """One result row (plan 7.4). Absent beats zero: None fields are left out, `unavailable` carries no value."""
    out: dict[str, Any] = {"run": run_id, "arm": arm, "field": field, "unit": unit, "tag": tag, "source": source}
    if tag != "unavailable":
        out["value"] = value
    for key, val in (("n", n), ("p50", p50), ("p95", p95), ("ci_low", ci_low), ("ci_high", ci_high),
                     ("ci_method", ci_method), ("overhead_ms", overhead_ms), ("note", note)):
        if val is not None:
            out[key] = val
    return out


def write_json(path, doc: dict) -> None:
    assert_publishable(doc)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")


def append_rows(path, rows: list[dict]) -> None:
    for r in rows:
        bad = validate_row(r)
        if bad:
            raise ValueError("; ".join(bad))
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(canonical(r) + "\n")


def read_json(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
