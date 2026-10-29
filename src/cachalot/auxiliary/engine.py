"""
AuxiliaryEngine: configuration, one provider, feature flags and counters behind the typed tasks.

The engine is not coupled to Apple's model: `build_provider` picks a backend from the
configuration, and a deterministic configuration runs every task on its heuristic. Each
task method checks its own feature flag and raises AuxDisabled when it is off, so a
caller that never enables a feature can never have its requests touched by it.
Counters are fixed-size (one record per task), so they cannot grow with traffic.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from cachalot.auxiliary.config import AuxiliaryConfig, load_aux_config
from cachalot.auxiliary.errors import AuxDisabled
from cachalot.auxiliary.provider import AuxiliaryProvider, UnavailableProvider
from cachalot.auxiliary.tasks import (
    Classification,
    ClassifyRequest,
    CompressedContext,
    Extraction,
    ExtractRequest,
    classify,
    compress,
    extract,
)

AVAILABILITY_TTL_S = 30.0   # re-ask the provider at most this often (assets can finish downloading)


def build_provider(config: AuxiliaryConfig, isolation: str | None = None) -> AuxiliaryProvider:
    """The configured provider; never raises (an unloadable backend reports itself unavailable)."""
    if config.provider == "deterministic":
        return UnavailableProvider(name="deterministic", reason="deterministic",
                                   message="the deterministic provider runs heuristics only")
    mode = isolation or config.isolation
    if mode == "process":
        from cachalot.auxiliary.isolated import IsolatedProvider

        env = None  # the child inherits CACHALOT_AUX_*; it builds its provider inline
        return IsolatedProvider(config.provider, timeout_s=config.timeout_s, env=env)
    from cachalot.auxiliary.apple_fm import AppleFoundationModelsProvider

    return AppleFoundationModelsProvider(
        timeout_s=config.timeout_s,
        max_concurrent_requests=config.max_concurrent_requests,
        queue_limit=config.queue_limit,
        use_case=config.use_case,
    )


@dataclass
class _TaskStats:
    calls: int = 0
    model: int = 0
    heuristic: int = 0
    fallbacks: dict[str, int] = field(default_factory=dict)
    latency_s_total: float = 0.0
    latency_s_max: float = 0.0

    def record(self, source: str, fallback: str | None, latency: float) -> None:
        self.calls += 1
        if source == "model":
            self.model += 1
        else:
            self.heuristic += 1
        if fallback:
            self.fallbacks[fallback] = self.fallbacks.get(fallback, 0) + 1
        self.latency_s_total += latency
        self.latency_s_max = max(self.latency_s_max, latency)


class AuxiliaryEngine:
    def __init__(self, config: AuxiliaryConfig | None = None, provider: AuxiliaryProvider | None = None):
        self.config = config or load_aux_config()
        self.provider = provider if provider is not None else build_provider(self.config)
        self._stats = {name: _TaskStats() for name in ("classification", "structured_extraction", "context_compression")}
        self._lock = threading.Lock()
        self._available = False
        self._availability_until = 0.0

    def _active_provider(self) -> AuxiliaryProvider | None:
        """The provider when it can run, else None (the tasks then use their heuristics)."""
        if isinstance(self.provider, UnavailableProvider):
            return None
        now = time.monotonic()
        if now >= self._availability_until:
            self._available = self.provider.is_available()
            self._availability_until = now + AVAILABILITY_TTL_S
        return self.provider if self._available else None

    def _require(self, feature: str) -> None:
        if not self.config.feature(feature):
            raise AuxDisabled(f"auxiliary feature {feature!r} is off (CACHALOT_AUX_ENABLED=1 "
                              f"CACHALOT_AUX_{feature.upper()}=1 turns it on)", reason="disabled")

    def _record(self, feature: str, source: str, fallback: str | None, latency: float) -> None:
        with self._lock:
            self._stats[feature].record(source, fallback, latency)

    def classify(self, req: ClassifyRequest, cancel: threading.Event | None = None) -> Classification:
        self._require("classification")
        result = classify(req, self._active_provider(), timeout_s=self.config.timeout_s, cancel=cancel)
        self._record("classification", result.source, result.fallback_reason, result.latency_s)
        return result

    def extract(self, req: ExtractRequest, cancel: threading.Event | None = None) -> Extraction:
        self._require("structured_extraction")
        result = extract(req, self._active_provider(), timeout_s=self.config.timeout_s, cancel=cancel)
        self._record("structured_extraction", result.source, result.fallback_reason, result.latency_s)
        return result

    def compress(self, text: str, *, target_tokens=None, query: str | None = None,
                 cancel: threading.Event | None = None) -> CompressedContext:
        self._require("context_compression")
        cfg = self.config
        result = compress(text, self._active_provider(), target_tokens=target_tokens,
                          min_tokens=cfg.compression_min_tokens, max_ratio=cfg.compression_max_ratio,
                          max_ungrounded=cfg.compression_max_ungrounded, query=query,
                          timeout_s=cfg.timeout_s, cancel=cancel)
        self._record("context_compression", result.source, result.reject_reason, result.latency_s)
        return result

    def stats(self) -> dict[str, Any]:
        with self._lock:
            tasks = {k: {**vars(v), "fallbacks": dict(v.fallbacks)} for k, v in self._stats.items()}
        return {"enabled": self.config.enabled, "provider": self.provider.name, "tasks": tasks}

    def close(self) -> None:
        self.provider.close()
