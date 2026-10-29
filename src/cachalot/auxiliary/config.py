"""
Auxiliary inference configuration, read from CACHALOT_AUX_* environment variables.

Same convention as `cachalot.config.RuntimeConfig`: a frozen dataclass whose
every field is overridden by an upper-case environment variable. Everything is
off by default. `CACHALOT_AUX_ENABLED=1` turns the layer on; each feature then
needs its own flag, so enabling the provider never changes what the target
model sees.

    CACHALOT_AUX_ENABLED=1 CACHALOT_AUX_CONTEXT_COMPRESSION=1 ...

There is deliberately no switch for Private Cloud Compute: the Apple provider
only constructs the on-device system model, and `local_only` cannot be turned
off from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"not a boolean: {raw!r}")


@dataclass(frozen=True)
class AuxiliaryConfig:
    enabled: bool = False
    provider: str = "apple_foundation_models"   # or "deterministic"
    model: str = "system"                       # the on-device system model; nothing else is accepted
    timeout_ms: int = 10_000
    max_concurrent_requests: int = 1            # the system model runs one request at a time anyway
    queue_limit: int = 4                        # waiting requests beyond this are refused (AuxBusy)
    use_case: str = "general"                   # general | content_tagging (the system model's adapters)
    # "process": the provider runs in one child process (cachalot.auxiliary.worker), so a fault
    # in the SDK's native bindings cannot take the caller down. "inline": in this process.
    isolation: str = "process"

    # Feature flags. Each is independent and none is validated for default use.
    classification: bool = False
    structured_extraction: bool = False
    context_compression: bool = False
    semantic_validation: bool = False           # research only: no implementation behind it yet
    adaptive_routing: bool = False

    # Compression policy (lever B).
    compression_min_tokens: int = 2_000         # below this the target prefills it faster than we compress it
    compression_max_ratio: float = 0.6          # reject a result that keeps more than this share of tokens
    compression_max_ungrounded: float = 0.1     # reject when more than this share of facts fail grounding

    @property
    def local_only(self) -> bool:
        return True

    @property
    def timeout_s(self) -> float:
        return self.timeout_ms / 1000.0

    def feature(self, name: str) -> bool:
        """True only when the layer is enabled and the named feature flag is set."""
        return self.enabled and bool(getattr(self, name))

    def with_env_overrides(self, environ: dict[str, str] | None = None) -> AuxiliaryConfig:
        env = os.environ if environ is None else environ
        updates: dict[str, object] = {}
        for f in fields(self):
            key = f"CACHALOT_AUX_{f.name.upper()}"
            if key not in env:
                continue
            raw = env[key]
            current = getattr(self, f.name)
            if isinstance(current, bool):
                updates[f.name] = _parse_bool(raw)
            elif isinstance(current, int):
                updates[f.name] = int(raw)
            elif isinstance(current, float):
                updates[f.name] = float(raw)
            else:
                updates[f.name] = raw
        cfg = replace(self, **updates) if updates else self
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.provider not in ("apple_foundation_models", "deterministic"):
            raise ValueError(f"unknown auxiliary provider {self.provider!r}")
        if self.model != "system":
            raise ValueError(f"auxiliary model must be 'system' (the on-device model), got {self.model!r}")
        if self.use_case not in ("general", "content_tagging"):
            raise ValueError(f"unknown use case {self.use_case!r}")
        if self.isolation not in ("process", "inline"):
            raise ValueError(f"isolation must be 'process' or 'inline', got {self.isolation!r}")
        if self.timeout_ms <= 0 or self.max_concurrent_requests < 1 or self.queue_limit < 0:
            raise ValueError("timeout_ms must be > 0, max_concurrent_requests >= 1, queue_limit >= 0")
        if not 0.0 < self.compression_max_ratio <= 1.0:
            raise ValueError("compression_max_ratio must be in (0, 1]")


def load_aux_config() -> AuxiliaryConfig:
    return AuxiliaryConfig().with_env_overrides()
