"""
Optimization levers that use auxiliary inference: what each declares, and the small common protocol.

A lever is something that could make a target-model request cheaper or better and that
must be measured before it is trusted. Every lever declares, as data, what it needs, what
it costs, how it fails and whether it changes what the target model sees or says
(`changes_output`). Levers whose execution differs in kind (a scheduling policy, a
research-only feasibility question) are declared here with a status but implement
nothing: forcing them into execute() would be inventing behaviour that was never measured.

Status vocabulary: "implemented" (code and tests exist, off by default), "experimental"
(code exists, quality unproven), "research" (no implementation, a measurement or design is
next), "no-go" (priced and rejected; the document names the numbers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from cachalot.auxiliary.engine import AuxiliaryEngine
from cachalot.auxiliary.tasks import (
    Classification,
    ClassifyRequest,
    CompressedContext,
    approx_tokens,
)


@dataclass(frozen=True)
class LeverSpec:
    id: str
    name: str
    status: str
    feature_flag: str | None
    requires: tuple[str, ...]          # capabilities, by Capabilities field name
    workloads: tuple[str, ...]
    expected_overhead: str
    resources: str
    failure: str
    quality_constraint: str
    metrics: tuple[str, ...]
    changes_output: bool
    doc: str


LEVERS: dict[str, LeverSpec] = {s.id: s for s in (
    LeverSpec(
        "A", "structured request classification", "experimental", "classification",
        ("structured_output",), ("chat", "agent turns"),
        "one structured call, ~0.5-1 s on the system model (measured, docs/APPLE-FM-RESULTS.md)",
        "system model service; no target-model memory", "heuristic classification",
        "never routes on its own; feeds lever C only after the route accuracy gate",
        ("latency", "route accuracy", "false routing rate", "failure rate"), False,
        "docs/APPLE-FM-AUXILIARY.md#lever-a"),
    LeverSpec(
        "B", "source-grounded context compression", "experimental", "context_compression",
        ("structured_output", "token_counting"), ("cold long context", "tool results", "documents"),
        "aux prefill ~0.8-1.2k tok/s + ~42 tok/s per output token (measured)",
        "system model service; shrinks target KV and prefill", "original context, per chunk or whole",
        "verbatim constraints and code; every fact cites a segment and passes the grounding check; "
        "reject above max_ratio or max_ungrounded",
        ("compression ratio", "fact retention", "constraint retention", "ungrounded share",
         "end-to-end latency", "target prefill"), True,
        "docs/APPLE-FM-AUXILIARY.md#lever-b"),
    LeverSpec(
        "C", "adaptive routing to the auxiliary model", "experimental", "adaptive_routing",
        ("structured_output", "text_generation"), ("short chat", "simple questions"),
        "classification call plus, when routed, a full auxiliary answer",
        "system model service", "the target model answers",
        "a false-routing rate gate on a labelled set before any automatic use; uncertain goes to target",
        ("routing accuracy", "target invocations avoided", "task success", "end-to-end latency"), True,
        "docs/APPLE-FM-AUXILIARY.md#lever-c"),
    LeverSpec(
        "D", "semantic cache assistance", "research", None, ("structured_output",),
        ("agent sessions with eviction",), "per-block classification calls",
        "system model service", "existing cache policy unchanged",
        "only through Cachalot's own cache code; no KV access through Apple's API",
        ("hit rate", "recompute", "policy cost"), False, "docs/APPLE-FM-HETEROGENEOUS.md#lever-d"),
    LeverSpec(
        "E", "heterogeneous ANE/GPU scheduling", "research", None, (), ("concurrent target + auxiliary",),
        "contention on unified-memory bandwidth (measured with a GPU stand-in)",
        "ANE + system service beside the MLX GPU working set", "run auxiliary work sequentially",
        "target decode slowdown within its noise band",
        ("target tok/s", "aux tok/s", "combined throughput", "tail latency"), False,
        "docs/APPLE-FM-HETEROGENEOUS.md"),
    LeverSpec(
        "F", "speculative decoding with the system model as drafter", "no-go", None,
        ("logits", "tokenizer_access", "kv_access"), (), "n/a", "n/a", "n/a",
        "lossless verification by the target", ("acceptance", "verify cost"), False,
        "docs/APPLE-FM-SPECULATIVE-FEASIBILITY.md"),
    LeverSpec(
        "G", "semantic output validation", "research", "semantic_validation", ("structured_output",),
        ("summaries", "extraction answers"), "one structured call per checked answer",
        "system model service", "no validation (deterministic validators always run)",
        "advisory only; never rewrites target output",
        ("false positives", "false negatives", "latency"), False, "docs/APPLE-FM-AUXILIARY.md#lever-g"),
)}


@dataclass
class CostEstimate:
    aux_seconds: float
    saved_target_seconds: float
    basis: str                          # which constants, measured when

    @property
    def net_seconds(self) -> float:
        return self.saved_target_seconds - self.aux_seconds


@dataclass
class LeverOutcome:
    applied: bool
    value: Any
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class OptimizationLever(Protocol):
    spec: LeverSpec

    def eligibility(self, request: Any) -> tuple[bool, str]: ...

    def estimate_cost(self, request: Any) -> CostEstimate | None: ...

    def execute(self, request: Any) -> LeverOutcome: ...

    def fallback(self, request: Any, reason: str) -> LeverOutcome: ...

    def report_metrics(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RateConstants:
    """Throughputs the compression cost model reads. Dated; stale when drive, budget or path changes."""

    target_prefill_tok_s: float          # cold prefill of the target, its own tokens
    aux_prefill_tok_s: float             # system model prompt processing
    aux_decode_tok_s: float              # system model output
    fact_output_fraction: float = 0.15   # aux output tokens per aux input token, compression prompts
    basis: str = ""


# DeepSeek V4.1 Flash cold prefill 99 tok/s (HANDOFF 18.64, 0.58.2, 2026-10-05, a 22.5k system block);
# GLM-5.3 ~90 tok/s (README, 0.18.0). System-model rates: docs/APPLE-FM-RESULTS.md (2026-10-29).
DEEPSEEK_RATES = RateConstants(99.0, 1_000.0, 42.0, basis="DeepSeek 0.58.2 cold prefill; system model 2026-10-29 probe")
GLM_RATES = RateConstants(90.0, 1_000.0, 42.0, basis="GLM 0.18.0 prefill; system model 2026-10-29 probe")


class CompressionLever:
    spec = LEVERS["B"]

    def __init__(self, engine: AuxiliaryEngine, rates: RateConstants = DEEPSEEK_RATES,
                 target_tokens=None, min_net_seconds: float = 5.0):
        self.engine = engine
        self.rates = rates
        self.target_tokens = target_tokens
        self.min_net_seconds = min_net_seconds
        self.outcomes: dict[str, int] = {}

    def estimate_cost(self, request: dict[str, Any]) -> CostEstimate:
        """request: {"text": str, "reused_prefix_tokens": int} -- a cached prefix costs the target nothing."""
        text = request["text"]
        n_target = (self.target_tokens or approx_tokens)(text) - int(request.get("reused_prefix_tokens", 0))
        n_aux = approx_tokens(text)
        r = self.rates
        # The auxiliary model writes the facts token by token, so its output is the expensive side:
        # break-even is (1 - f) / target_prefill > 1 / aux_prefill + f / aux_decode, f the kept share
        # (f < ~0.27 at DeepSeek's 99 tok/s against 1,000 / 42 tok/s).
        out = n_aux * r.fact_output_fraction
        aux = n_aux / r.aux_prefill_tok_s + out / r.aux_decode_tok_s
        kept = max(0.0, n_target) * r.fact_output_fraction
        saved = max(0.0, n_target - kept) / r.target_prefill_tok_s
        return CostEstimate(aux, saved, r.basis)

    def eligibility(self, request: dict[str, Any]) -> tuple[bool, str]:
        if not self.engine.config.feature("context_compression"):
            return False, "disabled"
        if (self.target_tokens or approx_tokens)(request["text"]) < self.engine.config.compression_min_tokens:
            return False, "below_min_tokens"
        est = self.estimate_cost(request)
        if est.net_seconds < self.min_net_seconds:
            return False, f"net {est.net_seconds:.1f} s below {self.min_net_seconds:.1f} s"
        return True, "eligible"

    def execute(self, request: dict[str, Any]) -> LeverOutcome:
        ok, why = self.eligibility(request)
        if not ok:
            return self.fallback(request, why)
        result: CompressedContext = self.engine.compress(request["text"], target_tokens=self.target_tokens,
                                                         query=request.get("query"))
        key = "accepted" if result.accepted else (result.reject_reason or "rejected")
        self.outcomes[key] = self.outcomes.get(key, 0) + 1
        return LeverOutcome(result.accepted, result.text, None if result.accepted else result.reject_reason,
                            {"ratio": result.ratio, "facts": len(result.facts),
                             "ungrounded": len(result.ungrounded_facts), "latency_s": result.latency_s})

    def fallback(self, request: dict[str, Any], reason: str) -> LeverOutcome:
        self.outcomes[reason] = self.outcomes.get(reason, 0) + 1
        return LeverOutcome(False, request["text"], reason)

    def report_metrics(self) -> dict[str, Any]:
        return {"lever": self.spec.id, "outcomes": dict(self.outcomes)}


class RoutingLever:
    """Lever C: returns a route decision; it never answers a request itself."""

    spec = LEVERS["C"]

    def __init__(self, engine: AuxiliaryEngine):
        self.engine = engine
        self.routes: dict[str, int] = {}

    def eligibility(self, request: ClassifyRequest) -> tuple[bool, str]:
        cfg = self.engine.config
        if not (cfg.feature("adaptive_routing") and cfg.feature("classification")):
            return False, "disabled"
        return True, "eligible"

    def estimate_cost(self, request: ClassifyRequest) -> CostEstimate | None:
        return None   # depends on the route; measured per request instead

    def execute(self, request: ClassifyRequest) -> LeverOutcome:
        ok, why = self.eligibility(request)
        if not ok:
            return self.fallback(request, why)
        c: Classification = self.engine.classify(request)
        self.routes[c.route] = self.routes.get(c.route, 0) + 1
        route = "target" if c.route == "uncertain" else c.route
        return LeverOutcome(route != "target", route, c.fallback_reason,
                            {"classification": vars(c).copy()})

    def fallback(self, request: ClassifyRequest, reason: str) -> LeverOutcome:
        self.routes["target"] = self.routes.get("target", 0) + 1
        return LeverOutcome(False, "target", reason)

    def report_metrics(self) -> dict[str, Any]:
        return {"lever": self.spec.id, "routes": dict(self.routes)}
