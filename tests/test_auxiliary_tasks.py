"""Auxiliary tasks, config, schema validation, engine gating and levers, with a scripted provider."""

from __future__ import annotations

import pytest

from cachalot.auxiliary import errors as E
from cachalot.auxiliary.config import AuxiliaryConfig
from cachalot.auxiliary.engine import AuxiliaryEngine
from cachalot.auxiliary.levers import LEVERS, CompressionLever, RateConstants, RoutingLever
from cachalot.auxiliary.provider import Capabilities, GenerationResult, UnavailableProvider
from cachalot.auxiliary.schema import validate
from cachalot.auxiliary.tasks import (
    ClassifyRequest,
    ExtractRequest,
    Fact,
    FieldSpec,
    Segment,
    classify,
    compress,
    extract,
    fact_grounded,
    grounded_in,
    heuristic_classify,
    segment_text,
)


class Scripted:
    """Provider double: returns queued structured values or raises queued errors."""

    name = "scripted"

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def capabilities(self):
        return Capabilities(provider=self.name, model="test", available=True, structured_output=True)

    def is_available(self):
        return True

    def generate_structured(self, prompt, schema, *, instructions=None, params=None, cancel=None):
        self.prompts.append(prompt)
        out = self.outputs.pop(0) if self.outputs else {}
        if callable(out):
            out = out(prompt)
        if isinstance(out, Exception):
            raise out
        return GenerationResult(text="{}", latency_s=0.01, structured=out)

    def close(self):
        pass


# ------------------------------------------------------------------------ config
def test_config_defaults_are_off_and_local():
    cfg = AuxiliaryConfig()
    assert not cfg.enabled and cfg.local_only and cfg.model == "system"
    assert not any(cfg.feature(f) for f in ("classification", "structured_extraction", "context_compression",
                                             "semantic_validation", "adaptive_routing"))


def test_config_env_overrides_and_feature_needs_enabled():
    cfg = AuxiliaryConfig().with_env_overrides({"CACHALOT_AUX_CLASSIFICATION": "1", "CACHALOT_AUX_TIMEOUT_MS": "500"})
    assert cfg.classification and cfg.timeout_s == 0.5 and not cfg.feature("classification")
    cfg = cfg.with_env_overrides({"CACHALOT_AUX_ENABLED": "true"})
    assert cfg.feature("classification")


@pytest.mark.parametrize("env", [{"CACHALOT_AUX_MODEL": "pcc"}, {"CACHALOT_AUX_PROVIDER": "cloud"},
                                 {"CACHALOT_AUX_ENABLED": "maybe"}, {"CACHALOT_AUX_ISOLATION": "thread"}])
def test_config_rejects_bad_values(env):
    with pytest.raises(ValueError):
        AuxiliaryConfig().with_env_overrides(env)


# ------------------------------------------------------------------------ schema
def test_schema_validator():
    schema = {"type": "object", "required": ["a", "b"], "additionalProperties": False,
              "properties": {"a": {"type": "integer", "minimum": 0}, "b": {"type": "array", "maxItems": 1,
                                                                         "items": {"type": "string", "enum": ["x"]}}}}
    assert validate({"a": 1, "b": ["x"]}, schema) == []
    assert validate({"a": True, "b": ["x"]}, schema)            # bool is not an integer
    assert len(validate({"a": -1, "b": ["y", "x"], "c": 0}, schema)) == 4


# ---------------------------------------------------------------- classification
def test_heuristic_routes():
    assert heuristic_classify(ClassifyRequest("What is 12*7?")).route == "deterministic"
    assert heuristic_classify(ClassifyRequest("compute (12+8)*3")).route == "deterministic"
    assert heuristic_classify(ClassifyRequest("Fix the segfault in resident_store.py")).route == "target"
    assert heuristic_classify(ClassifyRequest("hello")).route == "auxiliary"


def test_classify_model_path_and_conservative_disagreement():
    simple = {"task_type": "question", "complexity": "simple", "needs_tools": False, "needs_large_model": False}
    c = classify(ClassifyRequest("Who wrote Hamlet?"), Scripted(simple))
    assert c.source == "model" and c.route == "auxiliary"
    # the model calls it simple, the heuristic sees a hard request: uncertain, never auxiliary
    c = classify(ClassifyRequest("Analyze the trade-offs between LFU and LRU for this workload"), Scripted(simple))
    assert c.route == "uncertain"
    # the heuristic routes it to the target for another reason (task type): still not agreement
    c = classify(ClassifyRequest("Before: 80.1, 79.8 ms. After: 85.3, 84.9 ms. Is that significant"), Scripted(simple))
    assert c.route == "uncertain"


def test_classify_falls_back_on_invalid_and_errors():
    req = ClassifyRequest("Who wrote Hamlet?")
    bad = classify(req, Scripted({"task_type": "poetry", "complexity": "simple", "needs_tools": False,
                                  "needs_large_model": False}))
    assert bad.source == "heuristic" and bad.fallback_reason == "invalid_output" and bad.problems
    t = classify(req, Scripted(E.AuxTimeout("slow")))
    assert t.source == "heuristic" and t.fallback_reason == "timeout" and t.route == "target"
    assert classify(req, None).fallback_reason == "no_provider"


def test_classify_skips_model_for_deterministic():
    p = Scripted()
    c = classify(ClassifyRequest("2 + 2"), p)
    assert c.route == "deterministic" and p.prompts == []


# -------------------------------------------------------------------- extraction
def test_grounding():
    text = "Listen on port 8,080 at host 127.0.0.1 with budget 48 GiB."
    assert grounded_in(8080, text) and grounded_in(48.0, text) and grounded_in("127.0.0.1", text)
    assert not grounded_in(80, text) and not grounded_in("10.0.0.1", text)
    assert grounded_in(1240.5, "total 1,240.50 EUR") and not grounded_in(0, "version 0.69.0")


def test_extract_drops_ungrounded_values():
    fields = (FieldSpec("port", "integer"), FieldSpec("host"), FieldSpec("user"))
    out = {"port": 8080, "host": "10.0.0.9", "user": ""}
    r = extract(ExtractRequest("serve on port 8080 at host 127.0.0.1", fields), Scripted(out))
    assert r.values == {"port": 8080} and r.ungrounded == ["host"] and set(r.missing) == {"host", "user"}


def test_extract_heuristic_fallback():
    r = extract(ExtractRequest("port: 9000\nhost = example.org", (FieldSpec("port", "integer"), FieldSpec("host"))),
                Scripted(E.AuxUnavailable("off")))
    assert r.source == "heuristic" and r.values == {"port": 9000, "host": "example.org"}


def test_field_spec_validation():
    with pytest.raises(ValueError):
        FieldSpec("Bad Name")
    with pytest.raises(ValueError):
        FieldSpec("x", "date")


# ------------------------------------------------------------------- compression
DOC = "\n\n".join([
    "The cache server was deployed on 2026-10-01 in the Frankfurt region.",
    "You must not delete the snapshot directory without approval.",
    "```python\nbudget = 48\n```",
    "Throughput reached 1,250 requests per second after the change to version 3.4.1.",
    "The team met twice and discussed a number of general topics at length over coffee.",
] * 6)


def test_segmenting_keeps_code_whole():
    segs = segment_text("a\n\n```\nx\n\ny\n```\n\nb")
    assert [s.text for s in segs] == ["a", "```\nx\n\ny\n```", "b"]


def test_fact_grounding_checks_numbers_and_ids():
    segs = {"s4": Segment("s4", "Throughput reached 1,250 requests per second after version 3.4.1.")}
    assert fact_grounded(Fact("Throughput 1,250 req/s with 3.4.1", "s4", "high"), segs)
    assert not fact_grounded(Fact("Throughput 1,300 req/s", "s4", "high"), segs)
    assert not fact_grounded(Fact("Throughput 1,250", "s9", "high"), segs)


def _facts_for(prompt: str):
    facts = []
    for line in prompt.splitlines():
        if line.startswith("[s") and "Throughput" in line:
            facts.append({"text": "Throughput reached 1,250 requests per second on 3.4.1",
                          "source_id": line[1:line.index("]")], "importance": "high"})
    return {"facts": facts}


def test_compress_keeps_constraints_verbatim_and_cites_sources():
    r = compress(DOC, Scripted(_facts_for, _facts_for, _facts_for), min_tokens=10, max_ratio=0.9,
                 chunk_tokens=200)
    assert r.accepted and r.source == "model"
    assert "You must not delete the snapshot directory without approval." in r.text
    assert "budget = 48" in r.text
    assert all(f.source_id.startswith("s") for f in r.facts) and r.ungrounded_facts == []
    assert "coffee" not in r.text and r.compressed_tokens < r.original_tokens


def test_compress_rejects_ungrounded_and_keeps_failed_chunks():
    lie = {"facts": [{"text": "Throughput reached 9,999 requests per second", "source_id": "s4",
                      "importance": "critical"}]}
    r = compress(DOC, Scripted(lie, lie, lie, lie), min_tokens=10, max_ratio=0.95, chunk_tokens=200)
    assert not r.accepted and r.reject_reason == "too_many_ungrounded_facts" and r.text == DOC
    r = compress(DOC, Scripted(E.AuxTimeout("x"), E.AuxTimeout("x"), E.AuxTimeout("x"), E.AuxTimeout("x")),
                 min_tokens=10, max_ratio=1.0, chunk_tokens=200)
    assert r.failed_chunks == r.chunks and "coffee" in r.text      # lossless: every chunk kept verbatim


def test_compress_skips_short_and_cancel_returns_original():
    assert compress("short text", Scripted()).reject_reason == "below_min_tokens"
    r = compress(DOC, Scripted(E.AuxCancelled("x")), min_tokens=10, chunk_tokens=200)
    assert not r.accepted and r.reject_reason == "cancelled" and r.text == DOC


def test_compress_heuristic_baseline():
    r = compress(DOC, None, min_tokens=10, max_ratio=0.95)
    assert r.source == "heuristic" and "must not delete" in r.text and "coffee" not in r.text


# ------------------------------------------------------------------- engine/levers
def test_engine_gates_every_feature():
    eng = AuxiliaryEngine(AuxiliaryConfig(), provider=Scripted())
    with pytest.raises(E.AuxDisabled):
        eng.classify(ClassifyRequest("hi"))
    with pytest.raises(E.AuxDisabled):
        eng.compress(DOC)


def test_engine_unavailable_provider_uses_heuristics_and_counts():
    cfg = AuxiliaryConfig(enabled=True, classification=True)
    eng = AuxiliaryEngine(cfg, provider=UnavailableProvider())
    c = eng.classify(ClassifyRequest("Who wrote Hamlet?"))
    assert c.source == "heuristic"
    s = eng.stats()["tasks"]["classification"]
    assert s["calls"] == 1 and s["heuristic"] == 1 and s["fallbacks"] == {"no_provider": 1}


def test_compression_lever_eligibility_and_cost():
    cfg = AuxiliaryConfig(enabled=True, context_compression=True, compression_min_tokens=100)
    rates = RateConstants(target_prefill_tok_s=100.0, aux_prefill_tok_s=1000.0, aux_decode_tok_s=40.0)
    lever = CompressionLever(AuxiliaryEngine(cfg, provider=Scripted()), rates=rates, min_net_seconds=1.0)
    big = {"text": "x" * 40_000}
    est = lever.estimate_cost(big)
    assert est.saved_target_seconds > est.aux_seconds > 0
    assert lever.eligibility(big) == (True, "eligible")
    assert lever.eligibility({"text": "x" * 40_000, "reused_prefix_tokens": 10_000})[0] is False   # cached already
    assert lever.execute({"text": "short"}).reason == "below_min_tokens"
    off = CompressionLever(AuxiliaryEngine(AuxiliaryConfig(), provider=Scripted()))
    assert off.execute(big).applied is False and off.execute(big).value == big["text"]


def test_routing_lever_defaults_to_target():
    lever = RoutingLever(AuxiliaryEngine(AuxiliaryConfig(), provider=Scripted()))
    out = lever.execute(ClassifyRequest("hello"))
    assert out.value == "target" and not out.applied
    simple = {"task_type": "chat", "complexity": "trivial", "needs_tools": False, "needs_large_model": False}
    cfg = AuxiliaryConfig(enabled=True, classification=True, adaptive_routing=True)
    lever = RoutingLever(AuxiliaryEngine(cfg, provider=Scripted(simple)))
    assert lever.execute(ClassifyRequest("hello there")).value == "auxiliary"


def test_lever_registry_declares_output_semantics():
    assert set(LEVERS) == set("ABCDEFG")
    assert LEVERS["B"].changes_output and LEVERS["C"].changes_output and not LEVERS["A"].changes_output
    assert LEVERS["F"].status == "no-go"
    assert all(s.status in ("implemented", "experimental", "research", "no-go") for s in LEVERS.values())
