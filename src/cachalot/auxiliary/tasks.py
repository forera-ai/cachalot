"""
Typed auxiliary tasks: request classification, field extraction, source-grounded compression.

Each task has a typed request and result, a JSON schema the provider decodes against, a
deterministic validator that runs on every result, and a deterministic heuristic that is
both the fallback (provider unavailable, timed out, refused or invalid) and the baseline
every benchmark compares the model against. A result always says which path produced it
(`source`: "model" or "heuristic") and why a fallback happened.

Nothing here asks the model for a confidence value: a small model's self-reported
confidence is not calibrated, and routing does not read one.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cachalot.auxiliary.errors import AuxError, AuxInvalidOutput
from cachalot.auxiliary.provider import AuxiliaryProvider, GenerationParams
from cachalot.auxiliary.schema import validate

# ============================================================================ classification

TASK_TYPES = ("chat", "question", "code", "math", "summarize", "extract", "translate", "tool_use", "other")
COMPLEXITY = ("trivial", "simple", "complex")
ROUTES = ("deterministic", "auxiliary", "target", "uncertain")
# Task types a 3B on-device model may answer on its own when they are also simple.
AUX_SAFE_TYPES = frozenset({"chat", "question", "summarize", "extract", "translate"})

CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_type": {"type": "string", "enum": list(TASK_TYPES)},
        "complexity": {"type": "string", "enum": list(COMPLEXITY)},
        "needs_tools": {"type": "boolean"},
        "needs_large_model": {"type": "boolean"},
    },
    "required": ["task_type", "complexity", "needs_tools", "needs_large_model"],
}

CLASSIFY_INSTRUCTIONS = """You label requests sent to an AI assistant. Do not answer the request.
task_type: chat (greeting, small talk), question (a factual or general question), code (writing, reading, debugging or reviewing code), math (calculation or proof), summarize, extract (pull fields or facts out of given text), translate, tool_use (needs files, commands, the web or other tools), other.
complexity: trivial (one short obvious answer), simple (a short answer a small model gets right), complex (multi-step reasoning, long or technical output, specialist knowledge, or anything where a wrong answer is costly).
needs_tools: true if answering needs files, commands, browsing or other tools.
needs_large_model: true unless a small on-device model would answer correctly and completely."""


@dataclass(frozen=True)
class ClassifyRequest:
    text: str
    has_tools: bool = False          # the client offered tools with the request

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("ClassifyRequest.text must be a non-empty string")


@dataclass
class Classification:
    task_type: str
    complexity: str
    needs_tools: bool
    needs_large_model: bool
    route: str
    source: str                      # "model" | "heuristic"
    latency_s: float = 0.0
    fallback_reason: str | None = None
    problems: list[str] = field(default_factory=list)


_CODE = re.compile(r"```|\bdef |\bclass |\bfunction\b|\breturn\b|\bimport |#include|=>|\bstack ?trace\b|Traceback|"
                   r"\b(bug|compile|refactor|regex|SQL|API|endpoint|segfault|exception|unit test)s?\b|\.(py|js|ts|cs|cpp|rs|go|java)\b",
                   re.I)
_TOOLS = re.compile(r"\b(run|execute|open|read|list|search|browse|download|install|delete|grep|find) (the |my |this |all )?"
                    r"(file|files|folder|directory|repo|repository|command|web|internet|page|url|tests?)\b|\bterminal\b|\bshell\b", re.I)
_MATH = re.compile(r"\b(prove|integral|derivative|equation|solve|probability|matrix|theorem)\b|[0-9]\s*[-+*/^]\s*[0-9]", re.I)
_ARITH = re.compile(r"^\s*(what is|what's|compute|calculate)?\s*\(?\s*[0-9][0-9\s.+\-*/()]*[0-9)]\s*\??\s*$", re.I)
_GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening|night)|bye)\b[\s!.,]*\w{0,12}[\s!.]*$", re.I)
_SUMMARIZE = re.compile(r"\b(summari[sz]e|tl;?dr|in one sentence|key points)\b", re.I)
_TRANSLATE = re.compile(r"\btranslate\b|\bin (french|german|spanish|persian|farsi|italian|japanese|chinese)\b", re.I)
_EXTRACT = re.compile(r"\b(extract|pull out|list the (names|dates|numbers|emails))\b", re.I)
_HARD = re.compile(r"\b(design|architect|optimi[sz]e|analy[sz]e|compare|trade-?offs?|why does|explain how|step by step|"
                   r"benchmark|security|legal|medical|diagnos)\w*\b", re.I)


def heuristic_classify(req: ClassifyRequest) -> Classification:
    text = req.text.strip()
    words = len(text.split())
    if _ARITH.match(text):
        return Classification("math", "trivial", False, False, "deterministic", "heuristic")
    if _TOOLS.search(text) or req.has_tools and re.search(r"\b(file|repo|command|run)\b", text, re.I):
        task, needs_tools = "tool_use", True
    elif _CODE.search(text):
        task, needs_tools = "code", False
    elif _MATH.search(text):
        task, needs_tools = "math", False
    elif _SUMMARIZE.search(text):
        task, needs_tools = "summarize", False
    elif _TRANSLATE.search(text):
        task, needs_tools = "translate", False
    elif _EXTRACT.search(text):
        task, needs_tools = "extract", False
    elif _GREETING.match(text):
        task, needs_tools = "chat", False
    elif text.endswith("?") or re.match(r"^(what|who|when|where|which|how|is|are|can|does|do)\b", text, re.I):
        task, needs_tools = "question", False
    else:
        task, needs_tools = "other", False
    if task in ("code", "tool_use", "math") or _HARD.search(text) or words > 120:
        complexity = "complex"
    elif task == "chat" or words <= 8:
        complexity = "trivial"
    else:
        complexity = "simple"
    needs_large = complexity == "complex" or task not in AUX_SAFE_TYPES or needs_tools
    c = Classification(task, complexity, needs_tools, needs_large, "target", "heuristic")
    c.route = decide_route(c, None)
    return c


def decide_route(c: Classification, heuristic: Classification | None) -> str:
    """Conservative routing: anything short of agreement that a request is simple goes to the target."""
    if c.route == "deterministic":
        return "deterministic"
    if heuristic is not None and heuristic.route == "deterministic":
        return "deterministic"
    small_ok = (not c.needs_large_model and not c.needs_tools and c.complexity in ("trivial", "simple")
                and c.task_type in AUX_SAFE_TYPES)
    if not small_ok:
        return "target"
    if heuristic is not None and (heuristic.complexity == "complex" or heuristic.route == "target"):
        return "uncertain"   # the two disagree: the target answers
    return "auxiliary"


def classify(req: ClassifyRequest, provider: AuxiliaryProvider | None, *,
             timeout_s: float | None = None, cancel: threading.Event | None = None) -> Classification:
    t0 = time.perf_counter()
    base = heuristic_classify(req)
    if base.route == "deterministic":
        return base                  # a rule answers it; no model call
    if provider is None:
        base.fallback_reason = "no_provider"
        return base
    prompt = f"Request:\n{req.text}\n" + ("(Tools were offered with this request.)\n" if req.has_tools else "")
    try:
        out = provider.generate_structured(prompt, CLASSIFY_SCHEMA, instructions=CLASSIFY_INSTRUCTIONS,
                                           params=GenerationParams(max_tokens=64, timeout_s=timeout_s),
                                           cancel=cancel)
        value = out.structured or {}
        problems = validate(value, CLASSIFY_SCHEMA)
        if problems:
            raise AuxInvalidOutput("classification failed validation", problems=problems, raw=out.text)
    except AuxError as exc:
        # The model could not label it (refused, timed out, invalid): the heuristic's labels are kept
        # for the record, but the route is the target's. A failed classifier never offloads.
        base.fallback_reason = exc.kind
        base.problems = list(getattr(exc, "problems", []))
        base.route = "target"
        base.latency_s = time.perf_counter() - t0
        return base
    c = Classification(
        task_type=value["task_type"], complexity=value["complexity"],
        needs_tools=bool(value["needs_tools"]) or (req.has_tools and base.needs_tools),
        needs_large_model=bool(value["needs_large_model"]), route="target", source="model",
    )
    c.route = decide_route(c, base)
    c.latency_s = time.perf_counter() - t0
    return c


# ================================================================================ extraction

FIELD_TYPES = ("string", "integer", "number", "boolean")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str = "string"
    description: str = ""

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", self.name):
            raise ValueError(f"field name {self.name!r} must be snake_case, 1-48 chars")
        if self.type not in FIELD_TYPES:
            raise ValueError(f"field type {self.type!r} not in {FIELD_TYPES}")


@dataclass(frozen=True)
class ExtractRequest:
    text: str
    fields: tuple[FieldSpec, ...]

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("ExtractRequest.text must be non-empty")
        if not self.fields or len({f.name for f in self.fields}) != len(self.fields):
            raise ValueError("ExtractRequest.fields must be non-empty with unique names")


@dataclass
class Extraction:
    values: dict[str, Any]           # only values that occur in the text (grounded_in)
    missing: list[str]
    ungrounded: list[str]            # the model returned a value the text does not contain: dropped
    source: str
    latency_s: float = 0.0
    fallback_reason: str | None = None
    problems: list[str] = field(default_factory=list)


def extraction_schema(fields: tuple[FieldSpec, ...]) -> dict[str, Any]:
    """Flat: one typed value a field. The model is not asked whether it found a field: measured
    2026-10-29, the system model's found flags were false for values it had copied correctly, and
    it filled absent fields with invented values ("password": "password"). Grounding decides."""
    return {"type": "object", "properties": {f.name: {"type": f.type} for f in fields},
            "required": [f.name for f in fields]}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def grounded_in(value: Any, text: str) -> bool:
    """Deterministic grounding: the value, as written, occurs in the source text."""
    hay = _norm(text)
    if isinstance(value, bool):
        return True                  # a yes/no judgment has no literal span to check
    if isinstance(value, (int, float)):
        # numerically: "1,240.50" grounds 1240.5; thousands separators are dropped first
        numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])", re.sub(r"(?<=\d),(?=\d{3})", "", hay))
        return any(abs(float(n) - float(value)) < 1e-9 for n in numbers)
    s = _norm(str(value))
    return bool(s) and s in hay


def _heuristic_value(f: FieldSpec, text: str) -> Any:
    label = re.escape(f.name.replace("_", " "))
    m = re.search(rf"(?:^|\n|[;,.]\s*)\W*{label}\s*(?:[:=]|is|was|of)\s*([^\n;,]+)", text, re.I)
    if not m:
        return None
    raw = m.group(1).strip().rstrip(".")
    if f.type in ("integer", "number"):
        n = re.search(r"-?\d+(?:\.\d+)?", raw.replace(",", ""))
        if not n:
            return None
        return int(float(n.group())) if f.type == "integer" else float(n.group())
    if f.type == "boolean":
        return raw.lower() in ("yes", "true", "on", "enabled", "1")
    return raw


def heuristic_extract(req: ExtractRequest) -> Extraction:
    values, missing = {}, []
    for f in req.fields:
        v = _heuristic_value(f, req.text)
        if v is None:
            missing.append(f.name)
        else:
            values[f.name] = v
    return Extraction(values, missing, [], "heuristic")


def extract(req: ExtractRequest, provider: AuxiliaryProvider | None, *,
            timeout_s: float | None = None, cancel: threading.Event | None = None) -> Extraction:
    t0 = time.perf_counter()
    if provider is None:
        r = heuristic_extract(req)
        r.fallback_reason = "no_provider"
        return r
    schema = extraction_schema(req.fields)
    described = "\n".join(f"- {f.name} ({f.type}): {f.description or f.name.replace('_', ' ')}" for f in req.fields)
    instructions = ("Extract fields from the text. Copy values exactly as written in the text. "
                    "If the text does not state a field, give an empty string, or 0 for a number. Never guess.")
    prompt = f"Fields:\n{described}\n\nText:\n{req.text}"
    try:
        out = provider.generate_structured(prompt, schema, instructions=instructions,
                                           params=GenerationParams(max_tokens=64 + 48 * len(req.fields),
                                                                   timeout_s=timeout_s),
                                           cancel=cancel)
        value = out.structured or {}
        problems = validate(value, schema)
        if problems:
            raise AuxInvalidOutput("extraction failed validation", problems=problems, raw=out.text)
    except AuxError as exc:
        r = heuristic_extract(req)
        r.fallback_reason = exc.kind
        r.problems = list(getattr(exc, "problems", []))
        r.latency_s = time.perf_counter() - t0
        return r
    values, missing, ungrounded = {}, [], []
    for f in req.fields:
        item = value[f.name]
        # "" and 0 are the instructed "not stated" placeholders, so a stated zero reads as missing:
        # this schema cannot tell them apart (measured 2026-10-29: a placeholder 0 "grounded" in "0.69.0")
        if item in ("", None) or (f.type in ("integer", "number") and item == 0):
            missing.append(f.name)
        elif grounded_in(item, req.text):
            values[f.name] = item
        else:
            missing.append(f.name)
            ungrounded.append(f.name)
    return Extraction(values, missing, ungrounded, "model", latency_s=time.perf_counter() - t0)


# =============================================================================== compression

# Lines a compressor may never paraphrase: instructions, prohibitions, requirements.
CRITICAL = re.compile(
    r"\b(must|must not|mustn't|never|do not|don't|always|required|requires?|shall|forbidden|prohibited|"
    r"only|important|warning|caution|deadline|constraint|no later than|at least|at most|exactly)\b",
    re.I,
)
FENCE = re.compile(r"```.*?```", re.S)
IMPORTANCE = ("critical", "high", "normal")

COMPRESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_id": {"type": "string"},
                    "importance": {"type": "string", "enum": list(IMPORTANCE)},
                },
                "required": ["text", "source_id", "importance"],
            },
        },
    },
    "required": ["facts"],
}

COMPRESS_INSTRUCTIONS = """You condense source segments into a list of facts for another AI model that will not see the sources.
Each segment starts with its id in square brackets. For every fact give the id of the one segment it comes from.
Keep every number, name, identifier, path, version, date and unit exactly as written. State facts only; never add, infer or generalize.
Mark facts the reader cannot do without as critical. Skip greetings, repetition and filler."""


@dataclass(frozen=True)
class Segment:
    id: str
    text: str


@dataclass
class Fact:
    text: str
    source_id: str
    importance: str


@dataclass
class CompressedContext:
    text: str                        # what the target model would read
    accepted: bool                   # False: use the original; `text` then IS the original
    source: str                      # "model" | "heuristic" | "original"
    original_tokens: int
    compressed_tokens: int
    approximate_tokens: bool         # True when no target tokenizer was supplied
    verbatim_segments: list[str] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    ungrounded_facts: list[Fact] = field(default_factory=list)
    failed_chunks: int = 0
    chunks: int = 0
    latency_s: float = 0.0
    reject_reason: str | None = None

    @property
    def ratio(self) -> float:
        return self.compressed_tokens / self.original_tokens if self.original_tokens else 1.0

    @property
    def ungrounded_share(self) -> float:
        n = len(self.facts) + len(self.ungrounded_facts)
        return len(self.ungrounded_facts) / n if n else 0.0


def segment_text(text: str) -> list[Segment]:
    """Paragraphs as segments (s1, s2, ...); a fenced code block is never split."""
    pieces: list[str] = []
    pos = 0
    for m in FENCE.finditer(text):
        pieces.extend(p for p in re.split(r"\n\s*\n", text[pos:m.start()]) if p.strip())
        pieces.append(m.group())
        pos = m.end()
    pieces.extend(p for p in re.split(r"\n\s*\n", text[pos:]) if p.strip())
    return [Segment(f"s{i + 1}", p.strip()) for i, p in enumerate(pieces)]


def is_protected(seg: Segment) -> bool:
    return seg.text.startswith("```") or bool(CRITICAL.search(seg.text))


def approx_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


# Anchors a fact may not invent: numbers (dates, versions, sizes), identifiers with "_", "." or "/"
# between parts of two or more characters (paths, file names, snake_case), and CamelCase/ALLCAPS names.
# Hyphenated words and short unit abbreviations ("req/s") are left to the model's wording.
_ANCHOR = re.compile(r"\b\d[\d.,:/_-]*\d\b|\b\d\b|\b[A-Za-z_]\w*(?:[_./]\w{2,})+|\b[A-Z][A-Za-z0-9]*[A-Z0-9][A-Za-z0-9]*\b")
_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")


def fact_grounded(fact: Fact, segments: dict[str, Segment]) -> bool:
    """The cited segment exists and contains every number and identifier the fact states."""
    seg = segments.get(fact.source_id)
    if seg is None or not fact.text.strip():
        return False
    src = _DIGIT_COMMA.sub("", seg.text)
    return all(_DIGIT_COMMA.sub("", tok) in src for tok in _ANCHOR.findall(fact.text))


def _render(protected: list[Segment], facts: list[Fact], fallback: list[Segment], n_segments: int) -> str:
    out = [f"[Condensed context: {n_segments} source segments. Constraints and code are verbatim; "
           f"facts are extracted, each with its source segment id.]"]
    if protected:
        out.append("Verbatim:")
        out.extend(f"[{s.id}] {s.text}" for s in protected)
    if facts:
        out.append("Facts:")
        out.extend(f"- [{f.source_id}] {f.text}" for f in facts)
    if fallback:
        out.append("Unprocessed (verbatim):")
        out.extend(f"[{s.id}] {s.text}" for s in fallback)
    return "\n".join(out)


def heuristic_compress_segments(segments: list[Segment], query: str | None = None) -> list[Segment]:
    """Extractive baseline: keep protected segments, and sentences with numbers, identifiers or query words."""
    q = {w for w in re.findall(r"[a-z]{4,}", (query or "").lower())}
    kept: list[Segment] = []
    for seg in segments:
        if is_protected(seg):
            kept.append(seg)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", seg.text)
        keep = [s for s in sentences if _ANCHOR.search(s) or (q and q & set(re.findall(r"[a-z]{4,}", s.lower())))]
        if keep:
            kept.append(Segment(seg.id, " ".join(keep)))
    return kept


def compress(
    text: str,
    provider: AuxiliaryProvider | None,
    *,
    target_tokens: Callable[[str], int] | None = None,
    min_tokens: int = 2_000,
    max_ratio: float = 0.6,
    max_ungrounded: float = 0.1,
    chunk_tokens: int = 3_000,
    query: str | None = None,
    timeout_s: float | None = None,
    cancel: threading.Event | None = None,
) -> CompressedContext:
    """
    Condense `text` for the target model, or return it unchanged.

    `chunk_tokens` is in the auxiliary model's tokens (approximated as chars/4 for packing; the
    provider's own context check rejects an overfull chunk, which then stays verbatim).
    """
    t0 = time.perf_counter()
    count = target_tokens or approx_tokens
    original = count(text)
    segments = segment_text(text)

    def reject(reason: str, **extra) -> CompressedContext:
        r = CompressedContext(text=text, accepted=False, source="original", original_tokens=original,
                              compressed_tokens=original, approximate_tokens=target_tokens is None,
                              latency_s=time.perf_counter() - t0, reject_reason=reason)
        for k, v in extra.items():
            setattr(r, k, v)
        return r

    if original < min_tokens:
        return reject("below_min_tokens")
    protected = [s for s in segments if is_protected(s)]
    rest = [s for s in segments if not is_protected(s)]
    by_id = {s.id: s for s in segments}

    if provider is None:
        kept = heuristic_compress_segments(segments, query)
        rendered = _render([s for s in kept if is_protected(s)], [], [s for s in kept if not is_protected(s)],
                           len(segments))
        r = CompressedContext(rendered, True, "heuristic", original, count(rendered), target_tokens is None,
                              verbatim_segments=[s.id for s in kept], latency_s=time.perf_counter() - t0)
        if r.ratio > max_ratio:
            return reject("ratio_too_high")
        return r

    chunks: list[list[Segment]] = []
    current: list[Segment] = []
    size = 0
    for seg in rest:
        n = approx_tokens(seg.text) + 4
        if current and size + n > chunk_tokens:
            chunks.append(current)
            current, size = [], 0
        current.append(seg)
        size += n
    if current:
        chunks.append(current)

    facts: list[Fact] = []
    ungrounded: list[Fact] = []
    fallback: list[Segment] = []
    failed = 0
    for chunk in chunks:
        ids = {s.id for s in chunk}
        body = "\n\n".join(f"[{s.id}] {s.text}" for s in chunk)
        prompt = (f"Question the reader will answer: {query}\n\n" if query else "") + f"Segments:\n{body}"
        try:
            out = provider.generate_structured(prompt, COMPRESS_SCHEMA, instructions=COMPRESS_INSTRUCTIONS,
                                               params=GenerationParams(max_tokens=1_024, timeout_s=timeout_s),
                                               cancel=cancel)
            value = out.structured or {}
            problems = validate(value, COMPRESS_SCHEMA)
            if problems:
                raise AuxInvalidOutput("compression failed validation", problems=problems)
        except AuxError as exc:
            if exc.kind == "cancelled":
                return reject("cancelled")
            failed += 1
            fallback.extend(chunk)     # lossless for this chunk: its segments stay verbatim
            continue
        for item in value["facts"]:
            fact = Fact(item["text"].strip(), item["source_id"].strip().strip("[]"), item["importance"])
            if fact.source_id in ids and fact_grounded(fact, by_id):
                facts.append(fact)
            else:
                ungrounded.append(fact)
    # a non-protected segment no grounded fact cites is dropped; that is the compression
    rendered = _render(protected, facts, fallback, len(segments))
    r = CompressedContext(rendered, True, "model", original, count(rendered), target_tokens is None,
                          verbatim_segments=[s.id for s in protected + fallback], facts=facts,
                          ungrounded_facts=ungrounded, failed_chunks=failed, chunks=len(chunks),
                          latency_s=time.perf_counter() - t0)
    if r.ungrounded_share > max_ungrounded:
        return reject("too_many_ungrounded_facts", facts=facts, ungrounded_facts=ungrounded,
                      failed_chunks=failed, chunks=len(chunks))
    if r.ratio > max_ratio:
        return reject("ratio_too_high", facts=facts, ungrounded_facts=ungrounded,
                      failed_chunks=failed, chunks=len(chunks))
    return r
