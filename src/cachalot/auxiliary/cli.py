"""
`cachalot aux` subcommands. Running one is the opt-in for that one command: it enables the
layer and the one feature it exercises, and nothing else, and it never loads a target model.

    cachalot aux check
    cachalot aux respond [--stream] PROMPT
    cachalot aux classify TEXT
    cachalot aux extract --field port:integer --field host TEXT
    cachalot aux compress --file notes.md [--query "..."]
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path


def add_parser(sub) -> None:
    p = sub.add_parser("aux", help="Auxiliary inference (Apple on-device model): check, respond, classify, extract, compress.")
    p.add_argument("--provider", choices=("apple_foundation_models", "deterministic"), default=None)
    p.add_argument("--inline", action="store_true", help="Run the provider in this process instead of a worker child.")
    p.add_argument("--timeout-ms", type=int, default=None)
    aux = p.add_subparsers(dest="aux_command", required=True)
    aux.add_parser("check", help="Print the provider's capabilities as JSON.")
    r = aux.add_parser("respond", help="Generate text.")
    r.add_argument("--instructions", default=None)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--stream", action="store_true")
    r.add_argument("prompt", nargs="+")
    c = aux.add_parser("classify", help="Classify a request and print the route decision.")
    c.add_argument("--has-tools", action="store_true")
    c.add_argument("text", nargs="+")
    e = aux.add_parser("extract", help="Extract typed fields; values not in the text are dropped.")
    e.add_argument("--field", action="append", required=True, help="name[:type], type in string|integer|number|boolean")
    e.add_argument("--file", default=None)
    e.add_argument("text", nargs="*")
    z = aux.add_parser("compress", help="Source-grounded compression of a document.")
    z.add_argument("--file", required=True)
    z.add_argument("--query", default=None)
    z.add_argument("--min-tokens", type=int, default=None)


def _text(args) -> str:
    if getattr(args, "file", None):
        return Path(args.file).read_text()
    return " ".join(args.text) if args.text else sys.stdin.read()


def cmd_aux(args) -> None:
    from cachalot.auxiliary.config import load_aux_config
    from cachalot.auxiliary.engine import AuxiliaryEngine
    from cachalot.auxiliary.errors import AuxError
    from cachalot.auxiliary.provider import GenerationParams
    from cachalot.auxiliary.tasks import ClassifyRequest, ExtractRequest, FieldSpec

    feature = {"classify": "classification", "extract": "structured_extraction",
               "compress": "context_compression"}.get(args.aux_command)
    cfg = replace(load_aux_config(), enabled=True)
    if feature:
        cfg = replace(cfg, **{feature: True})
    if args.provider:
        cfg = replace(cfg, provider=args.provider)
    if args.inline:
        cfg = replace(cfg, isolation="inline")
    if args.timeout_ms:
        cfg = replace(cfg, timeout_ms=args.timeout_ms)
    if getattr(args, "min_tokens", None) is not None:
        cfg = replace(cfg, compression_min_tokens=args.min_tokens)
    cfg.validate()
    engine = AuxiliaryEngine(cfg)
    try:
        if args.aux_command == "check":
            print(json.dumps(engine.provider.capabilities().to_dict(), indent=2))
        elif args.aux_command == "respond":
            params = GenerationParams(max_tokens=args.max_tokens)
            prompt = " ".join(args.prompt)
            if args.stream:
                for chunk in engine.provider.generate_stream(prompt, instructions=args.instructions, params=params):
                    print(chunk, end="", flush=True)
                print()
            else:
                r = engine.provider.generate(prompt, instructions=args.instructions, params=params)
                print(r.text)
                print(f"[{r.latency_s * 1e3:.0f} ms, {r.prompt_tokens} prompt tokens]", file=sys.stderr)
        elif args.aux_command == "classify":
            print(json.dumps(asdict(engine.classify(ClassifyRequest(" ".join(args.text), args.has_tools))), indent=2))
        elif args.aux_command == "extract":
            fields = []
            for spec in args.field:
                name, _, kind = spec.partition(":")
                fields.append(FieldSpec(name, kind or "string"))
            print(json.dumps(asdict(engine.extract(ExtractRequest(_text(args), tuple(fields)))), indent=2))
        elif args.aux_command == "compress":
            r = engine.compress(_text(args), query=args.query)
            print(r.text)
            print(json.dumps({"accepted": r.accepted, "source": r.source, "reject_reason": r.reject_reason,
                              "ratio": round(r.ratio, 3), "approximate_tokens": r.approximate_tokens,
                              "facts": len(r.facts), "ungrounded": len(r.ungrounded_facts),
                              "chunks": r.chunks, "failed_chunks": r.failed_chunks,
                              "latency_s": round(r.latency_s, 2)}), file=sys.stderr)
    except AuxError as exc:
        print(f"auxiliary {exc.kind}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    finally:
        engine.close()
