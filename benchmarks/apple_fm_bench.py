"""
Apple Foundation Models auxiliary engine: the benchmark of docs/APPLE-FM-RECORD.md (AF-1 to AF-19).

No arm loads a target model. Each run writes a manifest (benchmarks/run_manifest.py) and one JSON row a
measurement to benchmarks/results/apple-fm/<arm>-<utc>.jsonl, and prints medians with their spread.
Arms run one at a time; run them on a quiet machine with no runtime process (the machine gate).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \
      benchmarks/apple_fm_bench.py overhead --trials 7
    ... classify | extract | compress | memory | contention

Row fields follow the mission's list where they apply; a field an arm cannot measure is absent, never zero
(target_model is "none" for every arm here: they measure the auxiliary side only).
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import run_manifest as rm  # noqa: E402
from cachalot.auxiliary.apple_fm import AppleFoundationModelsProvider  # noqa: E402
from cachalot.auxiliary.errors import AuxError  # noqa: E402
from cachalot.auxiliary.isolated import IsolatedProvider  # noqa: E402
from cachalot.auxiliary.provider import GenerationParams  # noqa: E402
from cachalot.auxiliary.tasks import (  # noqa: E402
    ClassifyRequest,
    ExtractRequest,
    FieldSpec,
    classify,
    compress,
    extract,
    heuristic_classify,
    heuristic_extract,
)

FIX = HERE / "apple_fm"
OUT = REPO / "benchmarks" / "results" / "apple-fm"
DEEPSEEK_TOKENIZER = Path("/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/tokenizer.json")
DEEPSEEK_PREFILL_TOK_S = 99.0   # HANDOFF 18.64, 0.58.2, cold 22.5k block (dated constant)

WORDS = ("river mill bridge village square stone market harbor tower garden lantern winter summer road field "
         "orchard station library clock window merchant baker painter sailor letter forest valley").split()


def utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def spread(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    if not xs:
        return {}
    q = statistics.quantiles(xs, n=4) if len(xs) >= 4 else [xs[0], statistics.median(xs), xs[-1]]
    return {"n": len(xs), "median": statistics.median(xs), "p25": q[0], "p75": q[-1], "min": xs[0], "max": xs[-1]}


class Recorder:
    def __init__(self, arm: str, workload: str, fixtures: list[Path]):
        OUT.mkdir(parents=True, exist_ok=True)
        text = b"".join(p.read_bytes() for p in fixtures)
        man = rm.collect(workload=workload, workload_text=text, instrument=f"apple_fm_bench:{arm}",
                         extra={"apple_fm_sdk": _sdk_version(), "fixtures": [str(p.relative_to(REPO)) for p in fixtures]})
        self.manifest_id = rm.write_manifest(OUT, man)
        self.path = OUT / f"{arm}-{utc()}.jsonl"
        self.base = {
            "experiment_id": f"apple-fm-{arm}", "manifest_id": self.manifest_id,
            "hardware": f"{rm.hardware().get('model')} {rm.hardware().get('chip')} {rm.hardware().get('memory_gib')} GiB",
            "macos_version": platform.mac_ver()[0], "provider": "apple_foundation_models",
            "provider_version": _sdk_version(), "target_model": "none", "fixtures_sha256": rm.content_hash(text),
        }

    def row(self, **fields) -> dict:
        r = {**self.base, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        return r


def _sdk_version() -> str | None:
    try:
        import importlib.metadata as md

        return md.version("apple-fm-sdk")
    except Exception:
        return None


def filler(n_words: int, seed: int) -> str:
    rnd = random.Random(seed)
    out = []
    while len(out) < n_words:
        s = [rnd.choice(WORDS) for _ in range(rnd.randint(8, 14))]
        out.extend(s[:-1] + [s[-1] + "."])
    return " ".join(out[:n_words])


def provider() -> AppleFoundationModelsProvider:
    p = AppleFoundationModelsProvider(timeout_s=60)
    if not p.is_available():
        sys.exit(f"system model unavailable: {p.capabilities().unavailable_reason}")
    return p


# ============================================================================ overhead (AF-1..AF-6)
def arm_overhead(args) -> None:
    rec = Recorder("overhead", "W1:apple-fm-overhead", [])
    p = provider()
    iso = IsolatedProvider("apple_foundation_models", timeout_s=60)
    short = GenerationParams(max_tokens=16)
    prompt = "Name the three primary colors, comma separated."
    for _ in range(2):   # warm-up: model load, worker start, first token_count
        p.generate(prompt, params=short)
        iso.generate(prompt, params=short)
        p.count_tokens("warm up")
    res: dict[str, list[float]] = {k: [] for k in ("inline_s", "isolated_s", "first_delta_s", "gen_tok_s",
                                                   "prompt_tok_s", "token_count_s")}
    for t in range(args.trials):
        a = p.generate(prompt, params=short)
        b = iso.generate(prompt, params=short)
        res["inline_s"].append(a.latency_s)
        res["isolated_s"].append(b.latency_s)
        rec.row(trial=t, measure="short_generate", inline_s=a.latency_s, isolated_s=b.latency_s,
                output=a.text, same_output=a.text == b.text, feature_flags={})
        # streamed first delta and generation rate (one long reply)
        t0 = time.perf_counter()
        first = last = None
        parts = []
        for chunk in p.generate_stream("Write a detailed description, about 300 words, of how a river forms "
                                       f"and changes over time. Variation {t}.", params=GenerationParams(max_tokens=420)):
            now = time.perf_counter()
            first = first if first is not None else now
            last = now
            parts.append(chunk)
        text = "".join(parts)
        n_out = p.count_tokens(text)
        rate = (n_out - 1) / (last - first) if last and first and last > first else None
        res["first_delta_s"].append(first - t0)
        if rate:
            res["gen_tok_s"].append(rate)
        rec.row(trial=t, measure="stream", time_to_first_token=first - t0, end_to_end_latency=last - t0,
                output_tokens=n_out, tokens_per_second=rate, auxiliary_tokens=n_out)
        # prompt processing: ~6k tokens, two output tokens, fresh text each trial
        long_prompt = filler(4_300, seed=1000 + t) + "\nReply with one word: which word is first in the text?"
        n_in = p.count_tokens(long_prompt)
        g = p.generate(long_prompt, params=GenerationParams(max_tokens=2))
        res["prompt_tok_s"].append(n_in / g.latency_s)
        rec.row(trial=t, measure="prompt", prompt_tokens=n_in, prefill_duration=g.latency_s,
                tokens_per_second=n_in / g.latency_s, note="latency includes the context pre-check's token count")
        # warm token_count
        for _ in range(3):
            t1 = time.perf_counter()
            p.count_tokens(filler(50, seed=t))
            res["token_count_s"].append(time.perf_counter() - t1)
    iso.close()
    p.close()
    summary = {k: spread(v) for k, v in res.items()}
    summary["isolation_overhead_s"] = statistics.median(res["isolated_s"]) - statistics.median(res["inline_s"])
    rec.row(measure="summary", summary=summary)
    print(json.dumps(summary, indent=1))


# ======================================================================== classify (AF-7, AF-8)
def _route_score(rows: list[dict], key: str) -> dict:
    n = len(rows)
    def eff(r):
        return "target" if r[key] == "uncertain" else r[key]
    correct = sum(eff(r) == r["gold"] for r in rows)
    gold_t = [r for r in rows if r["gold"] == "target"]
    false_routing = sum(eff(r) != "target" for r in gold_t)
    missed = sum(eff(r) == "target" for r in rows if r["gold"] != "target")
    confusion: dict[str, dict[str, int]] = {}
    for r in rows:
        confusion.setdefault(r["gold"], {}).setdefault(r[key], 0)
        confusion[r["gold"]][r[key]] += 1
    return {"accuracy": correct / n, "false_routing_rate": false_routing / len(gold_t),
            "false_routing": false_routing, "gold_target": len(gold_t), "missed_offload": missed,
            "confusion": confusion}


def arm_classify(args) -> None:
    path = FIX / "classify.jsonl"
    rec = Recorder("classify", "W1:apple-fm-classify", [path])
    items = [json.loads(line) for line in path.read_text().splitlines()]
    p = provider()
    p.generate("hi", params=GenerationParams(max_tokens=4))
    rows = []
    for it in items:
        req = ClassifyRequest(it["text"], it["has_tools"])
        h = heuristic_classify(req)
        m = classify(req, p)
        row = {"id": it["id"], "gold": it["route"], "heuristic": h.route, "model": m.route, "source": m.source,
               "fallback": m.fallback_reason, "latency_s": m.latency_s, "task_type": m.task_type,
               "complexity": m.complexity}
        rows.append(row)
        rec.row(measure="item", auxiliary_latency=m.latency_s, errors=m.fallback_reason,
                feature_flags={"classification": True}, **row)
    p.close()
    model_calls = [r for r in rows if r["heuristic"] != "deterministic"]
    summary = {"model": _route_score(rows, "model"), "heuristic": _route_score(rows, "heuristic"),
               "latency_model_calls": spread([r["latency_s"] for r in model_calls]),
               "failure_rate": sum(r["fallback"] is not None for r in model_calls) / max(1, len(model_calls))}
    rec.row(measure="summary", summary=summary)
    print(json.dumps(summary, indent=1))


# ======================================================================== extract (AF-9, AF-10)
def _field_ok(field: dict, values: dict, missing: list) -> tuple[bool, bool]:
    """(correct, invented): an absent field is correct when reported missing; invented = absent but returned."""
    name, gold = field["name"], field["gold"]
    if gold is None:
        return name in missing, name in values
    if name not in values:
        return False, False
    v = values[name]
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        try:
            return abs(float(v) - float(gold)) < 1e-6, False
        except (TypeError, ValueError):
            return False, False
    return str(v).strip().lower() == str(gold).strip().lower(), False


def arm_extract(args) -> None:
    path = FIX / "extract.jsonl"
    rec = Recorder("extract", "W1:apple-fm-extract", [path])
    items = [json.loads(line) for line in path.read_text().splitlines()]
    p = provider()
    totals = {"model": [0, 0, 0], "heuristic": [0, 0, 0]}   # correct, invented, fields
    lat = []
    for it in items:
        req = ExtractRequest(it["text"], tuple(FieldSpec(f["name"], f["type"]) for f in it["fields"]))
        m = extract(req, p)
        h = heuristic_extract(req)
        lat.append(m.latency_s)
        for kind, r in (("model", m), ("heuristic", h)):
            for f in it["fields"]:
                ok, inv = _field_ok(f, r.values, r.missing)
                totals[kind][0] += ok
                totals[kind][1] += inv
                totals[kind][2] += 1
        rec.row(measure="item", id=it["id"], auxiliary_latency=m.latency_s, model_values=m.values,
                model_ungrounded=m.ungrounded, heuristic_values=h.values, errors=m.fallback_reason,
                feature_flags={"structured_extraction": True})
    p.close()
    summary = {k: {"accuracy": v[0] / v[2], "invented_accepted": v[1], "fields": v[2]} for k, v in totals.items()}
    summary["latency"] = spread(lat)
    rec.row(measure="summary", summary=summary)
    print(json.dumps(summary, indent=1))


# ================================================================= compress (AF-11..AF-15)
def target_counter():
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(DEEPSEEK_TOKENIZER))
        return (lambda s: len(tok.encode(s, add_special_tokens=False).ids)), "deepseek-v4.1"
    except Exception as exc:
        print(f"DeepSeek tokenizer unavailable ({exc}); using chars/4", file=sys.stderr)
        return None, "approx"


def _retention(text: str, gold: dict) -> dict:
    facts = [bool(re.search(f, text)) for f in gold["facts"]]
    cons = [c in text for c in gold["constraints"]]
    return {"fact_retention": sum(facts) / len(facts), "facts_missing": [f for f, ok in zip(gold["facts"], facts, strict=True) if not ok],
            "constraint_retention": (sum(cons) / len(cons)) if cons else None}


def arm_compress(args) -> None:
    gold_all = json.loads((FIX / "compress" / "gold.json").read_text())
    docs = [FIX / "compress" / n for n in gold_all]
    rec = Recorder("compress", "W3:apple-fm-compress", docs + [FIX / "compress" / "gold.json"])
    count, tok_name = target_counter()
    p = provider()
    p.generate("hi", params=GenerationParams(max_tokens=4))
    summ = []
    for doc in docs:
        text = doc.read_text()
        gold = gold_all[doc.name]
        n_aux = p.count_tokens(text)
        for trial in range(args.trials):
            m = compress(text, p, target_tokens=count, min_tokens=300, max_ratio=1.0, max_ungrounded=1.0,
                         chunk_tokens=args.chunk_tokens)
            ret = _retention(m.text, gold)
            saved = (m.original_tokens - m.compressed_tokens) / DEEPSEEK_PREFILL_TOK_S
            r = rec.row(measure="model", doc=doc.name, trial=trial, target_tokenizer=tok_name,
                        prompt_tokens=m.original_tokens, aux_prompt_tokens=n_aux, compressed_tokens=m.compressed_tokens,
                        ratio=m.ratio, auxiliary_latency=m.latency_s, s_per_1k_input=m.latency_s / n_aux * 1000,
                        facts=len(m.facts), ungrounded=len(m.ungrounded_facts), ungrounded_share=m.ungrounded_share,
                        chunks=m.chunks, failed_chunks=m.failed_chunks,
                        accepted_at_defaults=(m.ratio <= 0.6 and m.ungrounded_share <= 0.1 and m.failed_chunks == 0),
                        saved_deepseek_prefill_s=saved, net_s_derived=saved - m.latency_s,
                        net_s_per_1k_derived=(saved - m.latency_s) / m.original_tokens * 1000,
                        feature_flags={"context_compression": True}, **ret,
                        output=m.text if trial == 0 else None,
                        ungrounded_examples=[asdict(f) for f in m.ungrounded_facts[:5]] if trial == 0 else None)
            summ.append(r)
        h = compress(text, None, target_tokens=count, min_tokens=300, max_ratio=1.0)
        hr = _retention(h.text, gold)
        rec.row(measure="heuristic", doc=doc.name, ratio=h.ratio, prompt_tokens=h.original_tokens,
                compressed_tokens=h.compressed_tokens, **hr)
        print(f"{doc.name}: heuristic ratio {h.ratio:.2f} facts {hr['fact_retention']:.2f} "
              f"constraints {hr['constraint_retention']}")
    p.close()
    by_doc: dict[str, list[dict]] = {}
    for r in summ:
        by_doc.setdefault(r["doc"], []).append(r)
    out = {d: {"ratio": spread([r["ratio"] for r in rs]), "fact_retention": spread([r["fact_retention"] for r in rs]),
               "constraint_retention": rs[0]["constraint_retention"], "s_per_1k": spread([r["s_per_1k_input"] for r in rs]),
               "net_s_per_1k_derived": spread([r["net_s_per_1k_derived"] for r in rs]),
               "ungrounded_share": spread([r["ungrounded_share"] for r in rs]),
               "facts_missing_trial0": rs[0]["facts_missing"]} for d, rs in by_doc.items()}
    rec.row(measure="summary", summary=out)
    print(json.dumps(out, indent=1))


# ============================================================================ memory (AF-16)
def _vm() -> dict[str, float]:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    vals = {k.strip().strip('"'): int(v.strip().rstrip(".")) for k, v in
            (line.split(":", 1) for line in out.splitlines()[1:] if ":" in line) if v.strip().rstrip(".").isdigit()}
    g = lambda k: vals.get(k, 0) * page / rm.GiB  # noqa: E731
    return {"wired": g("Pages wired down"), "active": g("Pages active"), "compressed": g("Pages occupied by compressor"),
            "free": g("Pages free"), "anonymous": g("Anonymous pages"), "file_backed": g("File-backed pages")}


def _service_rss_gib() -> dict[str, float]:
    out = subprocess.run(["ps", "-axo", "rss,comm"], capture_output=True, text=True).stdout
    tot: dict[str, float] = {}
    for line in out.splitlines()[1:]:
        rss, _, comm = line.strip().partition(" ")
        name = comm.rsplit("/", 1)[-1]
        if any(k in name for k in ("TGOnDeviceInference", "modelmanagerd", "aned", "GenerativeExperiences")):
            tot[name] = tot.get(name, 0.0) + int(rss) / 1024 / 1024
    return tot


def _footprint(name: str) -> str | None:
    try:
        out = subprocess.run(["footprint", name], capture_output=True, text=True, timeout=20)
        m = re.search(r"Footprint:\s*([\d.]+\s*[KMG]B)", out.stdout)
        return m.group(1) if m else (out.stderr.strip()[:120] or None)
    except Exception as exc:
        return str(exc)[:120]


def arm_memory(args) -> None:
    rec = Recorder("memory", "W1:apple-fm-memory", [])
    idle = {"vm": _vm(), "rss": _service_rss_gib(), "footprint": _footprint("TGOnDeviceInferenceProviderService")}
    rec.row(measure="before", **idle)
    p = provider()
    samples = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            samples.append({"t": time.time(), "vm": _vm(), "rss": _service_rss_gib()})
            time.sleep(0.5)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    for i in range(args.trials):
        p.generate(f"Write about 250 words on the history of clocks. Variation {i}.",
                   params=GenerationParams(max_tokens=350))
    during_fp = _footprint("TGOnDeviceInferenceProviderService")
    stop.set()
    th.join()
    p.close()
    time.sleep(args.after_s)
    after = {"vm": _vm(), "rss": _service_rss_gib(), "footprint": _footprint("TGOnDeviceInferenceProviderService")}
    peak_used = max(s["vm"]["wired"] + s["vm"]["active"] + s["vm"]["compressed"] for s in samples)
    base_used = idle["vm"]["wired"] + idle["vm"]["active"] + idle["vm"]["compressed"]
    peak_rss = max(sum(s["rss"].values()) for s in samples)
    summary = {"used_gib_before": base_used, "used_gib_peak": peak_used, "used_delta_gib": peak_used - base_used,
               "service_rss_gib_before": sum(idle["rss"].values()), "service_rss_gib_peak": peak_rss,
               "footprint_before": idle["footprint"], "footprint_during": during_fp, "footprint_after": after["footprint"],
               "samples": len(samples), "note": "used = wired + active + compressed (vm_stat), whole machine"}
    rec.row(measure="summary", summary=summary, after=after)
    print(json.dumps(summary, indent=1))


# ======================================================================== contention (AF-17..AF-19)
GPU_LOOP = r"""
import json, sys, time
import mlx.core as mx
kind, start, dur = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
mx.set_cache_limit(0)
if kind == "gemv":
    W = mx.random.normal((16384, 32768)).astype(mx.float16); x = mx.random.normal((1, 16384)).astype(mx.float16)
    reps, work = 8, 8 * W.nbytes
    def step():
        return [x @ W for _ in range(reps)]
else:
    A = mx.random.normal((4096, 4096)).astype(mx.float16); B = mx.random.normal((4096, 4096)).astype(mx.float16)
    reps, work = 8, 8 * 2 * 4096 ** 3
    def step():
        return [A @ B for _ in range(reps)]
mx.eval(step())
while time.time() < start:
    mx.eval(step())
vals = []
while time.time() < start + dur:
    t = time.perf_counter(); mx.eval(step()); vals.append(work / (time.perf_counter() - t))
print(json.dumps({"kind": kind, "n": len(vals), "rates": vals}))
"""

FM_LOOP = r"""
import json, sys, time
sys.path.insert(0, sys.argv[4])
from cachalot.auxiliary.apple_fm import AppleFoundationModelsProvider
from cachalot.auxiliary.provider import GenerationParams
start, dur = float(sys.argv[1]), float(sys.argv[2])
p = AppleFoundationModelsProvider(timeout_s=60)
p.generate("hi", params=GenerationParams(max_tokens=4))
while time.time() < start:
    time.sleep(0.01)
rates, i = [], 0
while time.time() < start + dur:
    first = last = None; parts = []
    for c in p.generate_stream(f"Write about 150 words on lighthouses. Variation {i}.", params=GenerationParams(max_tokens=200)):
        now = time.perf_counter(); first = first or now; last = now; parts.append(c)
    n = p.count_tokens("".join(parts))
    if last and first and last > first:
        rates.append((n - 1) / (last - first))
    i += 1
p.close()
print(json.dumps({"kind": "fm", "n": len(rates), "rates": rates}))
"""


def _launch(code: str, argv: list[str]) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code, *argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def arm_contention(args) -> None:
    rec = Recorder("contention", "W7:apple-fm-contention", [])
    order = ["gpu", "fm", "both", "both", "fm", "gpu"]
    results: dict[str, dict[str, list[float]]] = {}
    for kind in args.kinds:
        for rep, cell in enumerate(order):
            start = time.time() + args.warmup_s
            procs = {}
            if cell in ("gpu", "both"):
                procs["gpu"] = _launch(GPU_LOOP, [kind, str(start), str(args.seconds)])
            if cell in ("fm", "both"):
                procs["fm"] = _launch(FM_LOOP, [str(start), str(args.seconds), "", str(REPO / "src")])
            for who, proc in procs.items():
                out, err = proc.communicate(timeout=args.seconds + args.warmup_s + 120)
                if proc.returncode != 0:
                    sys.exit(f"{who} loop failed: {err[-400:]}")
                data = json.loads(out.strip().splitlines()[-1])
                med = statistics.median(data["rates"]) if data["rates"] else None
                key = f"{kind}:{cell}:{who}"
                results.setdefault(key, {"medians": []})["medians"].append(med)
                rec.row(measure="cell", gpu_kind=kind, cell=cell, rep=rep, who=who, n=data["n"], median_rate=med,
                        rate_unit="tok/s" if who == "fm" else ("B/s" if kind == "gemv" else "FLOP/s"),
                        spread=spread(data["rates"]), pressure=rm._sysctl("kern.memorystatus_vm_pressure_level"))
            print(f"{kind} {cell} rep {rep}: " + ", ".join(f"{k}={v['medians'][-1]:.4g}" for k, v in results.items()
                                                           if k.startswith(f'{kind}:{cell}:')), flush=True)
            time.sleep(args.gap_s)
    summary = {}
    for kind in args.kinds:
        gpu_alone = statistics.mean(results[f"{kind}:gpu:gpu"]["medians"])
        gpu_both = statistics.mean(results[f"{kind}:both:gpu"]["medians"])
        fm_alone = statistics.mean(results[f"{kind}:fm:fm"]["medians"])
        fm_both = statistics.mean(results[f"{kind}:both:fm"]["medians"])
        summary[kind] = {"gpu_alone": gpu_alone, "gpu_beside_fm": gpu_both, "gpu_slowdown": 1 - gpu_both / gpu_alone,
                         "fm_alone_tok_s": fm_alone, "fm_beside_gpu_tok_s": fm_both, "fm_slowdown": 1 - fm_both / fm_alone,
                         "cells": {k: v["medians"] for k, v in results.items() if k.startswith(kind)}}
    rec.row(measure="summary", summary=summary)
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="arm", required=True)
    s = sub.add_parser("overhead")
    s.add_argument("--trials", type=int, default=7)
    sub.add_parser("classify")
    sub.add_parser("extract")
    s = sub.add_parser("compress")
    s.add_argument("--trials", type=int, default=3)
    s.add_argument("--chunk-tokens", type=int, default=1_500)
    s = sub.add_parser("memory")
    s.add_argument("--trials", type=int, default=3)
    s.add_argument("--after-s", type=float, default=10.0)
    s = sub.add_parser("contention")
    s.add_argument("--kinds", nargs="+", default=["gemv", "gemm"])
    s.add_argument("--seconds", type=float, default=20.0)
    s.add_argument("--warmup-s", type=float, default=8.0)
    s.add_argument("--gap-s", type=float, default=3.0)
    args = ap.parse_args()
    out = subprocess.run(["pgrep", "-fl", r"deepseek-v41/bin/python .*cachalot\.cli (serve|chat)|serve(-glm|-minimax)?\.sh"],
                         capture_output=True, text=True).stdout.strip()
    if out:
        sys.exit(f"a Cachalot runtime is running; benchmarks need a quiet machine:\n{out}")
    {"overhead": arm_overhead, "classify": arm_classify, "extract": arm_extract, "compress": arm_compress,
     "memory": arm_memory, "contention": arm_contention}[args.arm](args)


if __name__ == "__main__":
    try:
        main()
    except AuxError as exc:
        sys.exit(f"auxiliary {exc.kind}: {exc}")
