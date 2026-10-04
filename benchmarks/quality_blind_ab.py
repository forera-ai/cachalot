"""
Blind agent-quality A/B of the decode miss budget (HANDOFF 18.60; the recipe of 18.29 and MEASUREMENT.md item 6).

The six request bodies of a dumped Hermes session (`/tmp/cachalot-requests-0.46b.jsonl`: a greeting, a tool call, a tool
result to summarise, a story, a C# snippet, the same in TypeScript) are each sampled N times per arm at the dump's
temperature through `Engine.chat`, the server's code path. The arms (exact, and the miss budget under test) are interleaved
in one process: sample j of a body uses the same seed in both arms, and the order within a pair alternates with j.

    run    sample everything; writes samples.jsonl (arm labels inside, never shown to the grader)
    sheet  mechanical checks (tool-call validity, loops, truncation, C# `dotnet build`) and a shuffled, arm-free
           grading sheet with a separate key file
    score  joins hand grades (grades.json: {"R017": {"verdict": "ok" | "flawed", "flaw": "..."}}) with the key and
           reports flawed rates per arm and per body with an exact (Fisher) test

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/quality_blind_ab.py run --budget 0 --out DIR --n 8
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/quality_blind_ab.py sheet DIR
    (write DIR/grades.json from sheet.md without opening key.json)
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/quality_blind_ab.py score DIR
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))

DUMP = "/tmp/cachalot-requests-0.46b.jsonl"
BODY_ROWS = (0, 2, 4, 6, 8, 10)
MAX_TOKENS = 700


def load_bodies() -> list[dict]:
    rows = [json.loads(line) for line in Path(DUMP).read_text().splitlines()]
    return [rows[i]["body"] for i in BODY_ROWS]


def run(budget: int, n: int, out: Path) -> None:
    import pareto

    pareto.serve_defaults()
    sys.path.insert(0, str(REPO / "src"))
    from cachalot.model.api import V41Model
    from cachalot.model.generation import SamplingParams
    from cachalot.server.engine import ChatRequest, Engine

    out.mkdir(parents=True, exist_ok=True)
    model = V41Model.from_pretrained(os.environ["CACHALOT_MODEL_PATH"], max_seq_len=65536,
                                     expert_cache_budget_bytes=int(48 * 2**30))
    engine = Engine(model)
    log = out / "samples.jsonl"
    with log.open("a") as fh:
        for bi, body in enumerate(load_bodies()):
            for j in range(n):
                order = ("exact", "capped") if j % 2 == 0 else ("capped", "exact")
                for arm in order:
                    model.set_decode_miss_budget(budget if arm == "capped" else None)
                    skipped0 = model.runtime.expert_store.skipped_experts
                    req = ChatRequest(messages=body["messages"], tools=body["tools"], thinking_mode="chat",
                                      params=SamplingParams(max_new_tokens=MAX_TOKENS, temperature=body["temperature"],
                                                            seed=1000 * bi + j))
                    res = engine.chat(req)
                    row = {"body": bi, "j": j, "arm": arm, "budget": budget if arm == "capped" else None,
                           "tokens": res.completion_tokens, "finish": res.finish_reason,
                           "ms_tok": round(1000 * res.decode_seconds / max(res.completion_tokens, 1), 1),
                           "skipped": model.runtime.expert_store.skipped_experts - skipped0,
                           "text": res.raw_text, "tool_calls": res.message.get("tool_calls")}
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print({k: v for k, v in row.items() if k not in ("text", "tool_calls")}, flush=True)
    model.close()


# ---------------------------------------------------------------------------------------------------------------
# mechanical checks (pure; tested)
# ---------------------------------------------------------------------------------------------------------------

def tool_call_problems(tool_calls: list[dict] | None, tools: list[dict]) -> list[str]:
    """Empty list = every call names a declared tool, has JSON arguments and all required parameters."""
    if not tool_calls:
        return ["no tool call"]
    declared = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}
    problems = []
    for call in tool_calls:
        fn = call.get("function", {})
        name = fn.get("name")
        if name not in declared:
            problems.append(f"unknown tool {name!r}")
            continue
        raw = fn.get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except ValueError:
            problems.append(f"arguments of {name} are not JSON")
            continue
        for key in declared[name].get("required", []):
            if key not in args:
                problems.append(f"{name} lacks required {key!r}")
    return problems


def repetition_share(text: str, n: int = 5) -> float:
    """Share of word n-grams that repeat an earlier n-gram: near 0 for prose, high for a loop."""
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def fisher_two_sided(a_bad: int, a_n: int, b_bad: int, b_n: int) -> float:
    """Two-sided Fisher exact p for two proportions (sum of tables no more likely than the observed)."""
    bad, total = a_bad + b_bad, a_n + b_n

    def pmf(k: int) -> float:
        return math.comb(bad, k) * math.comb(total - bad, a_n - k) / math.comb(total, a_n)

    lo, hi = max(0, a_n - (total - bad)), min(a_n, bad)
    p_obs = pmf(a_bad)
    return min(1.0, sum(pmf(k) for k in range(lo, hi + 1) if pmf(k) <= p_obs * (1 + 1e-9)))


def _context(body: dict) -> str:
    last = body["messages"][-1]
    text = last.get("content") if isinstance(last.get("content"), str) else json.dumps(last.get("content"))
    return f"[{last['role']}] {text[:5000]}"


def sheet(out: Path) -> None:
    import pareto

    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    bodies = load_bodies()
    rng = random.Random(20261004)
    order = list(range(len(samples)))
    rng.shuffle(order)
    key, mech, lines = {}, {}, []
    by_body: dict[int, list[int]] = {}
    for idx in order:
        by_body.setdefault(samples[idx]["body"], []).append(idx)
    n = 0
    for bi in sorted(by_body):
        lines.append(f"\n\n# Body {bi}\n\nLast message in the context:\n\n```\n{_context(bodies[bi])}\n```\n")
        for idx in by_body[bi]:
            n += 1
            sid = f"R{n:03d}"
            s = samples[idx]
            key[sid] = {"arm": s["arm"], "body": bi, "j": s["j"]}
            problems = tool_call_problems(s["tool_calls"], bodies[bi]["tools"]) if bi == 1 else None
            m = {"finish": s["finish"], "tokens": s["tokens"], "repetition": round(repetition_share(s["text"]), 3),
                 "tool_problems": problems}
            if bi == 4:
                m["csharp_builds"] = bool(pareto.check_csharp(s["text"], {}))
            mech[sid] = m
            shown = s["text"] if s["text"].strip() else "(no text)"
            if s["tool_calls"]:
                shown += "\n[tool_calls] " + json.dumps(s["tool_calls"])[:1500]
            lines.append(f"\n## {sid}\n\n```\n{shown[:3500]}\n```\n")
    (out / "sheet.md").write_text("# Blind grading sheet\n\nGrade each reply ok / flawed (flaw: invented, wrong-action,"
                                  " derail, loop, truncated, malformed-tool, wrong-code).\n" + "".join(lines))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "mech.json").write_text(json.dumps(mech, indent=1))
    print(f"{n} replies in sheet.md; key.json and mech.json hold the arms and the mechanical checks")


def score(out: Path) -> None:
    key = json.loads((out / "key.json").read_text())
    mech = json.loads((out / "mech.json").read_text())
    grades = json.loads((out / "grades.json").read_text())
    missing = sorted(set(key) - set(grades))
    if missing:
        sys.exit(f"ungraded: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    tally: dict[tuple[str, int], list[int]] = {}
    for sid, k in key.items():
        t = tally.setdefault((k["arm"], k["body"]), [0, 0])
        t[1] += 1
        t[0] += grades[sid]["verdict"] == "flawed"
    print("| body | exact flawed | capped flawed | Fisher p |")
    print("|---|---|---|---|")
    tot = {"exact": [0, 0], "capped": [0, 0]}
    for bi in sorted({b for _, b in tally}):
        e, c = tally[("exact", bi)], tally[("capped", bi)]
        print(f"| {bi} | {e[0]}/{e[1]} | {c[0]}/{c[1]} | {fisher_two_sided(e[0], e[1], c[0], c[1]):.3f} |")
        for arm, t in (("exact", e), ("capped", c)):
            tot[arm][0] += t[0]
            tot[arm][1] += t[1]
    e, c = tot["exact"], tot["capped"]
    print(f"| all | {e[0]}/{e[1]} | {c[0]}/{c[1]} | {fisher_two_sided(e[0], e[1], c[0], c[1]):.3f} |")
    for arm in ("exact", "capped"):
        ids = [s for s, k in key.items() if k["arm"] == arm]
        loops = sum(mech[s]["repetition"] > 0.3 for s in ids)
        trunc = sum(mech[s]["finish"] == "length" for s in ids)
        tool = [mech[s]["tool_problems"] for s in ids if mech[s]["tool_problems"] is not None]
        cs = [mech[s]["csharp_builds"] for s in ids if "csharp_builds" in mech[s]]
        print(f"{arm}: loops {loops}, truncated {trunc}, valid tool calls {sum(not p for p in tool)}/{len(tool)}, "
              f"C# builds {sum(cs)}/{len(cs)}")
    flaws: dict[str, dict[str, int]] = {"exact": {}, "capped": {}}
    for sid, k in key.items():
        if grades[sid]["verdict"] == "flawed":
            f = grades[sid].get("flaw", "unspecified")
            flaws[k["arm"]][f] = flaws[k["arm"]].get(f, 0) + 1
    print("flaws:", json.dumps(flaws))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--budget", type=int, default=0)
    r.add_argument("--n", type=int, default=8)
    r.add_argument("--out", required=True)
    for name in ("sheet", "score"):
        sub.add_parser(name).add_argument("out")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.budget, args.n, Path(args.out))
    else:
        {"sheet": sheet, "score": score}[args.cmd](Path(args.out))


if __name__ == "__main__":
    main()
