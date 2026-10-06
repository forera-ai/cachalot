"""
Blind agent-quality A/B of the decode miss budget (HANDOFF 18.60; the recipe of 18.29 and MEASUREMENT.md item 6).

The six request bodies of a dumped Hermes session (`/tmp/cachalot-requests-0.46b.jsonl`: a greeting, a tool call, a tool
result to summarise, a story, a C# snippet, the same in TypeScript) are each sampled N times per arm at the dump's
temperature through `Engine.chat`, the server's code path. The arms (exact, and the miss budget under test) are interleaved
in one process: sample j of a body uses the same seed in every arm, and the order of the arms rotates with j.
`--budget 1,0` runs three arms (exact, b1, b0) in one process; a single budget keeps the arm names exact / capped.

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

# Defaults are the 0.46b dump of HANDOFF 18.60; HANDOFF 18.67 ran the 0.58.1 dump:
#   CACHALOT_QB_DUMP=/tmp/cachalot-requests-0.58.1.jsonl CACHALOT_QB_ROWS=0,2,4,6,8,14 CACHALOT_QB_MAX_TOKENS=1500
# (the same six kinds of body: greeting, tool call, tool result, story, C#, TypeScript; row 14's earlier image is
# replaced by a text note). Set the same variables for `run` and `sheet`.
DUMP = os.environ.get("CACHALOT_QB_DUMP", "/tmp/cachalot-requests-0.46b.jsonl")
BODY_ROWS = tuple(int(x) for x in os.environ.get("CACHALOT_QB_ROWS", "0,2,4,6,8,10").split(","))
MAX_TOKENS = int(os.environ.get("CACHALOT_QB_MAX_TOKENS", "700"))


def _drop_images(body: dict) -> dict:
    """The same body with image parts turned into a text note (a text-only run; both arms see the same prompt)."""
    messages = []
    for m in body["messages"]:
        c = m.get("content")
        if isinstance(c, list):
            parts = [p if p.get("type") == "text" else {"type": "text", "text": "[image omitted]"} for p in c]
            m = dict(m, content=parts)
        messages.append(m)
    return dict(body, messages=messages)


def load_bodies() -> list[dict]:
    rows = [json.loads(line) for line in Path(DUMP).read_text().splitlines()]
    return [_drop_images(rows[i]["body"]) for i in BODY_ROWS]


def arm_names(budgets: list[int]) -> dict[str, int | None]:
    """Arm name -> per-layer decode miss budget (None = exact). One budget keeps the historical name "capped"."""
    if len(budgets) == 1:
        return {"exact": None, "capped": budgets[0]}
    return {"exact": None, **{f"b{b}": b for b in budgets}}


def arm_order(arms: list[str], j: int) -> list[str]:
    """Arms rotated by j, so every arm runs first equally often (two arms: the old even/odd swap)."""
    k = j % len(arms)
    return arms[k:] + arms[:k]


def run(budgets: list[int], n: int, out: Path) -> None:
    import pareto

    pareto.serve_defaults()
    sys.path.insert(0, str(REPO / "src"))
    from cachalot.model.api import V41Model
    from cachalot.model.generation import SamplingParams
    from cachalot.server.engine import ChatRequest, Engine

    import run_manifest

    out.mkdir(parents=True, exist_ok=True)
    run_manifest.write_manifest(out, run_manifest.collect(
        workload="W6:quality-blind", workload_text=json.dumps(load_bodies(), sort_keys=True), budget_gib=48,
        instrument="quality_blind_ab run", params={"budgets": budgets, "n": n, "max_tokens": MAX_TOKENS,
                                                   "dump": DUMP, "rows": BODY_ROWS}))
    model = V41Model.from_pretrained(os.environ["CACHALOT_MODEL_PATH"], max_seq_len=65536,
                                     expert_cache_budget_bytes=int(48 * 2**30))
    engine = Engine(model)
    arms = arm_names(budgets)
    log = out / "samples.jsonl"
    # resume: a sample already in the log (same body, j, arm) is not generated again; seeds depend on (body, j) only,
    # so a resumed run equals an uninterrupted one. CACHALOT_QB_BODIES=4,5 limits the bodies of this call.
    done = {(r["body"], r["j"], r["arm"]) for r in map(json.loads, log.read_text().splitlines())} if log.exists() else set()
    only = {int(x) for x in os.environ["CACHALOT_QB_BODIES"].split(",")} if os.environ.get("CACHALOT_QB_BODIES") else None
    with log.open("a") as fh:
        for bi, body in enumerate(load_bodies()):
            if only is not None and bi not in only:
                continue
            for j in range(n):
                for arm in arm_order(list(arms), j):
                    if (bi, j, arm) in done:
                        continue
                    model.set_decode_miss_budget(arms[arm])
                    skipped0 = model.runtime.expert_store.skipped_experts
                    req = ChatRequest(messages=body["messages"], tools=body["tools"], thinking_mode="chat",
                                      params=SamplingParams(max_new_tokens=MAX_TOKENS, temperature=body["temperature"],
                                                            seed=1000 * bi + j))
                    res = engine.chat(req)
                    row = {"body": bi, "j": j, "arm": arm, "budget": arms[arm],
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
            if "C#" in _context(bodies[bi]):
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
    arms = ["exact"] + sorted({a for a, _ in tally} - {"exact"})
    print("| body | " + " | ".join(f"{a} flawed" for a in arms) + " | " + " | ".join(f"p {a}" for a in arms[1:]) + " |")
    print("|---" * (2 * len(arms)) + "|")
    tot = {a: [0, 0] for a in arms}
    for bi in sorted({b for _, b in tally}):
        cells = {a: tally.get((a, bi), [0, 0]) for a in arms}
        e = cells["exact"]
        ps = [f"{fisher_two_sided(e[0], e[1], c[0], c[1]):.3f}" for c in (cells[a] for a in arms[1:])]
        print(f"| {bi} | " + " | ".join(f"{c[0]}/{c[1]}" for c in cells.values()) + " | " + " | ".join(ps) + " |")
        for a, t in cells.items():
            tot[a][0] += t[0]
            tot[a][1] += t[1]
    e = tot["exact"]
    ps = [f"{fisher_two_sided(e[0], e[1], tot[a][0], tot[a][1]):.3f}" for a in arms[1:]]
    print("| all | " + " | ".join(f"{tot[a][0]}/{tot[a][1]}" for a in arms) + " | " + " | ".join(ps) + " |")
    for arm in arms:
        ids = [s for s, k in key.items() if k["arm"] == arm]
        loops = sum(mech[s]["repetition"] > 0.3 for s in ids)
        trunc = sum(mech[s]["finish"] == "length" for s in ids)
        tool = [mech[s]["tool_problems"] for s in ids if mech[s]["tool_problems"] is not None]
        cs = [mech[s]["csharp_builds"] for s in ids if "csharp_builds" in mech[s]]
        print(f"{arm}: loops {loops}, truncated {trunc}, valid tool calls {sum(not p for p in tool)}/{len(tool)}, "
              f"C# builds {sum(cs)}/{len(cs)}")
    flaws: dict[str, dict[str, int]] = {a: {} for a in arms}
    for sid, k in key.items():
        if grades[sid]["verdict"] == "flawed":
            f = grades[sid].get("flaw", "unspecified")
            flaws[k["arm"]][f] = flaws[k["arm"]].get(f, 0) + 1
    print("flaws:", json.dumps(flaws))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--budget", default="0", help="one budget, or a comma list for several capped arms (1,0)")
    r.add_argument("--n", type=int, default=8)
    r.add_argument("--out", required=True)
    for name in ("sheet", "score"):
        sub.add_parser(name).add_argument("out")
    args = ap.parse_args()
    if args.cmd == "run":
        run([int(b) for b in args.budget.split(",")], args.n, Path(args.out))
    else:
        {"sheet": sheet, "score": score}[args.cmd](Path(args.out))


if __name__ == "__main__":
    main()
