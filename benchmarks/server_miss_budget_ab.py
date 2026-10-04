"""
Server-path A/B of the decode miss budget (HANDOFF 18.59): `Engine.chat`, the code `./serve.sh` runs, on a Hermes-shaped
22k-token context (a dumped body's system prompt and 25 tools), with the budget flipped every turn in one process.

Twelve user prompts, 220 greedy tokens each. Pass 1: odd prompts run with the budget, even ones exact. Pass 2: the parity
swapped, so every prompt runs both ways. Pass 3 (the A/A control): every prompt exact again, which measures what
alternating turns alone do to the step time. A first turn prefills the 22k context once; prompt 0 repeats it exactly
(an all-resident turn), so analysis starts at prompt 1.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/server_miss_budget_ab.py --budget 0 --out /tmp/ab-b0.json
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/server_miss_budget_ab.py --report /tmp/ab-b0.json

One runtime at a time (it loads the model); ~25 minutes at budget 0.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))

DUMP = "/tmp/cachalot-requests-0.46b.jsonl"
USER = [
    "List the files in the current directory and tell me which look like source code.",
    "Write a Python script that counts lines of code per file extension in a folder, then run it.",
    "Read the README and summarise what the project does in five bullet points.",
    "Find every TODO comment in the repository and group them by file.",
    "Write a unit test for a function that parses ISO dates, and explain the edge cases.",
    "Explain what this error means: ModuleNotFoundError: No module named 'yaml', and how to fix it.",
    "Create a small bash script that backs up my Documents folder to an external drive.",
    "Refactor a function with three nested loops into something readable; show before and after.",
    "Search the web for the latest stable Rust version and tell me what changed.",
    "Write a C# class for a thread-safe LRU cache with a capacity limit.",
    "Draft a commit message for a change that adds retry logic to an HTTP client.",
    "Explain git rebase versus merge and when each is safer on a shared branch.",
]


def run(budget: int, tokens: int, out: Path) -> None:
    import pareto

    pareto.serve_defaults()
    from cachalot.model.api import V41Model
    from cachalot.model.generation import SamplingParams
    from cachalot.server.engine import ChatRequest, Engine

    body = json.loads(Path(DUMP).read_text().splitlines()[0])["body"]
    system, tools = body["messages"][0], body["tools"]
    model = V41Model.from_pretrained(os.environ["CACHALOT_MODEL_PATH"], max_seq_len=65536,
                                     expert_cache_budget_bytes=int(48 * 2**30))
    engine = Engine(model)
    rows: list[dict] = []

    def turn(tag: str, i: int, cap: int | None) -> None:
        model.set_decode_miss_budget(cap)
        skipped0 = model.runtime.expert_store.skipped_experts
        req = ChatRequest(messages=[system, {"role": "user", "content": USER[i]}], tools=tools,
                          params=SamplingParams(max_new_tokens=tokens, temperature=0.0))
        res = engine.chat(req)
        n = max(res.completion_tokens, 1)
        rows.append({"tag": tag, "prompt": i, "budget": cap, "tokens": res.completion_tokens,
                     "reused": res.reused_prefix_tokens, "prefill_s": round(res.prefill_seconds, 2),
                     "decode_s": round(res.decode_seconds, 2), "ms_tok": round(1000 * res.decode_seconds / n, 1),
                     "skipped_tok": round((model.runtime.expert_store.skipped_experts - skipped0) / n, 1),
                     "text": res.raw_text[:1500]})
        print({k: v for k, v in rows[-1].items() if k != "text"}, flush=True)
        out.write_text(json.dumps(rows, indent=1))

    turn("warm", 0, None)
    for i in range(12):
        turn("p1", i, budget if i % 2 else None)
    for i in range(12):
        turn("p2", i, None if i % 2 else budget)
    for i in range(12):
        turn("aa", i, None)
    model.close()


def report(path: Path) -> None:
    rows = json.loads(path.read_text())
    by = {tag: {r["prompt"]: r for r in rows if r["tag"] == tag} for tag in ("p1", "p2", "aa")}
    ids = range(1, 12)
    exact = {i: next(r for r in (by["p1"][i], by["p2"][i]) if r["budget"] is None) for i in ids}
    capped = {i: next(r for r in (by["p1"][i], by["p2"][i]) if r["budget"] is not None) for i in ids}
    e = np.array([exact[i]["ms_tok"] for i in ids])
    c = np.array([capped[i]["ms_tok"] for i in ids])
    a = np.array([by["aa"][i]["ms_tok"] for i in ids])
    d = c - e
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    print(f"prompts 1-11, {len(rows)} turns; ms a token: exact {e.mean():.1f}, capped {c.mean():.1f}, A/A control {a.mean():.1f}")
    print(f"capped - exact per prompt: mean {d.mean():.1f} (95 % bootstrap [{np.percentile(boot, 2.5):.1f}, "
          f"{np.percentile(boot, 97.5):.1f}]), ratio {c.sum() / e.sum():.3f}; control - exact: mean {(a - e).mean():.1f} sd {(a - e).std(ddof=1):.1f}")
    odd = [capped[i]["ms_tok"] / exact[i]["ms_tok"] for i in ids if by["p1"][i]["budget"] is not None]
    even = [capped[i]["ms_tok"] / exact[i]["ms_tok"] for i in ids if by["p2"][i]["budget"] is not None]
    print(f"ratio by parity: pass 1 {np.mean(odd):.3f} (n={len(odd)}), pass 2 {np.mean(even):.3f} (n={len(even)}); "
          f"experts dropped a token {np.mean([capped[i]['skipped_tok'] for i in ids]):.1f}; "
          f"replies that differ from exact: {sum(capped[i]['text'] != exact[i]['text'] for i in ids)} of 11")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=220)
    ap.add_argument("--out")
    ap.add_argument("--report")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    if args.report:
        report(Path(args.report))
    else:
        if not args.out:
            ap.error("--out is required to run")
        run(args.budget, args.tokens, Path(args.out))


if __name__ == "__main__":
    main()
