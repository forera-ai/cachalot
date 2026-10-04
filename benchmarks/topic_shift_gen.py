"""
Long replies that change topic inside one answer, exact against the decode miss budget (HANDOFF 18.61, part B).

`topic_shift.py` teacher-forces one text and measures the likelihood cost after a topic shift; this generates. Six prompts
each ask for three unrelated pieces in one reply (a story, then a tutorial, then a recipe, and so on), up to 1,400 tokens
at temperature 0.7. Each arm runs in its own fresh process over the same prompts and seeds (a shared process would let the
exact arm warm the cache for the budget arm), so the cache starts as an agent's would after earlier requests.

    run    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/topic_shift_gen.py run exact --out DIR
           PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/topic_shift_gen.py run 0 --out DIR
    sheet  ... topic_shift_gen.py sheet DIR      (a shuffled arm-free sheet, key.json, mechanical checks)
    score  ... topic_shift_gen.py score DIR      (after grades.json: {"R01": {"verdict": "ok"|"flawed", "flaw": "..."}})
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))

PROMPTS = [
    "Write a 300-word story about a lighthouse keeper. Then write a short Python tutorial on decorators, with code. "
    "Then give a recipe for lentil soup.",
    "First explain in detail how TCP congestion control works. Then, in a separate section, write a poem about autumn. "
    "Then summarise the causes of the First World War.",
    "Write a C# class for an LRU cache. Then write a short essay on the ethics of AI. "
    "Then give a SQL schema (DDL) for a library system.",
    "Explain photosynthesis to a ten-year-old. Then write a bash script that renames files by modification date. "
    "Then write a short dialogue between a customer and a barista.",
    "Describe the plot of Hamlet. Then give a step-by-step guide to changing a bicycle tyre. "
    "Then write a tutorial on regular expressions with examples.",
    "Write five haiku about the sea. Then explain Git branching with the commands. "
    "Then draft a polite email declining a meeting.",
]
MAX_TOKENS = 1400


def run(arm: str, out: Path) -> None:
    import pareto

    pareto.serve_defaults()
    sys.path.insert(0, str(REPO / "src"))
    from cachalot.model.api import V41Model
    from cachalot.model.generation import SamplingParams
    from cachalot.server.engine import ChatRequest, Engine

    out.mkdir(parents=True, exist_ok=True)
    model = V41Model.from_pretrained(os.environ["CACHALOT_MODEL_PATH"], max_seq_len=8192,
                                     expert_cache_budget_bytes=int(48 * 2**30))
    model.set_decode_miss_budget(None if arm == "exact" else int(arm))
    engine = Engine(model)
    rows = []
    for i, prompt in enumerate(PROMPTS):
        skipped0 = model.runtime.expert_store.skipped_experts
        req = ChatRequest(messages=[{"role": "user", "content": prompt}], thinking_mode="chat",
                          params=SamplingParams(max_new_tokens=MAX_TOKENS, temperature=0.7, seed=500 + i))
        res = engine.chat(req)
        rows.append({"prompt": i, "arm": arm, "tokens": res.completion_tokens, "finish": res.finish_reason,
                     "ms_tok": round(1000 * res.decode_seconds / max(res.completion_tokens, 1), 1),
                     "skipped_tok": round((model.runtime.expert_store.skipped_experts - skipped0)
                                          / max(res.completion_tokens, 1), 1),
                     "text": res.raw_text})
        print({k: v for k, v in rows[-1].items() if k != "text"}, flush=True)
        (out / f"gen-{arm}.json").write_text(json.dumps(rows, indent=1))
    model.close()


def sheet(out: Path) -> None:
    from quality_blind_ab import repetition_share

    rows = [r for arm in ("exact", "0") for r in json.loads((out / f"gen-{arm}.json").read_text())]
    order = list(range(len(rows)))
    random.Random(20261004).shuffle(order)
    key, mech, lines = {}, {}, []
    for n, idx in enumerate(order, 1):
        r = rows[idx]
        sid = f"R{n:02d}"
        key[sid] = {"arm": "exact" if r["arm"] == "exact" else "capped", "prompt": r["prompt"]}
        mech[sid] = {"tokens": r["tokens"], "finish": r["finish"], "repetition": round(repetition_share(r["text"]), 3)}
        lines.append(f"\n\n## {sid}\n\nPrompt: {PROMPTS[r['prompt']]}\n\n```\n{r['text']}\n```\n")
    (out / "sheet.md").write_text("# Blind grading sheet (long topic-shift replies)\n\nGrade each reply ok / flawed (flaw: "
                                  "derail, loop, skipped-section, garbled, wrong-code, incoherent, invented).\n"
                                  + "".join(lines))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "mech.json").write_text(json.dumps(mech, indent=1))
    print(f"{len(rows)} replies in sheet.md")


def score(out: Path) -> None:
    from quality_blind_ab import fisher_two_sided

    key = json.loads((out / "key.json").read_text())
    mech = json.loads((out / "mech.json").read_text())
    grades = json.loads((out / "grades.json").read_text())
    tally = {"exact": [0, 0], "capped": [0, 0]}
    for sid, k in key.items():
        tally[k["arm"]][1] += 1
        tally[k["arm"]][0] += grades[sid]["verdict"] == "flawed"
    e, c = tally["exact"], tally["capped"]
    print(f"flawed: exact {e[0]}/{e[1]}, budget {c[0]}/{c[1]}, Fisher p = {fisher_two_sided(e[0], e[1], c[0], c[1]):.3f}")
    for arm in ("exact", "capped"):
        ids = [s for s, k in key.items() if k["arm"] == arm]
        print(f"{arm}: repetition>0.3 in {sum(mech[s]['repetition'] > 0.3 for s in ids)}, cut by the cap "
              f"{sum(mech[s]['finish'] == 'length' for s in ids)}, mean tokens {sum(mech[s]['tokens'] for s in ids) / len(ids):.0f}")
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
    r.add_argument("arm")
    r.add_argument("--out", required=True)
    for name in ("sheet", "score"):
        sub.add_parser(name).add_argument("out")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.arm, Path(args.out))
    else:
        {"sheet": sheet, "score": score}[args.cmd](Path(args.out))


if __name__ == "__main__":
    main()
