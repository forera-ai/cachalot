"""Score the screenshot-transcription replay (docs/v1017-image-ocr/RECORD.md) against the private ground truth
(~/cachalot-private/ocr-truth.json, holds addresses, never committed). Prints item ids and counts only, no transcribed text.
Usage: analyse.py <results dir> [truth.json]"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(sys.argv[1])
truth = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else Path.home() / "cachalot-private/ocr-truth.json").read_text())
ITEMS = truth["items"]


def norm(s: str) -> str:
    return re.sub(r"[`*_]", "", s or "")


def score(text: str) -> list[str]:
    """Item ids that the reply gets wrong: a required string missing, the Edmonton rows short, or punctuation glued to the typed value."""
    t = norm(text)
    bad = [k for k, v in ITEMS.items() if v not in t]
    if t.count("Edmonton") < truth["edmonton_rows"]:
        bad.append("edmonton")
    typed = truth["typed"]
    if re.search(re.escape(typed) + r"[,.;:](?!\s*\d)", t):
        bad.append("typed_punct")
    return bad


def perm_p(a: list[int], b: list[int], n: int = 20000) -> float:
    """Two-sided permutation p for the difference in mean errors a reply."""
    import random
    rnd = random.Random(0)
    obs = abs(sum(a) / len(a) - sum(b) / len(b))
    pool = a + b
    hits = 0
    for _ in range(n):
        rnd.shuffle(pool)
        x, y = pool[: len(a)], pool[len(a):]
        hits += abs(sum(x) / len(x) - sum(y) / len(y)) >= obs - 1e-12
    return hits / n


errs = {}
for tag, mode in (("A", "budget0"), ("B", "exact")):
    rows = [json.loads(l) for l in (D / f"{tag}.driver.jsonl").read_text().splitlines() if l.startswith("{")]
    log = (D / f"{tag}.server.log").read_text()
    full = re.search(r"run: output -> (\S+)", log)
    text = Path(full.group(1)).read_text() if full else log
    by = defaultdict(list)
    items = Counter()
    per = []
    for r in rows:
        if "error" in r or r.get("tool_calls"):
            bad = ["error" if "error" in r else "tool_call"]
        else:
            bad = score(r["content"])
        by[r["tag"]].append(len(bad))
        items.update(bad)
        per.append(len(bad))
    errs[mode] = per
    print(f"{tag} {mode}: {len(rows)} replies, budget line {'yes' if 'decode miss budget 0' in text else 'no'}; "
          f"errors a reply {sum(per) / len(per):.2f}, replies with any error {sum(1 for x in per if x)}/{len(per)}; items {dict(items)}")
    for c in sorted(by):
        print(f"   {c}: errors {by[c]}")
a, b = errs["budget0"], errs["exact"]
print(f"errors a reply: budget 0 {sum(a) / len(a):.2f}, exact {sum(b) / len(b):.2f}; permutation p = {perm_p(a, b):.3f}")
