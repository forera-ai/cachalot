"""Score the translate-skip replay (docs/v1017-translate-skip/RECORD.md): per arm and recorded copy, replies that started a translation,
stopped after the English (skip), hit the cap, made a tool call or failed; Fisher's exact test on skips. Usage: analyse.py <results dir>"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(sys.argv[1])


def fisher_two_sided(a, b, c, d):
    """2x2 table [[a, b], [c, d]]: sum of hypergeometric probabilities no larger than the observed one."""
    n1, n2, k = a + b, c + d, a + c
    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= obs * (1 + 1e-9)))


def kind(r):
    if "error" in r:
        return "error"
    if r["marker"]:
        return "translated"
    if r["tool"]:
        return "tool"
    if r["finish"] == "length":
        return "cap"
    return "skip"


arms = {}
for tag, mode in (("A", "budget0"), ("B", "exact")):
    rows = [json.loads(l) for l in (D / f"{tag}.driver.jsonl").read_text().splitlines() if l.startswith("{")]
    log = (D / f"{tag}.server.log").read_text()
    full = re.search(r"run: output -> (\S+)", log)
    text = Path(full.group(1)).read_text() if full else log
    budget_line = "decode miss budget 0" in text
    by = defaultdict(Counter)
    for r in rows:
        by[r["tag"]][kind(r)] += 1
    total = Counter(kind(r) for r in rows)
    arms[mode] = total
    print(f"{tag} {mode}: {len(rows)} replies, budget line {'yes' if budget_line else 'no'}; " + ", ".join(f"{k} {v}" for k, v in sorted(total.items())))
    for h in sorted(by):
        print(f"   {h}: " + ", ".join(f"{k} {v}" for k, v in sorted(by[h].items())))
    markers = Counter(r.get("marker") for r in rows if r.get("marker"))
    walls = [r["wall_s"] for r in rows if "wall_s" in r]
    print(f"   markers {dict(markers)}; wall a reply {min(walls):.0f}-{max(walls):.0f} s, total {sum(walls) / 60:.1f} min")
a, b = arms["budget0"], arms["exact"]
na, nb = sum(a.values()), sum(b.values())
p = fisher_two_sided(a["skip"], na - a["skip"], b["skip"], nb - b["skip"])
print(f"skips: budget 0 {a['skip']}/{na}, exact {b['skip']}/{nb}; Fisher two-sided p = {p:.3f}")
