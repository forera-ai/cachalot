"""Score the form A/B (docs/glm-image-ocr/FORM-RECORD.md) with the image-ocr scorer and the private truth. Prints item ids and counts only.
Usage: analyse-form.py <results dir>"""
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

results = sys.argv[1]
sys.argv = sys.argv[:1] + ["/unused"]
src = (Path(__file__).resolve().parent.parent / "v1017-image-ocr" / "analyse.py").read_text().split("errs = {}")[0]
ns: dict = {}
exec(src, ns)


def fisher(a, b, c, d):
    n1, n2, k = a + b, c + d, a + c
    p = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)  # noqa: E731
    o = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= o * (1 + 1e-9)))


cells = {}
for arm, mode in (("A", "budget2"), ("B", "exact")):
    rows = [json.loads(l) for l in (Path(results) / f"{arm}.driver.jsonl").read_text().splitlines() if l.startswith("{")]
    print(f"{arm} {mode}: {len(rows)} rows, {sum('error' in r for r in rows)} errors")
    for tag in ("A492", "A886"):
        rs = [r for r in rows if r["tag"] == tag and "error" not in r]
        bad = [ns["score"](r["content"] or "") for r in rs]
        cells[mode, tag] = (sum(1 for b in bad if b), len(bad))
        items = Counter(i for b in bad for i in b)
        print(f"  {tag}: {len(rs)} replies, with an error {cells[mode, tag][0]}, per reply {[len(b) for b in bad]}, items {dict(items)}, "
              f"prompt {sorted({r['usage']['prompt_tokens'] for r in rs})}, reply tokens {min(r['usage']['completion_tokens'] for r in rs)}-"
              f"{max(r['usage']['completion_tokens'] for r in rs)}, wall median {statistics.median(r['wall_s'] for r in rs):.1f} s")


def pooled(keys):
    return sum(cells[k][0] for k in keys), sum(cells[k][1] for k in keys)


nat, pad = pooled([("budget2", "A492"), ("exact", "A492")]), pooled([("budget2", "A886"), ("exact", "A886")])
bud, exa = pooled([("budget2", "A492"), ("budget2", "A886")]), pooled([("exact", "A492"), ("exact", "A886")])
print(f"native {nat[0]}/{nat[1]}, padded {pad[0]}/{pad[1]}, Fisher p = {fisher(nat[0], nat[1] - nat[0], pad[0], pad[1] - pad[0]):.3f}")
print(f"budget 2 {bud[0]}/{bud[1]}, exact {exa[0]}/{exa[1]}, Fisher p = {fisher(bud[0], bud[1] - bud[0], exa[0], exa[1] - exa[0]):.3f}")
