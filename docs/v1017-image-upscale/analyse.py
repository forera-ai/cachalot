"""Score the upscale A/B (docs/v1017-image-upscale/RECORD.md) with the image-ocr scorer and its private ground truth
(~/cachalot-private/ocr-truth.json; item ids only). Usage: analyse.py <results dir>"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.argv, results = sys.argv[:1] + ["/unused"], sys.argv[1]
src = (Path(__file__).resolve().parent.parent / "v1017-image-ocr" / "analyse.py").read_text().split("errs = {}")[0]
ns: dict = {}
exec(src, ns)  # score(), perm_p(), the truth file
D = Path(results)


def fisher(a, b, c, d):
    n1, n2, k = a + b, c + d, a + c
    p = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)  # noqa: E731
    o = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= o * (1 + 1e-9)))


arms = {}
for tag, mode in (("A", "shipped"), ("B", "upscale")):
    rows = [json.loads(l) for l in (D / f"{tag}.driver.jsonl").read_text().splitlines() if l.startswith("{")]
    log = (D / f"{tag}.server.log").read_text()
    full = re.search(r"run: output -> (\S+)", log)
    text = Path(full.group(1)).read_text() if full else log
    tools = sum(1 for r in rows if r.get("tool_calls"))
    errors = [r for r in rows if "error" in r]
    texts = [(r["tag"], ns["score"](r["content"] or "")) for r in rows if not r.get("tool_calls") and "error" not in r]
    bad = [t for t in texts if t[1]]
    items = Counter(i for _, b in texts for i in b)
    prompt = sorted({int(m) for m in re.findall(r"\[request\] prompt=(\d+)", text)})
    arms[mode] = (len(rows), tools, len(texts), len(bad))
    print(f"{tag} {mode}: {len(rows)} replies ({len(errors)} errors), upscale line {'yes' if 'vision upscale' in text else 'no'}, "
          f"prompt tokens {prompt}; tool calls {tools}; text replies {len(texts)}, with a character error {len(bad)}; items {dict(items)}")
(na, ta, xa, ba), (nb, tb, xb, bb) = arms["shipped"], arms["upscale"]
print(f"tool calls: shipped {ta}/{na}, upscale {tb}/{nb}, Fisher p = {fisher(ta, na - ta, tb, nb - tb):.3f}")
print(f"text replies with an error: shipped {ba}/{xa}, upscale {bb}/{xb}" + (f", Fisher p = {fisher(ba, xa - ba, bb, xb - bb):.3f}" if xa and xb else ""))
