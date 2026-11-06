"""Score the knob check (docs/glm-image-ocr/KNOB-RECORD.md) with the private truth files. Prints item ids and counts only. Usage: analyse-knob.py <results dir>"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

results = sys.argv[1]
sys.argv = sys.argv[:1] + ["/unused"]
src = (Path(__file__).resolve().parent.parent / "v1017-image-ocr" / "analyse.py").read_text().split("errs = {}")[0]
ns: dict = {}
exec(src, ns)
card = json.loads((Path.home() / "cachalot-private/ocr-truth-card.json").read_text())["items"]


def score_card(text):
    t = ns["norm"](text)
    return [k for k, v in card.items() if v not in t]


rows = [json.loads(l) for l in (Path(results) / "K.driver.jsonl").read_text().splitlines() if l.startswith("{")]
print(f"{len(rows)} rows, {sum('error' in r for r in rows)} errors")
for tag, fn in (("A492", ns["score"]), ("C361", score_card)):
    rs = [r for r in rows if r["tag"] == tag and "error" not in r]
    bad = [fn(r["content"] or "") for r in rs]
    print(f"{tag}: {len(rs)} replies, with an error {sum(1 for b in bad if b)}, per reply {[len(b) for b in bad]}, items {dict(Counter(i for b in bad for i in b))}, "
          f"prompt {sorted({r['usage']['prompt_tokens'] for r in rs})}, reply tokens {min(r['usage']['completion_tokens'] for r in rs)}-{max(r['usage']['completion_tokens'] for r in rs)}, "
          f"wall median {statistics.median(r['wall_s'] for r in rs):.1f} s")
