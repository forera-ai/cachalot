"""Score the GLM screenshot replies (docs/glm-image-ocr/RECORD.md) against the private truth files ~/cachalot-private/ocr-truth.json (the
address form, with the image-ocr scorer) and ocr-truth-card.json (the calendar card). Prints item ids and counts only, no transcribed
text. Usage: analyse.py <results dir> [tag]"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

results, tag = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "A"
sys.argv = sys.argv[:1] + ["/unused"]
src = (Path(__file__).resolve().parent.parent / "v1017-image-ocr" / "analyse.py").read_text().split("errs = {}")[0]
ns: dict = {}
exec(src, ns)  # score(), the address truth
card = json.loads((Path.home() / "cachalot-private/ocr-truth-card.json").read_text())["items"]


def score_card(text):
    t = ns["norm"](text)
    return [k for k, v in card.items() if v not in t]


rows = [json.loads(l) for l in (Path(results) / f"{tag}.driver.jsonl").read_text().splitlines() if l.startswith("{")]
by = defaultdict(list)
for r in rows:
    by[r["tag"]].append(r)
print(f"{len(rows)} replies, {sum('error' in r for r in rows)} errors")
for t in ("A492", "A906", "C361", "C902"):
    rs = [r for r in by[t] if "error" not in r]
    fn = ns["score"] if t.startswith("A") else score_card
    bad = [fn(r["content"] or "") for r in rs]
    items = Counter(i for b in bad for i in b)
    prompt = sorted({r["usage"]["prompt_tokens"] for r in rs})
    gen = [r["usage"]["completion_tokens"] for r in rs]
    wall = [r["wall_s"] for r in rs]
    print(f"{t}: {len(rs)} replies, with an error {sum(1 for b in bad if b)}, per reply {[len(b) for b in bad]}, items {dict(items)}, prompt tokens {prompt}, "
          f"reply tokens {min(gen)}-{max(gen)}, wall median {statistics.median(wall):.1f} s, finish {dict(Counter(r.get('finish') for r in rs))}")
