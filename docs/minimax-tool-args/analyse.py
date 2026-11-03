"""Score the MiniMax tool-call replay (docs/minimax-tool-args/RECORD.md): per arm, replies classified with the tool-exact replay's rule
(valid / nonjson / other-call / no-call / error) plus the server's [request] speed lines. Usage: analyse.py <results dir>"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

D = Path(sys.argv[1])
src = (Path(__file__).resolve().parent.parent / "v1017-tool-exact" / "analyse.py").read_text().split("by_mode = {")[0]
src = src.replace('D = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/tool-exact")', "D = None")
ns: dict = {}
exec(src, ns)  # classify(), fisher_one_sided()
REQ = re.compile(r"\[request\] prompt=(\d+) reused=(\d+) prefilled=(\d+) prefill=([\d.]+)s completion=(\d+) decode=([\d.]+)s \(([\d.]+) tok/s\)")
out = {}
for tag, mode in (("A", "shipped"), ("B", "exact")):
    allrows = [json.loads(l) for l in (D / f"{tag}.driver.jsonl").read_text().splitlines() if l.startswith("{")]
    rows = [r for r in allrows if "tag" in r]
    stats = next((r["stats"] for r in allrows if "stats" in r), {})
    flat = json.dumps(stats)
    print(f"{tag} {mode}: server stats substitution fields: " + ", ".join(f"{k}={v}" for k, v in re.findall(r'"(\w*subst\w*|\w*drop\w*)": ([\d.]+)', flat)))
    log = (D / f"{tag}.server.log").read_text(errors="replace")
    full = re.search(r"run: output -> (\S+)", log)
    text = Path(full.group(1)).read_text(errors="replace") if full else log
    kinds = [(r.get("tag"), *ns["classify"](r)) for r in rows]
    c = Counter(k for _, k, _ in kinds)
    out[mode] = (len(rows), c["nonjson"] + c["no-call"] + c["error"])
    reqs = REQ.findall(text)
    tps = [float(x[6]) for x in reqs]
    print(f"{tag} {mode}: {len(rows)} replies; {dict(c)}; limits {[v for _, k, v in kinds if k in ('valid', 'nonjson')]}")
    print(f"   decode {min(tps, default=0):.2f}-{max(tps, default=0):.2f} tok/s, completions {[int(x[4]) for x in reqs]}, "
          f"manifest env drops: see benchmarks/results/minimax-tool-args/manifest-*.json")
    for t, k, v in kinds:
        if k not in ("valid",):
            print(f"   {t}: {k} {v!r}")
(na, ba), (nb, bb) = out["shipped"], out["exact"]
print(f"bad (non-JSON, no call, error): shipped {ba}/{na}, exact {bb}/{nb}; one-sided p (exact no worse by chance) = "
      f"{ns['fisher_one_sided'](ba, na, bb, nb):.3f}")
