"""Score the rare-skip server ABBA (docs/prefill-rare-skip/RECORD.md): per arm the two cold prefills' seconds, then rare against exact. Usage: analyse.py <results dir>"""
import json
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
ARMS = [("A1", "exact"), ("B1", "rare"), ("B2", "rare"), ("A2", "exact")]
res = {}
for tag, mode in ARMS:
    full = Path(re.search(r"run: output -> (\S+)", (D / f"{tag}.server.log").read_text()).group(1))  # guarded_run's whole output
    text = full.read_text()
    req = [l for l in text.splitlines() if l.startswith("[request]")]
    rows = [json.loads(l) for l in (D / f"{tag}.driver.jsonl").read_text().splitlines() if l.startswith("{")]
    assert len(req) == 2 == len(rows), (len(req), len(rows))
    res[tag] = {"mode": mode, "di2_line": "prefill rare-expert skip" in text, "rare": re.findall(r"rare_skip=\S+", text)}
    for r, line in zip(rows, req):
        g = lambda k: float(re.search(k + r"=([\d.]+)", line).group(1))
        res[tag][r["tag"]] = {"prompt": int(g("prompt")), "prefill": g("prefill"), "out": json.dumps([r["content"], r["tool_calls"]])[:80]}
for tag, r in res.items():
    print(tag, r["mode"], "rare line" if r["di2_line"] else "no rare line", r["rare"],
          *(f"{k}: {r[k]['prompt']} tokens {r[k]['prefill']:.2f} s" for k in ("hermes-block", "mid-prompt")))
for k in ("hermes-block", "mid-prompt"):
    a = [r[k]["prefill"] for r in res.values() if r["mode"] == "exact"]
    b = [r[k]["prefill"] for r in res.values() if r["mode"] == "rare"]
    ma, mb = sum(a) / 2, sum(b) / 2
    print(f"{k}: exact {a[0]:.2f}/{a[1]:.2f} (mean {ma:.2f}), rare {b[0]:.2f}/{b[1]:.2f} (mean {mb:.2f}); rare - exact {mb - ma:+.2f} s "
          f"({100 * (mb - ma) / ma:+.1f} %); pairs apart exact {100 * abs(a[0] - a[1]) / min(a):.1f} %, rare {100 * abs(b[0] - b[1]) / min(b):.1f} %")
    print(f"  replies: " + " | ".join(f"{t}:{r[k]['out']}" for t, r in res.items()))
