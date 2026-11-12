"""Scores the G6 ABBA (docs/glm-speed/G6-RECORD.md): per rung, prefill seconds in S1, D1, D2, S2, the two pairs' D - S, and
the identity check (prefilled counts and reply hashes equal in all four arms). Usage: g6_analyse.py [results dir] [arms, S1,D1,D2,S2
in ABBA order; G6b passes S1,P1,P2,S2]."""
import json, sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/glm-speed/g6")
NAMES = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("S1", "D1", "D2", "S2")
arms = {}
for a in NAMES:
    arms[a] = {r["rung"]: r for r in (json.loads(l) for l in (d / f"{a}.driver.jsonl").read_text().splitlines() if l.startswith("{"))}
print(f"{'rung':>4} {'tokens':>6} | " + " ".join(f"{a:>6}" for a in NAMES) + " |" + f" {'pair1':>7} {'pair2':>7} {'mean':>7} | {'S spread':>8} | same")
for k in sorted(arms["S1"]):
    rows = [arms[a].get(k) for a in NAMES]
    if any(r is None or "usage" not in r for r in rows):
        print(k, "missing")
        continue
    u = [r["usage"] for r in rows]
    t = [x["cachalot"]["prefill_seconds"] for x in u]
    n = {x["prompt_tokens"] - x["cachalot"]["reused_prefix_tokens"] for x in u}
    same = len(n) == 1 and len({r["reply_sha"] for r in rows}) == 1
    p1, p2 = t[1] / t[0] - 1, t[2] / t[3] - 1
    print(f"{k:4d} {max(n):6d} | {t[0]:6.2f} {t[1]:6.2f} {t[2]:6.2f} {t[3]:6.2f} | {p1:+7.1%} {p2:+7.1%} {(p1 + p2) / 2:+7.1%} | "
          f"{abs(t[3] / t[0] - 1):8.1%} | {'yes' if same else 'NO'}")
