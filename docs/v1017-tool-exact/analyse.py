"""Scores the tool-exact replay A/B from benchmarks/results/tool-exact/ (driver replies and the servers' [request] lines).
Usage: analyse.py [results_dir]"""
import json, math, re, statistics, sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/tool-exact")
ARMS = {"A1": "budget0", "B1": "toolexact", "B2": "toolexact", "A2": "budget0"}
REQ = re.compile(r"\[request\] prompt=(\d+) reused=(\d+) prefilled=(\d+) prefill=([\d.]+)s completion=(\d+) decode=([\d.]+)s "
                 r"\(([\d.]+) tok/s\).*?miss/tok=([\d.]+) hit=(\d+)% .*?finish=(\w+)")


def classify(r):
    """valid | nonjson | other-call | no-call | error, and the limit value as written."""
    if "error" in r:
        return "error", None
    calls = r.get("tool_calls") or []
    if not calls:
        return "no-call", None
    for c in calls:
        fn = c.get("function", c)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                m = re.search(r'"limit":\s*([^,}]+)', fn.get("arguments", ""))
                return "nonjson", m.group(1).strip() if m else fn.get("arguments", "")[:40]
        if fn.get("name") == "search_files":
            lim = args.get("limit")
            if lim is not None and not isinstance(lim, int):
                return "nonjson", repr(lim)
            return "valid", lim
    return "other-call", calls[0].get("function", calls[0]).get("name")


def server_lines(tag):
    p = D / f"{tag}.server.log"
    path = p
    for line in p.read_text(errors="replace").splitlines():
        if "run: output ->" in line:
            path = Path(line.split("run: output ->", 1)[1].strip())
    return [m.groups() for m in REQ.finditer(path.read_text(errors="replace"))]


def fisher_one_sided(a_bad, a_n, b_bad, b_n):
    """P(B has <= b_bad bad | margins), hypergeometric: B no worse than A by chance."""
    k, n, N = a_bad + b_bad, b_n, a_n + b_n
    return sum(math.comb(k, x) * math.comb(N - k, n - x) for x in range(0, b_bad + 1)) / math.comb(N, n)


by_mode = {"budget0": [], "toolexact": []}
speed = {"budget0": [], "toolexact": []}
for tag, mode in ARMS.items():
    p = D / f"{tag}.driver.jsonl"
    if not p.exists():
        print(f"{tag}: missing")
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.startswith("{")]
    kinds = [(r.get("tag"), *classify(r)) for r in rows]
    by_mode[mode] += kinds
    reqs = server_lines(tag)
    speed[mode] += [(int(c), float(d), float(t), float(m), int(h)) for *_, c, d, t, m, h, fin in reqs]
    print(f"{tag} ({mode}): " + ", ".join(f"{t}:{k}" + (f"={v}" if k == 'nonjson' else "") for t, k, v in kinds))

print()
for mode, kinds in by_mode.items():
    n = len(kinds)
    bad = sum(k == "nonjson" for _, k, _ in kinds)
    counts = {k: sum(x == k for _, x, _ in kinds) for k in ("valid", "nonjson", "other-call", "no-call", "error")}
    per = {p: f"{sum(k == 'nonjson' for t, k, _ in kinds if t == p)}/{sum(t == p for t, _, _ in kinds)}" for p in ("P1", "P2", "P3")}
    print(f"{mode}: non-JSON {bad}/{n}  {counts}  by prompt {per}")
    s = speed[mode]
    if s:
        print(f"   decode tok/s median {statistics.median(x[2] for x in s):.2f} (range {min(x[2] for x in s):.2f}-{max(x[2] for x in s):.2f}), "
              f"ms/token median {statistics.median(1000 * x[1] / x[0] for x in s if x[0]):.1f}, "
              f"miss/tok median {statistics.median(x[3] for x in s):.1f}, completion median {statistics.median(x[0] for x in s)}")
a, b = by_mode["budget0"], by_mode["toolexact"]
if a and b:
    ab, bb = sum(k == "nonjson" for _, k, _ in a), sum(k == "nonjson" for _, k, _ in b)
    print(f"\none-sided Fisher, tool-exact fewer non-JSON than budget 0: p = {fisher_one_sided(ab, len(a), bb, len(b)):.4f}")
    sa, sb = speed["budget0"], speed["toolexact"]
    if sa and sb:
        ma = statistics.median(1000 * x[1] / x[0] for x in sa if x[0])
        mb = statistics.median(1000 * x[1] / x[0] for x in sb if x[0])
        print(f"ms a reply token, median: budget0 {ma:.1f}, toolexact {mb:.1f} ({100 * (mb / ma - 1):+.1f} %)")
