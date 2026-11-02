"""Score the hotlist ABBA: per arm the first turn's prefill, decode, misses and drops; then on against off. Usage: analyse.py <results dir>"""
import json
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
ARMS = [("A1", "on"), ("B1", "off"), ("B2", "off"), ("A2", "on")]
res = {}
for tag, mode in ARMS:
    f = D / f"{tag}.driver.jsonl"
    if not f.exists():
        continue
    rows = {r["tag"]: r for r in (json.loads(l) for l in f.read_text().splitlines() if l.startswith("{"))}
    full = Path(re.search(r"run: output -> (\S+)", (D / f"{tag}.server.log").read_text()).group(1))  # guarded_run's whole output
    text = full.read_text()
    req = [l for l in text.splitlines() if l.startswith("[request]")]
    assert len(req) == 1, len(req)
    line = req[0]
    g = lambda k: float(re.search(k + r"=([\d.]+)", line).group(1))
    pre = re.search(r"hotlist: (\d+) experts preloaded", text)
    a, b = rows["stats-before"]["stats"], rows["stats-after"]["stats"]
    hits, miss = b["expert_hits"] - a["expert_hits"], b["expert_misses"] - a["expert_misses"]
    turn = rows["first-turn"]
    comp = turn["usage"]["completion_tokens"]
    res[tag] = dict(mode=mode, prompt=int(g("prompt")), prefill=g("prefill"), comp=comp, decode_s=g("decode"),
                    ms=1000 * g("decode") / comp, miss_tok=g("miss/tok"), hit_line=g("hit"),
                    hit_turn=hits / (hits + miss), dropped=(b["skipped_experts"] - a["skipped_experts"]) / comp,
                    preloaded=int(pre.group(1)) if pre else 0, finish=turn["finish"],
                    out=json.dumps([turn["content"], turn["tool_calls"]])[:120])
for tag, r in res.items():
    print(f"{tag} {r['mode']:3s} preloaded {r['preloaded']:4d}  prompt {r['prompt']}  prefill {r['prefill']:7.2f} s  "
          f"decode {r['ms']:6.1f} ms/token over {r['comp']} ({r['finish']})  miss/tok {r['miss_tok']:5.1f}  "
          f"dropped/tok {r['dropped']:5.2f}  hit(turn) {r['hit_turn']:.3f}")
    print(f"     out {r['out']}")


def mean(mode, k):
    v = [r[k] for r in res.values() if r["mode"] == mode]
    return sum(v) / len(v) if v else float("nan")


if len(res) == 4:
    for k, unit in (("prefill", "s"), ("ms", "ms/token"), ("dropped", "dropped/token"), ("hit_turn", "hit"), ("miss_tok", "miss/token")):
        on, off = mean("on", k), mean("off", k)
        print(f"{k:9s} on {on:9.3f} off {off:9.3f}  on-off {on - off:+.3f} {unit} ({100 * (on - off) / off:+.2f} %)")
    for mode, (x, y) in (("on", ("A1", "A2")), ("off", ("B1", "B2"))):
        p, q = res[x]["prefill"], res[y]["prefill"]
        print(f"repeat {mode}: prefill {p:.2f} / {q:.2f} ({100 * abs(p - q) / min(p, q):.2f} % apart); outputs equal: "
              f"{res[x]['out'] == res[y]['out']}")
    print("all four outputs equal:", len({r["out"] for r in res.values()}) == 1)
