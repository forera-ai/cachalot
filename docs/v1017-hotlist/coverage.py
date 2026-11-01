"""Coverage of a session's routed-expert requests by a ranked hot set's first N experts (N = GiB / 9.49 MiB, the q2g128 expert):
the old checkpoint's hotlist, one built from the other 1017 session (held out), and one built from the same session (in-sample)."""
import json
import sys
from collections import Counter
from pathlib import Path

from cachalot.metrics.routing_trace import PHASE_PREFILL, load_trace

EXPERT_BYTES = 9.49 * 2**20


def counts(path, phase):
    arrays, _ = load_trace(path)
    c = Counter()
    for layer, experts, ph in zip(arrays["layer"], arrays["experts"], arrays["phase"]):
        if phase == "decode" and ph == PHASE_PREFILL:
            continue
        if phase == "prefill" and ph != PHASE_PREFILL:
            continue
        for e in experts:
            c[(int(layer), int(e))] += 1
    return c


def coverage(ranked, c, n):
    top = set(map(tuple, ranked[:n]))
    return sum(v for k, v in c.items() if k in top) / max(1, sum(c.values()))


s1, s2 = sys.argv[1], sys.argv[2]
old = json.loads((Path.home() / "cachalot-hotlist.json").read_text())["experts"]
for phase in ("both", "decode"):
    c = {"s1": counts(s1, phase), "s2": counts(s2, phase)}
    rank = {k: [list(x) for x, _ in v.most_common()] for k, v in c.items()}
    print(f"phase {phase}: requests s1 {sum(c['s1'].values()):,}, s2 {sum(c['s2'].values()):,}")
    for gib in (4, 8, 16):
        n = int(gib * 2**30 // EXPERT_BYTES)
        for tgt, other in (("s1", "s2"), ("s2", "s1")):
            print(f"  {gib:>2} GiB ({n} experts) on {tgt}: old hotlist {coverage(old, c[tgt], n):.1%}, "
                  f"built on {other} {coverage(rank[other], c[tgt], n):.1%}, in-sample {coverage(rank[tgt], c[tgt], n):.1%}")
