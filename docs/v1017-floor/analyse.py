"""Score one or more arms of the 1017 floor re-record: per-request ms a token and misses a token, the floor repeats, the mixed mean,
the cost of a miss and the end-of-run hit rate. Usage: analyse.py <results dir> A C"""
import json
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
for arm in sys.argv[2:]:
    rows = [json.loads(l) for l in (D / f"{arm}.driver.jsonl").read_text().splitlines() if l.startswith("{")]
    reqs = [r for r in rows if "usage" in r]
    server = (D / f"{arm}.server.log").read_text()
    full = Path(re.search(r"run: output -> (\S+)", server).group(1))  # guarded_run keeps the whole server output there
    log = [l for l in full.read_text().splitlines() if l.startswith("[request]")]
    assert len(log) == len(reqs), (len(log), len(reqs))
    out = {}
    for r, l in zip(reqs, log):
        c = r["usage"]["cachalot"]
        ms = 1000 * c["decode_seconds"] / r["usage"]["completion_tokens"]
        m = float(re.search(r"miss/tok=([\d.]+)", l).group(1))
        out[r["tag"]] = (ms, m)
    end = next(r["stats"] for r in rows if r.get("tag") == "stats-end")
    start = next(r["stats"] for r in rows if r.get("tag") == "stats-after-warmup")
    mix = [v for k, v in out.items() if k.startswith("mix")]
    mix_ms = sum(v[0] for v in mix) / len(mix)
    mix_m = sum(v[1] for v in mix) / len(mix)
    floor12 = (out["floor1"][0] + out["floor2"][0]) / 2
    print(f"arm {arm}")
    for k, (ms, m) in out.items():
        print(f"  {k:8s} {ms:7.1f} ms/token  {m:5.1f} miss/token")
    print(f"  floor repeats 1-2 {out['floor1'][0]:.1f} / {out['floor2'][0]:.1f} (mean {floor12:.1f}); after mix "
          f"{out['floor3'][0]:.1f} ({out['floor3'][1]:.1f} miss) / {out['floor4'][0]:.1f} ({out['floor4'][1]:.1f} miss)")
    print(f"  mixed mean {mix_ms:.1f} ms at {mix_m:.2f} miss/token; per miss (mixed - floor) / misses = "
          f"{(mix_ms - floor12) / mix_m if mix_m else float('nan'):.2f} ms")
    dh, dm = end["expert_hits"] - start["expert_hits"], end["expert_misses"] - start["expert_misses"]
    print(f"  hit rate end of run {end['expert_hit_rate']:.4f}; after warm-up {dh / (dh + dm):.4f}; "
          f"skipped {end['skipped_experts'] - start['skipped_experts']}; predicted loads {end['predicted_loads']} used {end['predicted_used']}")
