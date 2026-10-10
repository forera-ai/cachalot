"""Once a second: IOReport energy counters by component, SMC `PSTR`, and (every 2 s) the DeepSeek server's /v1/stats counters.

Started by benchmarks/energy_arms.sh after the server is up (the server's one-runtime guard refuses to start while a `benchmarks/...` python
process exists, so the logger must come second). Needs a root `powermetrics` sampling beside it, or the counters stay frozen
(docs/POWER-ACCOUNTING-PLAN.md section 8). Stop with SIGINT or SIGTERM. Imports no MLX.

    ~/venvs/deepseek-v41/bin/python benchmarks/energy_logger.py --out benchmarks/results/energy/e3-log.jsonl
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import power_sources as ps  # noqa: E402

KEYS = ("ssd_bytes_read", "expert_read_seconds", "expert_reads", "expert_misses", "expert_hits", "tokens_generated", "requests_served")


def joules(rep) -> dict[str, float]:
    out: dict[str, float] = {}
    for ch in rep.sample():
        cm = ps._component(ch["group"], ch["name"])
        sc = ps._UNIT_TO_J.get(ch["unit"])
        if cm and sc and ch["value"] is not None:
            out[cm] = out.get(cm, 0.0) + ch["value"] * sc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-url", default="http://127.0.0.1:8011/v1/stats")
    a = ap.parse_args()
    stop = {"flag": False}
    for sg in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sg, lambda *_: stop.__setitem__("flag", True))
    rep, smc = ps.IOReport("Energy Model"), ps.SMC()
    n = 0
    with open(a.out, "w") as f:
        while not stop["flag"]:
            t = time.time()
            try:
                pstr = smc.read("PSTR")[2]
            except Exception:
                pstr = None
            row = {"t": t, "pstr": pstr, "joules": joules(rep)}
            if n % 2 == 0:
                try:
                    d = json.loads(urllib.request.urlopen(a.stats_url, timeout=3).read())
                    row["stats"] = {k: d.get(k) for k in KEYS}
                except Exception:
                    pass
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += 1
            time.sleep(max(0.0, 1.0 - (time.time() - t)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
