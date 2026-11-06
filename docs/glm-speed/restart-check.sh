#!/bin/bash
# usage: docs/glm-speed/restart-check.sh <tag>   a fresh ./serve-glm.sh at 46 GiB with the shipped snapshot directory, the live session's first request only, then the server stops.
# Does a restart reuse the system block's disk snapshot? (docs/glm-speed/G2-RECORD.md)
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
TAG=$1; D=benchmarks/results/glm-speed; mkdir -p "$D"
benchmarks/guarded_run.sh --budget-gib 46 --max-seconds 900 --force --tag "glm-restart-$TAG" -- ./serve-glm.sh --expert-budget-gib 46 > "$D/$TAG.server.log" 2>&1 &
GPID=$!
sleep 12  # guarded_run.sh refuses to start if this driver is already running
~/venvs/deepseek-v41/bin/python - "$D/$TAG.driver.jsonl" <<'PY'
import json, os, sys, time, urllib.request
URL = "http://127.0.0.1:8011/v1"
body = next(b for b in (json.loads(l).get("body") or {} for l in open(os.path.expanduser("~/cachalot-dump-glm-upscale.jsonl"))) if b.get("messages"))
t0 = time.time()
while time.time() - t0 < 400:
    try:
        urllib.request.urlopen(URL + "/models", timeout=5).read(); break
    except Exception:
        time.sleep(2)
body = dict(body, stream=False, max_tokens=30); body.pop("stream_options", None)
t0 = time.time()
out = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=800).read())
open(sys.argv[1], "w").write(json.dumps({"wall_s": round(time.time() - t0, 2), "usage": out.get("usage")}) + "\n")
PY
for _ in $(seq 1 60); do SPID=$(lsof -ti tcp:8011 -sTCP:LISTEN); [ -n "$SPID" ] && break; kill -0 $GPID 2>/dev/null || break; sleep 5; done
[ -n "${SPID:-}" ] && kill -INT $SPID
wait $GPID
echo done $(date +%T)
