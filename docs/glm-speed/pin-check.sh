#!/bin/bash
# Arms A, B and C of docs/glm-speed/G2-RECORD.md: three fresh ./serve-glm.sh at 46 GiB on temporary snapshot directories, one request each. About 10 minutes.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
D=benchmarks/results/glm-speed/pin-check; rm -rf "$D"; mkdir -p "$D/snap-ab" "$D/snap-c"
one() {  # tag, snapshot dir, dump, line of a request in that dump
  export CACHALOT_GLM_SNAPSHOT_DIR=$PWD/$2
  benchmarks/guarded_run.sh --budget-gib 46 --max-seconds 900 --force --tag "glm-pin-$1" -- ./serve-glm.sh --expert-budget-gib 46 > "$D/$1.server.log" 2>&1 &
  G=$!
  sleep 12  # guarded_run.sh refuses to start if this driver is already running
  ~/venvs/deepseek-v41/bin/python - "$D/$1.driver.jsonl" "$3" <<'PY'
import json, os, sys, time, urllib.request
URL = "http://127.0.0.1:8011/v1"
body = next(b for b in (json.loads(l).get("body") or {} for l in open(os.path.expanduser(sys.argv[2]))) if b.get("messages"))
t0 = time.time()
while time.time() - t0 < 400:
    try:
        urllib.request.urlopen(URL + "/models", timeout=5).read(); break
    except Exception:
        time.sleep(2)
body = dict(body, stream=False, max_tokens=2, model="glm-5.3-flash"); body.pop("stream_options", None)
t0 = time.time()
out = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=800).read())
open(sys.argv[1], "w").write(json.dumps({"wall_s": round(time.time() - t0, 2), "usage": out.get("usage")}) + "\n")
PY
  for _ in $(seq 1 60); do S=$(lsof -ti tcp:8011 -sTCP:LISTEN); [ -n "$S" ] && break; kill -0 $G 2>/dev/null || break; sleep 5; done
  [ -n "${S:-}" ] && kill -INT $S
  wait $G
  echo "arm $1 done $(date +%T): $(cut -c1-150 "$D/$1.driver.jsonl")"
  sleep 5
}
one A "$D/snap-ab" "~/cachalot-dump-glm-upscale.jsonl"
one B "$D/snap-ab" "~/cachalot-dump-upscale.jsonl"
one C "$D/snap-c" "~/cachalot-dump-upscale.jsonl"
~/venvs/deepseek-v41/bin/python docs/glm-speed/compare_blocks.py "$D/snap-ab" "$D/snap-c"
rm -rf "$D/snap-ab" "$D/snap-c"
echo "all arms done $(date +%T)"
