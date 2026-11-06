"""Replays the five requests of the 2026-11-06 live GLM session (HANDOFF 18.271) against a running server and records each turn's usage, never the text
in the repository (docs/glm-speed/G3-RECORD.md). Usage: driver.py [dump]. The dump holds personal paths and stays outside the repository; requests are posted as
Hermes sent them, non-streaming. A reply here differs from the recorded one, so each later request carries the recorded history: the same prompt as live."""
from __future__ import annotations

import json, os, sys, time, urllib.request

URL = "http://127.0.0.1:8011/v1"
DUMP = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/cachalot-dump-glm-upscale.jsonl")


def wait_ready(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(URL + "/models", timeout=5).read()
            return True
        except Exception:
            time.sleep(2)
    return False


bodies = [json.loads(l).get("body") or {} for l in open(DUMP)]  # the dump also holds response rows, without a body
bodies = [b for b in bodies if b.get("messages")]
if not wait_ready():
    sys.exit("server not ready")
for k, body in enumerate(bodies):
    body = dict(body, stream=False)
    body.pop("stream_options", None)
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        choice = out["choices"][0]
        print(json.dumps({"turn": k, "wall_s": round(time.time() - t0, 2), "finish": choice.get("finish_reason"), "usage": out.get("usage")}), flush=True)
    except Exception as exc:
        print(json.dumps({"turn": k, "error": repr(exc)}), flush=True)
print("driver done", flush=True)
