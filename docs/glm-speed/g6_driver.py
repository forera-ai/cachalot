"""G6 lever A: a ladder of short prefills after Hermes's system block (docs/glm-speed/G6-RECORD.md).

Usage: g6_driver.py [dump]. Request 0 is the live session's first request ("Hi" after the 15.6k-token block), which
prefills the block cold. Each later request keeps the same system block and replaces the user message with a slice of
docs/RESEARCH-DIRECTION.md of about N tokens, so it reuses the block's snapshot and prefills only its own text. Every
request is greedy and stops after one token, so no decode changes residency between rungs and both arms see the same
tokens. Records usage and a hash of the one-token reply, never the text."""
from __future__ import annotations

import hashlib, json, os, sys, time, urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8011/v1"
DUMP = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/cachalot-dump-glm-upscale.jsonl")
RUNGS = [25, 60, 120, 160, 250, 400, 528, 800, 1100]
CHARS_PER_TOKEN = 4


def wait_ready(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(URL + "/models", timeout=5).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def post(body):
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    text = out["choices"][0]["message"].get("content") or json.dumps(out["choices"][0]["message"].get("tool_calls"))
    return round(time.time() - t0, 2), out.get("usage"), hashlib.sha256(text.encode()).hexdigest()[:12]


rows = [json.loads(l).get("body") or {} for l in open(DUMP)]
base = next(b for b in rows if b.get("messages"))
base = dict(base, stream=False, temperature=0.0, max_tokens=1)
base.pop("stream_options", None)
source = Path(__file__).resolve().parents[1].joinpath("RESEARCH-DIRECTION.md").read_text()
if not wait_ready():
    sys.exit("server not ready")
offset = 0
for k, n in enumerate([None] + RUNGS):
    body = dict(base)
    if n is not None:
        text = source[offset:offset + n * CHARS_PER_TOKEN]
        offset += n * CHARS_PER_TOKEN
        body["messages"] = base["messages"][:-1] + [{"role": "user", "content": "Read this and say OK.\n\n" + text}]
    try:
        wall, usage, digest = post(body)
        print(json.dumps({"rung": k, "target": n, "wall_s": wall, "reply_sha": digest, "usage": usage}), flush=True)
    except Exception as exc:
        print(json.dumps({"rung": k, "target": n, "error": repr(exc)}), flush=True)
print("driver done", flush=True)
