"""Replays Hermes's screenshot-transcription turn ("check this image and write me what is written inside the image") against a running
server and records each reply (docs/v1017-image-ocr/RECORD.md). Usage: driver.py <replies per recorded copy>

The three recorded copies, read from the local session dumps (the screenshot and the replies hold addresses: they stay outside the
repository; pinned by line and by the sha256 of their messages):
  C1  ~/cachalot-dump-toolexact.jsonl line 12  (2026-11-02, 492 x 512 PNG; live reply read the typed value as "1227,")  sha 66682b17f222397e
  C2  ~/cachalot-dump-1017.jsonl line 48       (2026-11-01, 492 x 512 PNG; live reply exact)                             sha c41de81c960d60b6
  C3  ~/cachalot-dump-di2.jsonl line 29        (2026-11-02, 906 x 943 PNG; no reply recorded)                             sha 6cb2d0436d8e9ee3
Each is posted as Hermes sent it (temperature 0.7, its tools, reasoning_effort none), non-streaming, max_tokens 600, copy-major."""
from __future__ import annotations

import hashlib, json, os, sys, time, urllib.request

URL = "http://127.0.0.1:8011/v1"
SOURCES = [("C1", "cachalot-dump-toolexact.jsonl", 12, "66682b17f222397e"),
           ("C2", "cachalot-dump-1017.jsonl", 48, "c41de81c960d60b6"),
           ("C3", "cachalot-dump-di2.jsonl", 29, "6cb2d0436d8e9ee3")]


def load(name, line, sha):
    with open(os.path.expanduser("~/" + name)) as f:
        for i, l in enumerate(f):
            if i == line:
                body = json.loads(l)["body"]
                got = hashlib.sha256(json.dumps(body["messages"], sort_keys=True).encode()).hexdigest()[:16]
                if got != sha:
                    sys.exit(f"{name}:{line} messages sha {got}, expected {sha}")
                return body
    sys.exit(f"{name} has no line {line}")


def wait_ready(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(URL + "/models", timeout=5).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def ask(body, tag, k):
    body = dict(body, stream=False, max_tokens=600)
    body.pop("stream_options", None)
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    except Exception as exc:
        print(json.dumps({"tag": tag, "k": k, "error": repr(exc)}), flush=True)
        return
    choice = out["choices"][0]
    msg = choice["message"]
    print(json.dumps({"tag": tag, "k": k, "wall_s": round(time.time() - t0, 3), "finish": choice.get("finish_reason"),
                      "usage": out.get("usage"), "content": msg.get("content"), "tool_calls": msg.get("tool_calls")},
                     ensure_ascii=False), flush=True)


bodies = [(tag, load(name, line, sha)) for tag, name, line, sha in SOURCES]
if not wait_ready():
    sys.exit("server not ready")
for tag, body in bodies:
    for k in range(int(sys.argv[1])):
        ask(body, tag, k)
print("driver done", flush=True)
