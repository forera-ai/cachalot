"""The knob check: native image bytes, the server upscales (docs/glm-image-ocr/KNOB-RECORD.md). Usage: driver-knob.py [max_tokens]
A492 and C361 as in driver.py (private, pinned by sha), five replies and three, in rotation while both remain. Same request as driver.py."""
from __future__ import annotations

import hashlib, json, os, sys, time, urllib.request

URL = "http://127.0.0.1:8011/v1"
MAX_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
PROMPT = "check this image and write me what is written inside the image"
SOURCES = [("A492", "cachalot-dump-toolexact.jsonl", 12, 13, "b32d000fc11f"), ("C361", "cachalot-dump-upscale.jsonl", 16, 17, "8f926e78694f")]


def load(name, line, msg, sha):
    with open(os.path.expanduser("~/" + name)) as f:
        for i, l in enumerate(f):
            if i == line:
                for part in json.loads(l)["body"]["messages"][msg]["content"]:
                    if part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if hashlib.sha256(url.encode()).hexdigest()[:12] != sha:
                            sys.exit(f"{name} image sha differs")
                        return url
    sys.exit(f"{name} has no image")


def wait_ready(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(URL + "/models", timeout=5).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def ask(url, tag, k):
    body = {"model": "glm-5.3-flash", "stream": False, "max_tokens": MAX_TOKENS, "temperature": 0.7, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]}]}
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    except Exception as exc:
        print(json.dumps({"tag": tag, "k": k, "error": repr(exc)}), flush=True)
        return
    choice = out["choices"][0]
    print(json.dumps({"tag": tag, "k": k, "wall_s": round(time.time() - t0, 3), "finish": choice.get("finish_reason"),
                      "usage": out.get("usage"), "content": choice["message"].get("content")}, ensure_ascii=False), flush=True)


urls = {tag: load(name, line, msg, sha) for tag, name, line, msg, sha in SOURCES}
if not wait_ready():
    sys.exit("server not ready")
for k in range(5):
    ask(urls["A492"], "A492", k)
    if k < 3:
        ask(urls["C361"], "C361", k)
print("driver done", flush=True)
