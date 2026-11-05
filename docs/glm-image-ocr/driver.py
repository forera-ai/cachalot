"""Asks GLM-5.3-Flash to transcribe screenshots through a running server, with no agent context (docs/glm-image-ocr/RECORD.md).
Usage: driver.py <replies per image> [max_tokens]

The images come from the local session dumps (they stay outside the repository, as do the replies), pinned by line, message index
and the sha256 of the image URL:
  A492  ~/cachalot-dump-toolexact.jsonl line 12 message 13  492 x 512 PNG, the address form    sha b32d000fc11f
  A906  ~/cachalot-dump-di2.jsonl       line 29 message 13  906 x 943 PNG, the same form       sha 99fc93b87e91
  C361  ~/cachalot-dump-upscale.jsonl   line 16 message 17  361 x 108 JPEG, a calendar card    sha 8f926e78694f
  C902  C361 padded up to 902 x 270 PNG here, as `ImageOps.pad` does for DeepSeek's CACHALOT_VISION_MIN_PIXELS
Images rotate within each round (A492, A906, C361, C902), so warming of the expert cache falls on all four alike. Temperature 0.7,
reasoning_effort none, as Hermes sends; no system prompt, no tools, non-streaming."""
from __future__ import annotations

import base64, hashlib, io, json, os, sys, time, urllib.request

from PIL import Image, ImageOps

URL = "http://127.0.0.1:8011/v1"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
PROMPT = "check this image and write me what is written inside the image"
SOURCES = [("A492", "cachalot-dump-toolexact.jsonl", 12, 13, "b32d000fc11f"),
           ("A906", "cachalot-dump-di2.jsonl", 29, 13, "99fc93b87e91"),
           ("C361", "cachalot-dump-upscale.jsonl", 16, 17, "8f926e78694f")]


def load(name, line, msg, sha):
    with open(os.path.expanduser("~/" + name)) as f:
        for i, l in enumerate(f):
            if i == line:
                for part in json.loads(l)["body"]["messages"][msg]["content"]:
                    if part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        got = hashlib.sha256(url.encode()).hexdigest()[:12]
                        if got != sha:
                            sys.exit(f"{name}:{line} image sha {got}, expected {sha}")
                        return url
    sys.exit(f"{name} line {line} has no image")


def padded(url, size):
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    out = io.BytesIO()
    ImageOps.pad(img, size).save(out, "PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


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
urls["C902"] = padded(urls["C361"], (902, 270))
if not wait_ready():
    sys.exit("server not ready")
for k in range(int(sys.argv[1])):
    for tag in ("A492", "A906", "C361", "C902"):
        ask(urls[tag], tag, k)
print("driver done", flush=True)
