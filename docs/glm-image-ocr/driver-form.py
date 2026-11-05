"""The address form only: native against padded, one server's decode setting (docs/glm-image-ocr/FORM-RECORD.md).
Usage: driver-form.py <replies per image> [max_tokens]

A492 is the 492 x 512 PNG of ~/cachalot-dump-toolexact.jsonl line 12 message 13 (sha b32d000fc11f, private, as are the replies); A886 is
that image padded to 886 x 922 PNG here with `ImageOps.pad`, as DeepSeek's CACHALOT_VISION_MIN_PIXELS does. The two rotate within each
round. Temperature 0.7, reasoning_effort none, no system prompt, no tools, non-streaming, Hamed's recorded words."""
from __future__ import annotations

import base64, hashlib, io, json, os, sys, time, urllib.request

from PIL import Image, ImageOps

URL = "http://127.0.0.1:8011/v1"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
PROMPT = "check this image and write me what is written inside the image"


def load():
    with open(os.path.expanduser("~/cachalot-dump-toolexact.jsonl")) as f:
        for i, l in enumerate(f):
            if i == 12:
                for part in json.loads(l)["body"]["messages"][13]["content"]:
                    if part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        got = hashlib.sha256(url.encode()).hexdigest()[:12]
                        if got != "b32d000fc11f":
                            sys.exit(f"image sha {got}")
                        return url
    sys.exit("image not found")


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


native = load()
urls = {"A492": native, "A886": padded(native, (886, 922))}
if not wait_ready():
    sys.exit("server not ready")
for k in range(int(sys.argv[1])):
    for tag in ("A492", "A886"):
        ask(urls[tag], tag, k)
print("driver done", flush=True)
