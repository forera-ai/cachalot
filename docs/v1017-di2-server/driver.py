"""Two cold prefills against a fresh server: request 0 of the 1017 live-session dump (Hermes system block, 21 tools, ~16k tokens) and a
mid-size prompt (docs/v1017-di2-server/mid_prompt.txt, ~4.4k tokens, no tools). Greedy, non-streaming, 16 new tokens each; the server's
[request] lines carry the prefill times. Usage: driver.py <dump.jsonl> <mid_prompt.txt>"""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:8011/v1"


def wait_ready(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(URL + "/models", timeout=5).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def ask(body, tag):
    body = dict(body, stream=False, temperature=0, max_tokens=16)
    body.pop("stream_options", None)
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    msg = out["choices"][0]["message"]
    print(json.dumps({"tag": tag, "wall_s": round(time.time() - t0, 3), "usage": out.get("usage"), "content": msg.get("content"),
                      "tool_calls": msg.get("tool_calls")}), flush=True)


if not wait_ready():
    sys.exit("server not ready")
ask(json.loads(open(sys.argv[1]).readline())["body"], "hermes-block")
ask({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": open(sys.argv[2]).read()}]}, "mid-prompt")
print("driver done", flush=True)
