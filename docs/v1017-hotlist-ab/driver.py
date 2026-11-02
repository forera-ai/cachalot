"""One Hermes first turn against a fresh server: request 0 of the 1017 live-session dump (system block, 21 tools, the first user
message), non-streaming, greedy, at most 160 new tokens; `/v1/stats` before and after. Usage: driver.py <dump.jsonl>"""
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


def stats(tag):
    out = json.loads(urllib.request.urlopen(URL + "/stats", timeout=30).read())
    print(json.dumps({"tag": tag, "stats": out}), flush=True)


body = json.loads(open(sys.argv[1]).readline())["body"]
body.update({"stream": False, "temperature": 0, "max_tokens": 160})
body.pop("stream_options", None)
if not wait_ready():
    sys.exit("server not ready")
stats("stats-before")
req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
out = json.loads(urllib.request.urlopen(req, timeout=3600).read())
msg = out["choices"][0]["message"]
print(json.dumps({"tag": "first-turn", "wall_s": round(time.time() - t0, 3), "usage": out.get("usage"),
                  "finish": out["choices"][0].get("finish_reason"), "content": msg.get("content"),
                  "tool_calls": msg.get("tool_calls")}), flush=True)
stats("stats-after")
print("driver done", flush=True)
