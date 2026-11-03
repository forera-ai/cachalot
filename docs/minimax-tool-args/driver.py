"""Replays Hermes's "list the Desktop" tool request against a running server (MiniMax-M3 here, docs/minimax-tool-args/RECORD.md),
rotating three recorded versions of it, and records each reply's tool calls. Usage: driver.py <n_samples> [max_tokens]

The three requests are the turn that produced non-JSON `search_files` `limit` values live (HANDOFF 18.243, 18.254), read
from the local session dumps (they hold personal file names and are not in the repository; pinned here by line and by the
sha256 of their messages):
  P1  ~/cachalot-dump-di2.jsonl line 2   (2026-11-02 run 1, Hermes's MiniMax entry selected)  messages sha 012968ae424c5f70
  P2  ~/cachalot-dump-1017.jsonl line 3  (2026-11-01, 1017's first live session)               messages sha 8b36d0d55b893190
  P3  ~/cachalot-dump-di2.jsonl line 19  (2026-11-02 run 3)                                     messages sha 4ca01daf92e57ee6
Each is posted as Hermes sent it (temperature 0.7, its tools, reasoning_effort none), non-streaming, max_tokens 1,500 by default (MiniMax
may think before calling a tool). The request's model field names DeepSeek; the server answers with its own model."""
import hashlib, json, os, sys, time, urllib.request

URL = "http://127.0.0.1:8011/v1"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
SOURCES = [("P1", "cachalot-dump-di2.jsonl", 2, "012968ae424c5f70"),
           ("P2", "cachalot-dump-1017.jsonl", 3, "8b36d0d55b893190"),
           ("P3", "cachalot-dump-di2.jsonl", 19, "4ca01daf92e57ee6")]


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
    body = dict(body, stream=False, max_tokens=MAX_TOKENS)
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
                      "usage": out.get("usage"), "content": msg.get("content"), "reasoning": (msg.get("reasoning_content") or "")[:2000],
                      "tool_calls": msg.get("tool_calls")}), flush=True)


bodies = [(tag, load(name, line, sha)) for tag, name, line, sha in SOURCES]
if not wait_ready():
    sys.exit("server not ready")
for k in range(int(sys.argv[1])):
    tag, body = bodies[k % len(bodies)]
    ask(body, tag, k)
try:  # the server's counters (MiniMax reports its decode substitutions here), as proof of the arm's state
    print(json.dumps({"stats": json.loads(urllib.request.urlopen(URL + "/stats", timeout=30).read())}), flush=True)
except Exception as exc:
    print(json.dumps({"stats_error": repr(exc)}), flush=True)
print("driver done", flush=True)
