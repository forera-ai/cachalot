"""Requests for one server arm: warm-up, floor prompt x3, 12 mixed prompts, floor prompt x2 (greedy, 120 tokens)."""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:8011/v1"
FLOOR = "Write a detailed paragraph about how lighthouses work, from the lamp to the lens to the keeper's routine."
MIXED = [
    "Write a Python function that merges overlapping intervals and explain its complexity.",
    "Implement a thread-safe LRU cache in Go with a short usage example.",
    "Write a SQL query that finds the top three customers by revenue per region, with a window function.",
    "Write a TypeScript debounce helper with generics and a test.",
    "Write a Rust function that parses a CSV line with quoted fields.",
    "Describe a quiet morning in a mountain village in early autumn.",
    "Explain the causes of the French Revolution to a high-school student.",
    "Write a short story opening about a cartographer who finds a map of a city that does not exist.",
    "Summarize the main arguments for and against nuclear power.",
    "A train leaves at 9:40 and travels 210 km at 84 km/h. A second leaves at 10:05 at 105 km/h on the same track. When does the second catch up? Reason step by step.",
    "Is 391 prime? Reason carefully and show each check.",
    "Explain how a hash map handles collisions, then compare open addressing with chaining.",
]


def wait_ready(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(URL + "/models", timeout=5).read()
            return True
        except Exception:
            time.sleep(5)
    return False


def ask(text, tag):
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": text}],
            "max_tokens": 120, "temperature": 0, "stream": False}
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    print(json.dumps({"tag": tag, "wall_s": round(time.time() - t0, 3), "usage": out.get("usage")}), flush=True)


def stats(tag):
    out = json.loads(urllib.request.urlopen(URL + "/stats", timeout=30).read())
    print(json.dumps({"tag": tag, "stats": out}), flush=True)


if not wait_ready():
    sys.exit("server not ready")
ask("Say hello in five words.", "warmup")
stats("stats-after-warmup")
for i in range(3):
    ask(FLOOR, f"floor{i}")
for i, p in enumerate(MIXED):
    ask(p, f"mix{i:02d}")
for i in range(3, 5):
    ask(FLOOR, f"floor{i}")
stats("stats-end")
print("driver done", flush=True)
