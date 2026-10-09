"""Energy arms driver (HANDOFF 18.116): after the server is ready, four phases with wall-clock times logged for powermetrics_parse.py.
A idle-server 90 s, B all-resident decode (one prompt repeated, greedy 120 tokens), C read-bound decode (fresh prompts), D idle-server 90 s."""
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
    "Explain how a hash map handles collisions, then compare open addressing with chaining.",
]
def now(): return time.strftime("%H:%M:%S")
def ask(text):
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": text}], "max_tokens": 120, "temperature": 0, "stream": False}
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return out["usage"]["completion_tokens"], out["usage"].get("cachalot", {})
def log(rec): print(json.dumps(rec), flush=True)
t0 = time.time()
while time.time() - t0 < 900:
    try: urllib.request.urlopen(URL + "/models", timeout=5).read(); break
    except Exception: time.sleep(3)
log({"event": "ready", "clock": now()})
ask("Say hello in five words."); ask(FLOOR); ask(FLOOR)  # warm: the floor prompt's experts are resident now
log({"event": "warm", "clock": now()})
def phase(name, fn):
    s = now(); tok = fn(); e = now(); log({"event": "phase", "name": name, "start": s, "end": e, "tokens": tok})
def idle(sec):
    def f(): time.sleep(sec); return 0
    return f
def resident(sec):
    def f():
        tok, t = 0, time.time()
        while time.time() - t < sec: tok += ask(FLOOR)[0]
        return tok
    return f
def readbound(sec):
    def f():
        tok, t = 0, time.time()
        for p in MIXED:
            if time.time() - t > sec: break
            tok += ask(p)[0]
        return tok
    return f
phase("A idle-server", idle(90)); phase("B all-resident", resident(120)); phase("C read-bound", readbound(150)); phase("D idle-server", idle(90))
log({"event": "stats", "stats": json.loads(urllib.request.urlopen(URL + "/stats", timeout=30).read())})
log({"event": "done", "clock": now()})
