"""Replays Hermes's hummingbird turn ("change the story to a small humming bird ... and translate that to these 3 languages too")
against a running server and records whether each reply starts a translation or stops after the English rewrite
(docs/v1017-translate-skip/RECORD.md). Usage: driver.py <replies per recorded copy>

The four recorded copies of the turn, read from the local session dumps (they hold personal file names and are not in the
repository; pinned by line and by the sha256 of their messages):
  H1  ~/cachalot-dump-toolexact.jsonl line 22  (2026-11-02, 0.76.0 live session; skipped the translations)  sha 1e8643d7ceaa1d81
  H2  ~/cachalot-dump-1017.jsonl line 32       (2026-11-01, first 1017 session)                              sha 2b51ab0ed9e01e7c
  H3  ~/cachalot-dump-1017.jsonl line 58       (2026-11-01, second 1017 session)                             sha 9a62acba52472d11
  H4  ~/cachalot-dump-di2.jsonl line 41        (2026-11-02, DI-2 session run 3)                              sha b93a9008fb32dcd9
Each is posted as Hermes sent it (temperature 0.7, its 21 tools, reasoning_effort none), streamed, max_tokens 1,500, copy-major
(all replies of H1, then H2, ...) so each conversation is prefilled once. The client reads the stream and disconnects as soon as
`started()` sees a translation begin (the server cancels the request), so a reply costs only the English part and the decision."""
from __future__ import annotations

import hashlib, json, os, re, sys, time, urllib.request

URL = "http://127.0.0.1:8011/v1"
SOURCES = [("H1", "cachalot-dump-toolexact.jsonl", 22, "1e8643d7ceaa1d81"),
           ("H2", "cachalot-dump-1017.jsonl", 32, "2b51ab0ed9e01e7c"),
           ("H3", "cachalot-dump-1017.jsonl", 58, "9a62acba52472d11"),
           ("H4", "cachalot-dump-di2.jsonl", 41, "b93a9008fb32dcd9")]
HEADING = re.compile(r"^\W*(German|Deutsch|French|Fran[cç]ais|Farsi|Persian|فارسی)\b", re.I | re.M)
PERSIAN = re.compile(r"[؀-ۿ]")
DE = re.compile(r"\b(und|der|die|das|ist|nicht|ein|eine|sich|mit|auf|dem|den)\b")
FR = re.compile(r"\b(le|les|et|une|est|dans|qui|des|du|sur|pas|il|elle)\b")


def started(text: str) -> str | None:
    """Which marker shows a translation has begun, or None."""
    if PERSIAN.search(text):
        return "persian"
    if HEADING.search(text):
        return "heading"
    if len(DE.findall(text)) >= 6:
        return "german-words"
    if len(FR.findall(text)) >= 8:
        return "french-words"
    return None


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
    body = dict(body, stream=True, max_tokens=1500)
    body.pop("stream_options", None)
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    text, finish, marker, tool, first = "", None, None, False, None
    try:
        resp = urllib.request.urlopen(req, timeout=3600)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ch = json.loads(line[5:])["choices"]
            if not ch:
                continue
            d = ch[0].get("delta") or {}
            if d.get("content"):
                if first is None:
                    first = round(time.time() - t0, 3)
                text += d["content"]
            if d.get("tool_calls"):
                tool = True
            if ch[0].get("finish_reason"):
                finish = ch[0]["finish_reason"]
            marker = started(text)
            if marker:
                resp.close()
                break
    except Exception as exc:
        print(json.dumps({"tag": tag, "k": k, "error": repr(exc), "text": text[-300:]}), flush=True)
        return
    print(json.dumps({"tag": tag, "k": k, "wall_s": round(time.time() - t0, 3), "first_s": first, "finish": finish, "marker": marker,
                      "tool": tool, "chars": len(text), "text": text}, ensure_ascii=False), flush=True)


bodies = [(tag, load(name, line, sha)) for tag, name, line, sha in SOURCES]
if not wait_ready():
    sys.exit("server not ready")
n = int(sys.argv[1])
for tag, body in bodies:
    for k in range(n):
        ask(body, tag, k)
print("driver done", flush=True)
