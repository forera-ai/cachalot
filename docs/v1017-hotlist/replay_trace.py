"""Step 7: replay each 1017 live session's Hermes requests through serve.sh with a routing trace (one server per session, an empty
snapshot directory so the system block is traced too, replies capped at 256 tokens). The trace is written when the server stops."""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/Users/hamedprooshani/Projects/deepseek-v41-mac")
OUT = Path(sys.argv[1])
DUMP = Path.home() / "cachalot-dump-1017.jsonl"
URL = "http://127.0.0.1:8011/v1"
SPLIT = 1793530000  # session 1 ran 02:07-02:41, session 2 12:36-13:04 (2026-11-01)


def wait_ready(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(URL + "/models", timeout=2).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def post(body):
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0, finish, err = time.time(), None, None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                err = ev.get("error", err)
                for ch in ev.get("choices", []):
                    finish = ch.get("finish_reason") or finish
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    return {"finish": finish, "error": err, "seconds": round(time.time() - t0, 1)}


def stop(srv):
    """Ctrl-C the server process itself, not guarded_run's group: guarded_run answers an interrupt by killing its child at once,
    before uvicorn's exit hook writes the routing trace."""
    pids = subprocess.run(["pgrep", "-g", str(srv.pid), "-f", r"cachalot\.cli serve"], capture_output=True, text=True).stdout.split()
    for pid in pids:
        os.kill(int(pid), signal.SIGINT)
    try:
        srv.wait(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(srv.pid, signal.SIGINT)
        srv.wait()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in DUMP.read_text().splitlines()]
    bodies = [r for r in rows if "body" in r]
    sessions = {"s1": [r["body"] for r in bodies if r["t"] < SPLIT], "s2": [r["body"] for r in bodies if r["t"] >= SPLIT]}
    for name, reqs in sessions.items():
        trace = Path.home() / f"cachalot-routing-1017-{name}.trace.npz"
        if trace.exists():
            print(f"TRACE {name}: trace already on disk, skipped", flush=True)
            continue
        snap = OUT / f"snapshots-{name}"
        snap.mkdir(exist_ok=True)
        env = dict(os.environ, CACHALOT_MODEL_PATH="/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-1017", CACHALOT_SNAPSHOT_DIR=str(snap),
                   CACHALOT_ROUTING_TRACE=str(trace), CACHALOT_SERVER_DUMP=str(OUT / f"dump-{name}.jsonl"))
        srv = subprocess.Popen(["benchmarks/guarded_run.sh", "--budget-gib", "48", "--max-seconds", "1800", "--tag", f"trace1017_{name}",
                                "--force", "--", "./serve.sh"], cwd=REPO, env=env, stdout=open(OUT / f"server-{name}.log", "w"),
                               stderr=subprocess.STDOUT, start_new_session=True)
        if not wait_ready():
            print(f"TRACE {name}: server not ready", flush=True)
            os.killpg(srv.pid, signal.SIGINT)
            srv.wait()
            sys.exit(1)
        print(f"TRACE {name}: ready, {len(reqs)} requests", flush=True)
        for k, body in enumerate(reqs):
            body = dict(body, max_tokens=256)
            body.pop("max_completion_tokens", None)
            res = post(body)
            print(f"TRACE {name} {k + 1}/{len(reqs)} {res}", flush=True)
        stop(srv)
        for _ in range(120):
            if trace.exists():
                break
            time.sleep(1)
        print(f"TRACE {name}: stopped, trace {'written' if trace.exists() else 'MISSING'}", flush=True)


if __name__ == "__main__":
    main()
