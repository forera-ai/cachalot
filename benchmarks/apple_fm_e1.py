"""
E-1 (docs/APPLE-FM-E1-RECORD.md): DeepSeek's server decode with and without auxiliary classification calls beside it.

Three roles. `drive` and `score` use only the standard library and run under the system python3, so the
runtime guards never see them; `load` runs the auxiliary model and needs the runtime venv (apple-fm-sdk).
The whole arm is `benchmarks/apple_fm_e1.sh`; the roles are not meant to be started by hand.

    load   --log F [--seconds S | --control F --status F]   classification calls back to back
    drive  --out DIR --python VENV_PYTHON [--cycles 6]      warm-up, then A B B A x cycles against the server
    score  --out DIR --server-log F                         the arm table and the E1-* scores
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BASE = "http://127.0.0.1:8011"
PROMPT = ("Explain in about 150 words how a lighthouse warns ships at night: the light source, the lens, "
          "the rotation and how sailors tell one lighthouse from another.")
MAX_TOKENS = 128


# ------------------------------------------------------------------------------------------------ load
def cmd_load(args) -> None:
    sys.path.insert(0, str(REPO / "src"))
    from cachalot.auxiliary.apple_fm import AppleFoundationModelsProvider
    from cachalot.auxiliary.provider import GenerationParams
    from cachalot.auxiliary.tasks import CLASSIFY_INSTRUCTIONS, CLASSIFY_SCHEMA

    texts = [json.loads(line)["text"] for line in (HERE / "apple_fm" / "classify.jsonl").read_text().splitlines()]
    p = AppleFoundationModelsProvider(timeout_s=30)
    if not p.is_available():
        sys.exit(f"system model unavailable: {p.capabilities().unavailable_reason}")
    params = GenerationParams(max_tokens=64)
    p.generate_structured("Request:\nhi\n", CLASSIFY_SCHEMA, instructions=CLASSIFY_INSTRUCTIONS, params=params)
    log = open(args.log, "a")
    control = Path(args.control) if args.control else None
    status = Path(args.status) if args.status else None
    t_end = time.time() + args.seconds if args.seconds else None
    i = 0

    def set_status(s: str) -> None:
        if status is not None:
            tmp = status.with_suffix(".tmp")
            tmp.write_text(s)
            os.replace(tmp, status)

    while True:
        if t_end is not None:
            if time.time() >= t_end:
                break
            state = "on"
        else:
            state = control.read_text().strip() if control.exists() else "off"
        if state == "quit":
            break
        if state != "on":
            set_status("idle")
            time.sleep(0.02)
            continue
        set_status("busy")
        text = texts[i % len(texts)]
        i += 1
        t0 = time.time()
        err = None
        try:
            p.generate_structured(f"Request:\n{text}\n", CLASSIFY_SCHEMA, instructions=CLASSIFY_INSTRUCTIONS,
                                  params=params)
        except Exception as exc:  # recorded, the load keeps going
            err = f"{type(exc).__name__}: {exc}"[:200]
        t1 = time.time()
        log.write(json.dumps({"t0": t0, "t1": t1, "latency_s": t1 - t0, "item": i, "error": err}) + "\n")
        log.flush()
    set_status("idle")
    p.close()


# ----------------------------------------------------------------------------------------------- drive
def _post(path: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _wait_health(timeout: float) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit(f"server not healthy after {timeout:.0f} s")


def _wait_status(status: Path, want: str, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if status.exists() and status.read_text().strip() == want:
            return True
        time.sleep(0.02)
    return False


def cmd_drive(args) -> None:
    out = Path(args.out)
    control, status = out / "aux-control", out / "aux-status"
    control.write_text("off")
    ready_s = _wait_health(args.health_timeout)
    print(f"server healthy after {ready_s:.0f} s", flush=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    load = subprocess.Popen([args.python, str(Path(__file__)), "load", "--log", str(out / "aux-during.jsonl"),
                             "--control", str(control), "--status", str(status)], env=env)
    if not _wait_status(status, "idle", 120):
        load.kill()
        raise SystemExit("auxiliary load did not start")
    order = ["warm", "warm"] + ["A", "B", "B", "A"] * args.cycles
    prompts = [line.strip() for line in Path(args.prompts).read_text().splitlines() if line.strip()] if args.prompts else None
    if prompts is not None and len(prompts) < len(order):
        raise SystemExit(f"{args.prompts}: {len(prompts)} prompts for {len(order)} replies")
    rows = open(out / "replies.jsonl", "a")
    try:
        for n, arm in enumerate(order):
            if arm == "B":
                control.write_text("on")
                if not _wait_status(status, "busy", 10):
                    raise SystemExit("auxiliary load did not go busy")
                time.sleep(0.5)          # the first call is past its start-up and into the model
            else:
                control.write_text("off")
                if not _wait_status(status, "idle", 30):
                    raise SystemExit("auxiliary load did not go idle")
            prompt = prompts[n] if prompts is not None else PROMPT
            body = {"model": args.model_id, "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "max_tokens": args.max_tokens, "thinking": False, "stream": False}
            t0 = time.time()
            r = _post("/v1/chat/completions", body)
            t1 = time.time()
            control.write_text("off")
            u = r["usage"]
            c = u.get("cachalot", {})
            text = r["choices"][0]["message"].get("content") or ""
            row = {"n": n, "arm": arm, "cycle": (n - 2) // 4 if arm != "warm" else None, "t0": t0, "t1": t1,
                   "wall_s": t1 - t0, "completion_tokens": u["completion_tokens"], "prompt_tokens": u["prompt_tokens"],
                   "reused": c.get("reused_prefix_tokens"), "prefill_s": c.get("prefill_seconds"),
                   "decode_s": c.get("decode_seconds"),
                   "ms_token": 1000 * c["decode_seconds"] / u["completion_tokens"] if u["completion_tokens"] else None,
                   "sha": hashlib.sha256(text.encode()).hexdigest()[:16]}
            rows.write(json.dumps(row) + "\n")
            rows.flush()
            print(f"{n:2d} {arm:4s} {row['ms_token']:.1f} ms/token  {row['completion_tokens']} tok  {row['sha']}", flush=True)
    finally:
        control.write_text("quit")
        try:
            load.wait(timeout=60)
        except subprocess.TimeoutExpired:
            load.kill()


# ----------------------------------------------------------------------------------------------- score
def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def cmd_score(args) -> None:
    out = Path(args.out)
    replies = [r for r in _jsonl(out / "replies.jsonl") if r["arm"] != "warm"]
    lines = [ln for ln in Path(args.server_log).read_text().splitlines() if ln.startswith("[request] prompt=")]
    misses, reads = [], []
    for ln in lines:
        m = [tok for tok in ln.split() if tok.startswith("miss/tok=")]
        misses.append(float(m[0].split("=")[1]) if m else None)
        rd = [tok for tok in ln.split() if tok.startswith("read=")]
        reads.append(float(rd[0].split("=")[1].rstrip("ms")) if rd else None)
    all_rows = _jsonl(out / "replies.jsonl")
    if len(misses) == len(all_rows):
        for r, m, rd in zip(all_rows, misses, reads):  # noqa: B905 - runs under the system python3 (3.9)
            r["miss_tok"] = m
            r["read_ms"] = rd
    by = {a: [r for r in replies if r["arm"] == a] for a in ("A", "B")}
    med = {a: statistics.median(r["ms_token"] for r in rs) for a, rs in by.items()}
    cycles = sorted({r["cycle"] for r in replies})
    diffs, ratios = [], []
    for c in cycles:
        a = [r["ms_token"] for r in by["A"] if r["cycle"] == c]
        b = [r["ms_token"] for r in by["B"] if r["cycle"] == c]
        diffs.append(statistics.mean(b) - statistics.mean(a))
        ratios.append(statistics.mean(b) / statistics.mean(a))
    shas = {r["sha"] for r in replies}
    pre = [x["latency_s"] for x in _jsonl(out / "aux-pre.jsonl") if not x["error"]]
    post = [x["latency_s"] for x in _jsonl(out / "aux-post.jsonl") if not x["error"]]
    during_all = _jsonl(out / "aux-during.jsonl")
    b_windows = [(r["t0"], r["t1"]) for r in by["B"]]
    during = [x["latency_s"] for x in during_all if not x["error"] and any(t0 <= x["t0"] and x["t1"] <= t1 for t0, t1 in b_windows)]
    errors = sum(1 for x in during_all if x["error"])
    alone = pre + post
    miss = {a: [r.get("miss_tok") for r in all_rows if r["arm"] == a and r.get("miss_tok") is not None] for a in ("A", "B")}
    summary = {
        "replies": {a: len(rs) for a, rs in by.items()},
        "ms_token_median": med,
        "ms_token": {a: sorted(round(r["ms_token"], 2) for r in rs) for a, rs in by.items()},
        "b_minus_a_by_cycle": [round(d, 2) for d in diffs],
        "b_minus_a_median": statistics.median(diffs), "b_slower_cycles": sum(d > 0 for d in diffs),
        "b_over_a_by_cycle": [round(x, 4) for x in ratios], "b_over_a_median": statistics.median(ratios),
        "read_ms_mean": {a: (statistics.mean(v) if v else None) for a, v in
                         {a: [r.get("read_ms") for r in all_rows if r["arm"] == a and r.get("read_ms") is not None]
                          for a in ("A", "B")}.items()},
        "distinct_texts": len(shas), "completion_tokens": sorted({r["completion_tokens"] for r in replies}),
        "miss_tok_mean": {a: (statistics.mean(v) if v else None) for a, v in miss.items()},
        "aux_alone_median_s": statistics.median(alone) if alone else None, "aux_alone_n": len(alone),
        "aux_during_b_median_s": statistics.median(during) if during else None, "aux_during_b_n": len(during),
        "aux_errors": errors,
        "aux_change": (statistics.median(during) / statistics.median(alone) - 1) if during and alone else None,
        "server_request_lines": len(lines),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("load")
    s.add_argument("--log", required=True)
    s.add_argument("--seconds", type=float, default=None)
    s.add_argument("--control", default=None)
    s.add_argument("--status", default=None)
    s = sub.add_parser("drive")
    s.add_argument("--out", required=True)
    s.add_argument("--python", required=True)
    s.add_argument("--cycles", type=int, default=6)
    s.add_argument("--health-timeout", type=float, default=900)
    s.add_argument("--model-id", default="deepseek-v4.1-flash")
    s.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    s.add_argument("--prompts", default=None, help="one prompt a line, reply n uses line n (keeps a read-bound model cold)")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--server-log", required=True)
    args = ap.parse_args()
    {"load": cmd_load, "drive": cmd_drive, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
