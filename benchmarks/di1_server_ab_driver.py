"""DI-1 server A/B driver (docs/DI1-SERVER-AB-RECORD.md): replay the live session's first four request bodies against a running ./serve.sh,
each with max_tokens 1, temperature 0 and stream false, and record the server's prefill numbers. Run by benchmarks/di1_server_ab.sh after the
server answers /v1/models (the system Python, so serve.sh's one-runtime guard never sees it). A watchdog samples memory pressure and swap every
2 s and SIGINTs the server on the record's stop rules. --score reads the four arm files and prints the record's quantities.

    /usr/bin/env python3 -I benchmarks/di1_server_ab_driver.py --dump ~/cachalot-di1-live.jsonl --out docs/di1-server-ab/arm-1-off.json \
        --arm 1 --knob 0 --server-pid 12345
    /usr/bin/env python3 -I benchmarks/di1_server_ab_driver.py --score docs/di1-server-ab
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("DI1_AB_URL", "http://127.0.0.1:8011/v1")
REQUESTS = 4                   # the dump's first four request bodies: the cold system block, a short turn, two large tool results
CRITICAL = 4                   # kern.memorystatus_vm_pressure_level
CRITICAL_SAMPLES = 2
SWAP_GROWTH_MB = 512


def bodies(dump: Path, n: int = REQUESTS) -> list[dict]:
    """The first n request bodies of a CACHALOT_SERVER_DUMP file (its other lines are the server's reply records)."""
    out = []
    for line in dump.read_text().splitlines():
        rec = json.loads(line)
        if "body" in rec:
            out.append(rec["body"])
            if len(out) == n:
                break
    return out


def prepare(body: dict) -> dict:
    """The replayed body: the same messages, tools and reasoning effort; one greedy token, not streamed."""
    b = {k: v for k, v in body.items() if k not in ("stream", "stream_options", "max_tokens", "max_completion_tokens", "temperature")}
    b.update({"max_tokens": 1, "temperature": 0, "stream": False})
    return b


def swap_used_mb(text: str) -> float:
    m = re.search(r"used = ([0-9.]+)M", text)
    return float(m.group(1)) if m else 0.0


def sample() -> tuple[int, float]:
    level = int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True).stdout.strip() or 0)
    swap = swap_used_mb(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)
    return level, swap


class Watchdog(threading.Thread):
    """Samples pressure and swap every period; on a stop rule records why and SIGINTs the server."""

    def __init__(self, server_pid: int | None, period: float = 2.0, sampler=sample):
        super().__init__(daemon=True)
        self.server_pid, self.period, self.sampler = server_pid, period, sampler
        self.stop = threading.Event()
        self.tripped: str | None = None
        level, swap = sampler()
        self.swap0, self.swap_max, self.level_max, self._critical = swap, swap, level, 0

    def check(self) -> None:
        level, swap = self.sampler()
        self.level_max, self.swap_max = max(self.level_max, level), max(self.swap_max, swap)
        self._critical = self._critical + 1 if level >= CRITICAL else 0
        if self._critical >= CRITICAL_SAMPLES:
            self.tripped = f"critical pressure on {CRITICAL_SAMPLES} samples"
        elif swap - self.swap0 > SWAP_GROWTH_MB:
            self.tripped = f"swap grew {swap - self.swap0:.0f} MB"
        if self.tripped and self.server_pid:
            try:
                os.kill(self.server_pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    def run(self) -> None:
        while not self.stop.wait(self.period) and not self.tripped:
            self.check()

    def summary(self) -> dict:
        return {"tripped": self.tripped, "pressure_max": self.level_max, "swap_start_mb": self.swap0, "swap_max_mb": self.swap_max}


def post(body: dict, url: str = URL, timeout: float = 1800) -> dict:
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def run(dump: Path, out: Path, arm: int, knob: int, server_pid: int | None, url: str = URL, sampler=sample) -> int:
    dog = Watchdog(server_pid, sampler=sampler)
    dog.start()
    rec = {"arm": arm, "knob": knob, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "requests": []}
    rc = 0
    try:
        for i, body in enumerate(bodies(dump), start=1):
            t0 = time.time()
            res = post(prepare(body), url)
            cx = res["usage"].get("cachalot", {})
            prompt, reused = res["usage"]["prompt_tokens"], cx.get("reused_prefix_tokens")
            msg = res["choices"][0]["message"]
            rec["requests"].append({"request": i, "prompt_tokens": prompt, "reused": reused,
                                    "prefilled": prompt - reused if reused is not None else None,
                                    "prefill_s": cx.get("prefill_seconds"), "wall_s": round(time.time() - t0, 3),
                                    "completion_tokens": res["usage"]["completion_tokens"],
                                    "token": (msg.get("content") or "") + (msg.get("reasoning_content") or ""),
                                    "tool_calls": bool(msg.get("tool_calls"))})
            print(json.dumps(rec["requests"][-1]), flush=True)
            rec["watchdog"] = dog.summary()
            out.write_text(json.dumps(rec, indent=1))
            if dog.tripped:
                rc = 3
                break
    except Exception as e:                                    # a request error is a stop rule: record it, do not continue
        rec["error"] = repr(e)
        rc = 2
    finally:
        dog.stop.set()
        dog.join(timeout=5)
        rec["watchdog"] = dog.summary()
        rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        out.write_text(json.dumps(rec, indent=1))
    return rc


def score(d: Path) -> dict:
    arms = [json.loads(p.read_text()) for p in sorted(d.glob("arm-*.json"))]
    by = {0: [a for a in arms if a["knob"] == 0], 1: [a for a in arms if a["knob"] == 1]}

    def mean(k: int, r: int) -> float | None:
        v = [q["prefill_s"] for a in by[k] for q in a["requests"] if q["request"] == r]
        return sum(v) / len(v) if v else None

    def ratio(r: int) -> float | None:
        on, off = mean(1, r), mean(0, r)
        return round(on / off, 3) if on is not None and off else None

    out: dict = {"arms": [{"arm": a["arm"], "knob": a["knob"], "watchdog": a.get("watchdog"), "error": a.get("error"),
                           "prefill_s": [q["prefill_s"] for q in a["requests"]], "prefilled": [q["prefilled"] for q in a["requests"]],
                           "tokens": [q["token"] for q in a["requests"]]} for a in arms]}
    out["mean_prefill_s"] = {("on" if k else "off"): [None if mean(k, r) is None else round(mean(k, r), 2) for r in range(1, REQUESTS + 1)]
                             for k in (0, 1)}
    out["ratio"] = {f"request {r}": ratio(r) for r in range(1, REQUESTS + 1)}
    s = {k: [mean(k, r) for r in (1, 3, 4)] for k in (0, 1)}
    out["ratio"]["requests 1+3+4"] = round(sum(s[1]) / sum(s[0]), 3) if None not in s[0] + s[1] else None
    for k, name in ((0, "off"), (1, "on")):
        v = [a["requests"][0]["prefill_s"] for a in by[k] if a["requests"]]
        out[f"request 1 spread {name}"] = round(abs(v[0] - v[1]) / min(v), 3) if len(v) == 2 else None
    pre = [tuple(q["prefilled"] for q in a["requests"]) for a in arms]
    tok = [tuple(q["token"] for q in a["requests"]) for a in arms]
    out["prefilled equal across arms"] = [len({p[r] for p in pre if len(p) > r}) == 1 for r in range(REQUESTS)]
    out["token equal across arms"] = [len({t[r] for t in tok if len(t) > r}) == 1 for r in range(REQUESTS)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--arm", type=int)
    ap.add_argument("--knob", type=int, choices=(0, 1))
    ap.add_argument("--server-pid", type=int)
    ap.add_argument("--score", type=Path)
    a = ap.parse_args()
    if a.score:
        print(json.dumps(score(a.score), indent=1, ensure_ascii=False))
        return
    raise SystemExit(run(a.dump.expanduser(), a.out, a.arm, a.knob, a.server_pid))


if __name__ == "__main__":
    main()
