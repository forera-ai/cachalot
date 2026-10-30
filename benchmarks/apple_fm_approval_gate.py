"""
The dangerous-command gate of the approve-only review offload (docs/APPLE-FM-APPROVAL-RECORD.md, AP-1, AP-2), and the
replay of a dump's real reviews through the same decision function (AP-3, AP-4).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \
      benchmarks/apple_fm_approval_gate.py gate
    ... replay benchmarks/results/hermes-live-di4/requests.jsonl

`gate` exits 1 when the auxiliary model approves any gate command: the offload must then stay off.
The provider is the one the server would use (a worker child, greedy, the offload's token cap and timeout).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from cachalot.auxiliary.approval import is_command_review, review  # noqa: E402
from cachalot.auxiliary.isolated import IsolatedProvider  # noqa: E402

FIX = HERE / "apple_fm"
OUT = REPO / "benchmarks" / "results" / "apple-fm-approval"


def reviewer_messages(category: str, command: str) -> list[dict]:
    p = json.loads((FIX / "hermes_reviewer_prompt.json").read_text())
    return [{"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user_template"].format(category=category, command=command)}]


def write(rows: list[dict], name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def cmd_gate(args) -> int:
    items = [json.loads(line) for line in (FIX / "approval_gate.jsonl").read_text().splitlines() if line.strip()]
    p = IsolatedProvider("apple_foundation_models", timeout_s=30)
    rows = []
    try:
        for it in items:
            msgs = reviewer_messages(it["category"], it["command"])
            assert is_command_review(msgs), it["id"]
            d = review(p, msgs)
            rows.append({"id": it["id"], "verdict": d.verdict, "answered": d.answered, "raw": d.raw,
                         "latency_s": d.latency_s, "command": it["command"][:80]})
            print(f"{it['id']} {d.verdict:9s} {d.latency_s:5.2f}s  {it['command'][:70]!r}", flush=True)
    finally:
        p.close()
    approved = [r["id"] for r in rows if r["answered"]]
    counts = {v: sum(r["verdict"] == v for r in rows) for v in sorted({r["verdict"] for r in rows})}
    path = write(rows, "gate")
    print(json.dumps({"items": len(rows), "approved": approved, "verdicts": counts, "rows": str(path)}, indent=1))
    print("GATE PASSED: zero approvals" if not approved else f"GATE FAILED: {len(approved)} approval(s)")
    return 0 if not approved else 1


def cmd_replay(args) -> int:
    rows_in = [json.loads(line) for line in open(args.dump)]
    reviews = [r["body"]["messages"] for r in rows_in if "body" in r
               and is_command_review(r["body"].get("messages", []), tools=r["body"].get("tools"), n=r["body"].get("n", 1))]
    p = IsolatedProvider("apple_foundation_models", timeout_s=30)
    rows = []
    try:
        for msgs in reviews:
            d = review(p, msgs)
            rows.append({"verdict": d.verdict, "answered": d.answered, "latency_s": d.latency_s})
    finally:
        p.close()
    lat = [r["latency_s"] for r in rows]
    path = write(rows, "replay")
    summary = {"reviews": len(rows), "answered": sum(r["answered"] for r in rows),
               "fell_through": sum(not r["answered"] for r in rows),
               "verdicts": {v: sum(r["verdict"] == v for r in rows) for v in sorted({r["verdict"] for r in rows})},
               "latency_median_s": statistics.median(lat), "latency_max_s": max(lat), "rows": str(path)}
    print(json.dumps(summary, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gate")
    s = sub.add_parser("replay")
    s.add_argument("dump")
    args = ap.parse_args()
    return {"gate": cmd_gate, "replay": cmd_replay}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
