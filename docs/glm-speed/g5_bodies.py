"""
G5 (HANDOFF 18.279): build the blind replay's request bodies from one Hermes request dump.

The dump is Hermes's own record of the 2026-11-02 live session (Hamed's standard script, `reasoning_effort` none,
`extra_body.temperature` 0.7, 21 tools). Each body is that request cut just after one turn, so the model writes the
reply it wrote live: 0 the Desktop tool call, 1 the answer to the tool result, 2 a C# importer, 3 a 200-word story,
4 the Python importer, 5 the translation into German, French and Farsi. The output holds Hamed's conversation and file
names, so it is written under benchmarks/results/ (gitignored) and never committed.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python docs/glm-speed/g5_bodies.py DUMP OUT.jsonl
"""

import json
import sys
from pathlib import Path

# (label, prefix of the user message the cut follows, extra messages kept after it)
CUTS = (
    ("tool-call", "What files are in my home directory's Desktop folder?", 0),
    ("tool-result", "What files are in my home directory's Desktop folder?", 2),  # + the call and its result
    ("csharp", "write a C# code to import json and csv files", 0),
    ("story", "write a 200-word story about a lucky coin", 0),
    ("python", "Now write the same importer in Python.", 0),
    ("translate", "translate it to German, French and Farsi", 0),
)


def cut(messages: list[dict], prefix: str, extra: int) -> list[dict]:
    for i, m in enumerate(messages):
        if m["role"] == "user" and isinstance(m.get("content"), str) and m["content"].startswith(prefix):
            out = messages[: i + 1 + extra]
            if extra:
                assert out[-1]["role"] == "tool", f"cut after {prefix!r} does not end on a tool result"
            return out
    raise SystemExit(f"no user message starts with {prefix!r}")


def main() -> None:
    dump, out = Path(sys.argv[1]), Path(sys.argv[2])
    req = json.loads(dump.read_text())["request"]
    body = req.get("body", req)
    rows = []
    for label, prefix, extra in CUTS:
        b = {k: v for k, v in body.items() if k != "messages"}
        b["messages"] = cut(body["messages"], prefix, extra)
        rows.append({"label": label, "body": b})
        print(label, len(b["messages"]), "messages, last", b["messages"][-1]["role"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
