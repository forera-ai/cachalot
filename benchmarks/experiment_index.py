"""
Regenerate docs/EXPERIMENTS.md from the section-18 records of docs/HANDOFF.md (charter section 7.2 / 10).

Lists every `### 18.N` section from 18.75 on with its title, whether it has a **Prediction** paragraph and its own **Kind** line.
It does not interpret results. Run from the repository root:

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/experiment_index.py            # writes docs/EXPERIMENTS.md
    ~/venvs/deepseek-v41/bin/python benchmarks/experiment_index.py --check    # exit 1 if the file is out of date
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIRST = 75


def kind_of(text: str) -> str:
    t = text.lower()
    for k in ("research", "engineering", "measurement", "pricing", "planning"):
        if t.startswith(k):
            return k
    return "unlabelled"


def build(version: str) -> str:
    handoff = (REPO / "docs/HANDOFF.md").read_text()
    rows = []
    for p in re.split(r"\n(?=### 18\.\d+ )", handoff)[1:]:
        m = re.match(r"### (18\.(\d+)) (.*)", p.split("\n", 1)[0])
        if not m or int(m.group(2)) < FIRST:
            continue
        pred = re.search(r"\*\*Prediction[^*]*\*\*", p)
        kind = re.search(r"\*\*Kind\.\*\*\s*(.*)", p)
        title = re.sub(r" — 20\d\d-\d\d-\d\d \((\d+\.\d+\.\d+)\)$", r" (\1)", m.group(3))
        rows.append((int(m.group(2)), title, bool(pred), kind.group(1).strip() if kind else ""))
    rows.sort(reverse=True)
    counts: dict[str, int] = {}
    for r in rows:
        counts[kind_of(r[3])] = counts.get(kind_of(r[3]), 0) + 1
    out = [
        "# Experiment index (charter section 7.2 and section 10)\n",
        f"Written 2026-10-09 against runtime {version}; regenerate with `benchmarks/experiment_index.py`. The charter asks for \"a file of experiment records whose predictions were written before the build\". The records themselves live in `docs/HANDOFF.md` section 18 (each with Baseline, Prediction, Method, Result, Explanation, Limits, Kind); this file is the index: where each record is, whether it carries a written prediction, and what kind of result it is, as its own **Kind** line says. It does not restate results. Sections before 18.75 predate the record format and are not listed; sections without a **Kind** line are listed as unlabelled and have no claim made for them here.\n",
        f"Counts of the {len(rows)} sections listed (18.{FIRST}-18.{rows[0][0]}): " + ", ".join(f"{v} {a}" for a, v in sorted(counts.items(), key=lambda x: -x[1])) + ".\n",
        "Research means predicted, measured and explained (charter section 8.1 rule 6); a falsified prediction is still research and is marked so in its Kind line. Engineering means it works without a written prediction, or an instrument was built.\n",
        "\"Prediction written\" means the section has a **Prediction** paragraph; whether it was written before the run is what that section says (the 0.62.23 and 0.62.25 ones were, in the earlier section they cite). Only the Kind column is the record's own verdict.\n",
        "| section | title | prediction written | kind (as recorded) |",
        "|---|---|:---:|---|",
    ]
    for n, t, pr, kd in rows:
        out.append(f"| §18.{n} | {t} | {'yes' if pr else 'no'} | {kd or 'not labelled'} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    version = re.search(r'__version__ = "([^"]+)"', (REPO / "src/cachalot/__init__.py").read_text()).group(1)
    text = build(version)
    target = REPO / "docs/EXPERIMENTS.md"
    if args.check:
        sys.exit(0 if target.exists() and target.read_text() == text else 1)
    target.write_text(text)
    print(f"wrote {target} ({text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
