"""
Parse `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -o FILE` output (charter L5, HANDOFF 18.115).

One sample is a block that starts `*** Sampled system activity (<time>) (<ms> elapsed) ***` and holds the lines `CPU Power: N mW`,
`GPU Power: N mW`, `ANE Power: N mW`, `Combined Power (CPU + GPU + ANE): N mW`. Reported: watts by component (mean, median, p95),
energy in joules (power x the sample's own elapsed time), and joules per token when the number of tokens decoded in the window is
given. Memory and drive power are NOT reported by powermetrics on this platform: the totals are CPU + GPU + ANE package power,
not wall power, and any figure for DRAM or the SSD is an estimate to be labelled as one (charter section 8.1 rule 11).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/powermetrics_parse.py benchmarks/results/energy/idle-60s.txt
    ~/venvs/deepseek-v41/bin/python benchmarks/powermetrics_parse.py FILE --from 22:31:05 --to 22:34:05 --tokens 120

`--from`/`--to` are local clock times (HH:MM:SS) of the samples to keep, so a decode arm is cut out of a longer recording using the
server's request timestamps; the first sample of a recording is dropped (it covers startup).
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from dataclasses import dataclass

HEAD = re.compile(r"\*\*\* Sampled system activity \((.*?)\) \(([\d.]+)ms elapsed\) \*\*\*")
POWER = re.compile(r"^(CPU|GPU|ANE) Power: ([\d.]+) mW|^Combined Power \(CPU \+ GPU \+ ANE\): ([\d.]+) mW", re.M)


@dataclass
class Sample:
    clock: str  # HH:MM:SS
    elapsed_ms: float
    cpu_mw: float = 0.0
    gpu_mw: float = 0.0
    ane_mw: float = 0.0
    combined_mw: float = 0.0


def parse(text: str) -> list[Sample]:
    out: list[Sample] = []
    parts = HEAD.split(text)  # [pre, time, ms, body, time, ms, body, ...]
    for i in range(1, len(parts) - 2, 3):
        m = re.search(r"(\d\d:\d\d:\d\d)", parts[i])
        s = Sample(m.group(1) if m else "", float(parts[i + 1]))
        for pm in POWER.finditer(parts[i + 2]):
            if pm.group(1) == "CPU":
                s.cpu_mw = float(pm.group(2))
            elif pm.group(1) == "GPU":
                s.gpu_mw = float(pm.group(2))
            elif pm.group(1) == "ANE":
                s.ane_mw = float(pm.group(2))
            else:
                s.combined_mw = float(pm.group(3))
        out.append(s)
    return out


def _p95(v: list[float]) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(0.95 * len(v)))]


def summarize(samples: list[Sample], tokens: int | None = None) -> dict:
    if not samples:
        raise ValueError("no samples in the window")
    secs = [s.elapsed_ms / 1000 for s in samples]
    res = {"samples": len(samples), "seconds": sum(secs)}
    for name in ("cpu", "gpu", "ane", "combined"):
        mw = [getattr(s, f"{name}_mw") for s in samples]
        joules = sum(p / 1000 * t for p, t in zip(mw, secs))
        res[name] = {"mean_w": joules / sum(secs), "median_w": st.median(mw) / 1000, "p95_w": _p95(mw) / 1000, "joules": joules}
    if tokens:
        res["tokens"] = tokens
        res["joules_per_token"] = res["combined"]["joules"] / tokens
        res["tokens_per_joule"] = tokens / res["combined"]["joules"]
    return res


def window(samples: list[Sample], start: str | None, end: str | None) -> list[Sample]:
    keep = samples[1:]  # the first sample covers startup
    if start:
        keep = [s for s in keep if s.clock >= start]
    if end:
        keep = [s for s in keep if s.clock <= end]
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end")
    ap.add_argument("--tokens", type=int, default=None, help="tokens decoded inside the window (adds joules per token)")
    args = ap.parse_args()
    samples = parse(open(args.file).read())
    kept = window(samples, args.start, args.end)
    r = summarize(kept, args.tokens)
    print(f"{args.file}: {len(samples)} samples, {r['samples']} kept ({r['seconds']:.1f} s)")
    print(f"{'component':10s} {'mean W':>8s} {'median W':>9s} {'p95 W':>7s} {'joules':>9s}")
    for name in ("cpu", "gpu", "ane", "combined"):
        c = r[name]
        print(f"{name:10s} {c['mean_w']:8.2f} {c['median_w']:9.2f} {c['p95_w']:7.2f} {c['joules']:9.1f}")
    if "joules_per_token" in r:
        print(f"\n{r['tokens']} tokens: {r['joules_per_token']:.2f} J a token ({r['tokens_per_joule']:.3f} tokens a joule), CPU+GPU+ANE package power only")
    print("\nNote: powermetrics does not report DRAM or SSD power; these are package totals, not wall power.")


if __name__ == "__main__":
    main()
