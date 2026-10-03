"""
Pareto harness: speed against quality for any DeepSeek configuration ("arm"), judged the way HANDOFF 18.50-18.56
says output-changing levers must be (docs/SPEED-RESEARCH-2026-10-03.md section 5, MEASUREMENT.md "Quality gate").

One arm is a named configuration: environment knobs, the expert budget, an optional per-layer decode miss budget
(`V41Model.set_decode_miss_budget`, the only output-changing decode knob that exists today) and an optional prefill
chunk size. A run loads the model once under that arm and measures, in one process:

  1. Teacher-forced decode on fixed texts (prose, code, markdown): per position the log-probability of the true
     token, the arm's log-probabilities at the reference arm's top-64 tokens (for KL), the arm's argmax, the wall
     time of the decode step and the expert misses it caused. Prefill runs once per text, then every position is a
     decode step, because decode is where substitution and drops act.
  2. A battery of checkable tasks (`pareto_tasks.json`: exact answers, Python functions run against asserts, JSON,
     C# that must `dotnet build`), generated greedily from a fresh context each.

`report` compares every arm with the reference arm: paired NLL difference with a block-bootstrap interval, KL
(reference to arm, top-64 plus tail), top-1 agreement, task pass rates with Wilson intervals, ms and misses a
token, and which arms sit on the Pareto frontier. An arm marked `"noise": true` should be numerically equivalent to
the reference (a different prefill chunk size): its differences define the rounding band, and an arm inside the band
on NLL and KL is "inside noise". An arm whose log-probabilities equal the reference's bit for bit is "identical".

What it does not do: speed here is the teacher-forced decode step (indicative; alternate arms and confirm any win
through the server path, MEASUREMENT.md), the tasks do not include Hermes's real tool-call prompt (extend
`pareto_tasks.json` with a dump-based task for that), and default-on stays Hamed's call. HANDOFF section 18.57.

Usage (one runtime at a time; a sweep takes ~10 minutes an arm):
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/pareto.py sweep benchmarks/pareto_arms.example.json --out /tmp/pareto-1
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/pareto.py report /tmp/pareto-1 --arms benchmarks/pareto_arms.example.json
    (one arm alone:  ... pareto.py run NAME --arms FILE --out DIR [--ref DIR/ref.npz] [--quick])
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TASKS_FILE = Path(__file__).resolve().parent / "pareto_tasks.json"
TOP_K = 64

# Prose, code and numeric markdown, fixed so every arm sees the same tokens (the manifest hashes them).
DEFAULT_TEXTS = [
    {"name": "prose", "path": "docs/SPEED-RESEARCH-2026-10-03.md", "start": 2000, "chars": 3000},
    {"name": "code", "path": "src/cachalot/cache/wired_governor.py", "start": 0, "chars": 3000},
    {"name": "markdown", "path": "CHANGELOG.md", "start": 20000, "chars": 3000},
]


# ---------------------------------------------------------------------------------------------------------------
# scoring: pure functions (tested in tests/test_pareto.py)
# ---------------------------------------------------------------------------------------------------------------

def wilson(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a pass rate."""
    if total == 0:
        return (0.0, 1.0)
    p = passed / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def block_bootstrap_ci(diff: np.ndarray, groups: np.ndarray, *, block: int = 16, n: int = 2000,
                       seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95 % interval of paired differences, resampling blocks of `block` consecutive positions inside
    each group (a text), since neighbouring token losses are not independent."""
    rng = np.random.default_rng(seed)
    blocks = []
    for g in np.unique(groups):
        d = diff[groups == g]
        for i in range(0, len(d), block):
            blocks.append(d[i:i + block])
    sums = np.array([b.sum() for b in blocks])
    counts = np.array([len(b) for b in blocks])
    means = np.empty(n)
    for k in range(n):
        pick = rng.integers(0, len(blocks), len(blocks))
        means[k] = sums[pick].sum() / counts[pick].sum()
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def kl_top_tail(ref_logp: np.ndarray, arm_logp: np.ndarray) -> np.ndarray:
    """KL(ref || arm) per position from log-probabilities at the reference's top-K ids plus one tail bucket
    holding the remaining mass. Shapes [positions, K]."""
    p = np.exp(ref_logp.astype(np.float64))
    q = np.exp(arm_logp.astype(np.float64))
    p_tail = np.clip(1.0 - p.sum(axis=1), 1e-12, 1.0)
    q_tail = np.clip(1.0 - q.sum(axis=1), 1e-12, 1.0)
    head = (p * (ref_logp.astype(np.float64) - arm_logp.astype(np.float64))).sum(axis=1)
    return np.maximum(head + p_tail * (np.log(p_tail) - np.log(q_tail)), 0.0)


def pareto_front(points: list[tuple[str, tuple[float, ...]]]) -> set[str]:
    """Names of the non-dominated points; every coordinate is 'lower is better'."""
    front = set()
    for name, a in points:
        dominated = any(
            all(bx <= ax for bx, ax in zip(b, a, strict=True)) and any(bx < ax for bx, ax in zip(b, a, strict=True))
            for other, b in points if other != name
        )
        if not dominated:
            front.add(name)
    return front


def classify_arm(delta_nll: float, mean_kl: float, identical: bool,
                 band: tuple[float, float] | None) -> str:
    """'identical', 'inside noise' (|dNLL| and mean KL no larger than the noise arm's), 'outside noise', or
    'no band' when no noise arm was run."""
    if identical:
        return "identical"
    if band is None:
        return "no band"
    band_nll, band_kl = band
    return "inside noise" if abs(delta_nll) <= band_nll and mean_kl <= band_kl else "outside noise"


# ---------------------------------------------------------------------------------------------------------------
# task checks
# ---------------------------------------------------------------------------------------------------------------

def extract_code(text: str, lang: str) -> str | None:
    """The last fenced block of the language (or of any language), else None."""
    import re

    blocks = re.findall(r"```([A-Za-z#+]*)\n(.*?)```", text, flags=re.S)
    tagged = [b for tag, b in blocks if tag.lower() in {lang, "cs" if lang == "csharp" else lang,
                                                       "c#" if lang == "csharp" else lang}]
    pick = tagged or [b for _, b in blocks]
    return pick[-1] if pick else None


def check_exact(text: str, task: dict) -> bool:
    import re

    t = text.strip()
    if "regex" in task:
        return re.search(task["regex"], t, flags=re.S | re.I) is not None
    return all(s.lower() in t.lower() for s in task["contains"]) and not any(
        s.lower() in t.lower() for s in task.get("excludes", []))


def check_json(text: str, task: dict) -> bool:
    import re

    m = re.search(r"```(?:json)?\n(.*?)```", text, flags=re.S)
    raw = m.group(1) if m else text.strip()
    try:
        value = json.loads(raw)
    except ValueError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return False
        try:
            value = json.loads(raw[start:end + 1])
        except ValueError:
            return False
    return isinstance(value, dict) and all(
        k in value and (task["keys"][k] is None or value[k] == task["keys"][k]) for k in task["keys"])


def check_python(text: str, task: dict, timeout: float = 20.0) -> bool:
    code = extract_code(text, "python")
    if not code:
        return False
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.py"
        path.write_text(code + "\n\n" + task["asserts"] + "\n")
        try:
            r = subprocess.run([sys.executable, str(path)], capture_output=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


CSPROJ = textwrap.dedent("""\
    <Project Sdk="Microsoft.NET.Sdk">
      <PropertyGroup>
        <OutputType>Library</OutputType>
        <TargetFramework>net9.0</TargetFramework>
        <ImplicitUsings>enable</ImplicitUsings>
        <Nullable>disable</Nullable>
      </PropertyGroup>
    </Project>
    """)


def check_csharp(text: str, task: dict, timeout: float = 180.0) -> bool:
    code = extract_code(text, "csharp")
    if not code:
        return False
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "T.csproj").write_text(CSPROJ)
        (Path(d) / "Code.cs").write_text(code)
        try:
            r = subprocess.run(["dotnet", "build", "-nologo", "-v", "q", "-o", str(Path(d) / "out")],
                               capture_output=True, timeout=timeout, cwd=d)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        return r.returncode == 0


CHECKS = {"exact": check_exact, "json": check_json, "python": check_python, "csharp": check_csharp}


# ---------------------------------------------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------------------------------------------

def load_arms(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    arms = {a["name"]: a for a in cfg["arms"]}
    if len({a["name"] for a in cfg["arms"]}) != len(cfg["arms"]):
        raise ValueError("arm names must be unique")
    return {"base": cfg.get("base", {}), "arms": arms, "order": [a["name"] for a in cfg["arms"]],
            "reference": cfg.get("reference", cfg["arms"][0]["name"])}


def serve_defaults() -> None:
    """The environment `serve.sh` gives DeepSeek, unless already set (so an arm measures what is served)."""
    home = Path.home()
    verified = home / "DeepSeek-V4.1-Flash-q2g128" / ".cachalot-verified"
    bank = home / "DeepSeek-V4.1-Flash-q2g128" if verified.exists() else Path(
        "/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q2g128")
    os.environ.setdefault("CACHALOT_MODEL_PATH", "/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash")
    os.environ.setdefault("CACHALOT_EXPERT_BANK", str(bank))
    os.environ.setdefault("CACHALOT_PAGE_CACHE", "1")
    os.environ.setdefault("CACHALOT_MLX_WIRED_LIMIT_GIB", "80")
    os.environ.setdefault("CACHALOT_HOTLIST", str(home / "cachalot-hotlist.json"))
    os.environ.setdefault("CACHALOT_HOTLIST_GIB", "8")
    os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")


def git_state() -> dict:
    def run(*a: str) -> str:
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO).stdout.strip()

    return {"sha": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


# ---------------------------------------------------------------------------------------------------------------
# run one arm
# ---------------------------------------------------------------------------------------------------------------

def _load_texts(specs: list[dict]) -> list[tuple[str, str]]:
    out = []
    for s in specs:
        text = (REPO / s["path"]).read_text()
        out.append((s["name"], text[s["start"]:s["start"] + s["chars"]]))
    return out


def teacher_forced(model, texts: list[tuple[str, str]], *, prefill: int, decode: int, prefill_chunk: int | None,
                   ref_ids: np.ndarray | None) -> dict:
    """Prefill each text's first `prefill` tokens, then teacher-force the next `decode` tokens one decode step at a
    time. With `ref_ids` ([positions, TOP_K]) the arm's log-probabilities are read at those ids, else at its own
    top-K."""
    import mlx.core as mx

    rt = model.runtime
    cols: dict[str, list] = {k: [] for k in ("target_logp", "group", "ids", "logp", "top1", "ms", "misses")}
    pos = 0
    for gi, (_, text) in enumerate(texts):
        ids = list(rt.tokenizer.encode(text))[: prefill + decode + 1]
        if len(ids) < prefill + decode + 1:
            raise ValueError("text too short for the requested prefill + decode")
        rt.reset()
        chunk = prefill_chunk or prefill
        res = None
        for a in range(0, prefill, chunk):
            res = rt.prefill_tokens(ids[a:min(a + chunk, prefill)])
        logits = res.logits
        for i in range(prefill, prefill + decode):
            lp = logits.astype(mx.float32)
            lp = np.array(lp - mx.logsumexp(lp))
            target = ids[i]
            if ref_ids is None:
                top = np.argpartition(-lp, TOP_K)[:TOP_K]
                top = top[np.argsort(-lp[top])]
            else:
                top = ref_ids[pos]
            cols["target_logp"].append(lp[target])
            cols["group"].append(gi)
            cols["ids"].append(top)
            cols["logp"].append(lp[top])
            cols["top1"].append(int(lp.argmax()))
            before = rt.expert_store.stats().cache_misses
            t0 = time.perf_counter()
            logits = rt.decode_token(target).logits
            mx.eval(logits)
            cols["ms"].append(1000 * (time.perf_counter() - t0))
            cols["misses"].append(rt.expert_store.stats().cache_misses - before)
            pos += 1
    return {k: np.array(v) for k, v in cols.items()}


def run_battery(model, tasks: list[dict], quick: bool) -> list[dict]:
    results = []
    for t in tasks:
        if quick and not t.get("quick", False):
            continue
        t0 = time.perf_counter()
        resp = model.chat([{"role": "user", "content": t["prompt"]}], max_new_tokens=t.get("max_tokens", 256),
                          temperature=0.0)
        text = resp.content
        ok = bool(CHECKS[t["kind"]](text, t))
        results.append({"id": t["id"], "kind": t["kind"], "passed": ok, "seconds": round(time.perf_counter() - t0, 1),
                        "chars": len(text), "text": text[:3000]})
        print(f"[task] {t['id']:<22} {'pass' if ok else 'FAIL'}  {results[-1]['seconds']}s", file=sys.stderr, flush=True)
    return results


def cmd_run(args) -> None:
    cfg = load_arms(args.arms)
    arm = cfg["arms"][args.name]
    base = cfg["base"]
    for k, v in {**base.get("env", {}), **arm.get("env", {})}.items():
        os.environ[k] = str(v)
    serve_defaults()
    sys.path.insert(0, str(REPO / "src"))
    from cachalot.model.api import V41Model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = _load_texts(base.get("texts", DEFAULT_TEXTS))
    ref_ids = np.load(args.ref)["ids"] if args.ref else None
    budget = float(arm.get("budget_gib", base.get("budget_gib", 48)))
    manifest = {"arm": args.name, "config": arm, "base": base, "git": git_state(), "budget_gib": budget,
                "texts_hash": hashlib.sha256("\x00".join(t for _, t in texts).encode()).hexdigest()[:16],
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "complete": False}
    prefill, decode = int(base.get("prefill", 384)), int(base.get("decode", 192))
    if args.quick:
        decode = min(decode, 48)
    print(f"[arm {args.name}] loading (budget {budget:g} GiB, env {arm.get('env', {})})", file=sys.stderr, flush=True)
    with V41Model.from_pretrained(os.environ["CACHALOT_MODEL_PATH"],
                                  max_seq_len=int(base.get("max_seq_len", 4096)),
                                  expert_cache_budget_bytes=int(budget * 2**30)) as model:
        model.set_decode_miss_budget(arm.get("miss_budget"))
        tf = teacher_forced(model, texts, prefill=prefill, decode=decode,
                            prefill_chunk=arm.get("prefill_chunk"), ref_ids=ref_ids)
        tasks = json.loads(TASKS_FILE.read_text())["tasks"]
        battery = run_battery(model, tasks, args.quick)
    np.savez_compressed(out / f"{args.name}.npz", **tf)
    manifest.update({"tasks": battery, "complete": True, "positions": int(len(tf["ms"]))})
    (out / f"{args.name}.json").write_text(json.dumps(manifest, indent=1))  # written last: a killed run has none
    print(f"[arm {args.name}] done: {manifest['positions']} positions, "
          f"{sum(t['passed'] for t in battery)}/{len(battery)} tasks", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# sweep and report
# ---------------------------------------------------------------------------------------------------------------

def cmd_sweep(args) -> None:
    cfg = load_arms(args.arms)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    order = [cfg["reference"]] + [n for n in cfg["order"] if n != cfg["reference"]]
    for name in order:
        if (out / f"{name}.json").exists():
            print(f"[sweep] {name}: already complete, skipped", file=sys.stderr, flush=True)
            continue
        settle = REPO / "benchmarks" / "settle.sh"
        if settle.exists():
            subprocess.run([str(settle)], check=False)
        cmd = [sys.executable, str(Path(__file__).resolve()), "run", name, "--arms", args.arms, "--out", str(out)]
        if name != cfg["reference"]:
            cmd += ["--ref", str(out / f"{cfg['reference']}.npz")]
        if args.quick:
            cmd.append("--quick")
        env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
        if subprocess.run(cmd, env=env).returncode != 0:
            sys.exit(f"[sweep] arm {name} failed; its partial result is not kept")


def load_arm(out: Path, name: str) -> tuple[dict, dict]:
    manifest = json.loads((out / f"{name}.json").read_text())
    with np.load(out / f"{name}.npz") as z:
        return manifest, {k: z[k] for k in z.files}


def summarise(out: Path, cfg: dict) -> list[dict]:
    ref_name = cfg["reference"]
    ref_m, ref = load_arm(out, ref_name)
    rows: dict[str, dict] = {}
    for name in cfg["order"]:
        if not (out / f"{name}.json").exists():
            continue
        m, a = load_arm(out, name)
        if m["texts_hash"] != ref_m["texts_hash"] or len(a["ms"]) != len(ref["ms"]):
            raise SystemExit(f"{name}: not the same texts or positions as the reference")
        d = ref["target_logp"] - a["target_logp"]          # NLL arm - NLL ref = ref_logp - arm_logp
        mean, lo, hi = block_bootstrap_ci(d, a["group"])
        kl = kl_top_tail(ref["logp"], a["logp"]) if name != ref_name else np.zeros(len(d))
        passed = sum(t["passed"] for t in m["tasks"])
        total = len(m["tasks"])
        rows[name] = {
            "name": name, "d_nll": mean, "d_lo": lo, "d_hi": hi, "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
            "top1": float((a["top1"] == ref["top1"]).mean()), "ms": float(np.median(a["ms"])),
            "misses": float(a["misses"].mean()), "passed": passed, "total": total,
            "pass_lo": wilson(passed, total)[0], "identical": bool(np.all(d == 0)) and name != ref_name,
            "noise": bool(cfg["arms"][name].get("noise", False)), "nll": float(-a["target_logp"].mean()),
        }
    noise = [r for r in rows.values() if r["noise"]]
    band = None
    if noise:
        n = noise[0]
        band = (max(abs(n["d_lo"]), abs(n["d_hi"])), n["kl_mean"])
    for r in rows.values():
        r["verdict"] = "reference" if r["name"] == ref_name else (
            "noise arm" if r["noise"] else classify_arm(r["d_nll"], r["kl_mean"], r["identical"], band))
    front = pareto_front([(r["name"], (r["ms"], max(r["d_nll"], 0.0), 1 - r["pass_lo"])) for r in rows.values()])
    for r in rows.values():
        r["frontier"] = r["name"] in front
    return list(rows.values())


def cmd_report(args) -> None:
    cfg = load_arms(args.arms)
    rows = summarise(Path(args.out), cfg)
    print("| arm | verdict | dNLL nats [95 % CI] | KL mean / max | top-1 | tasks (Wilson low) | ms a token | misses | frontier |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['verdict']} | {r['d_nll']:+.4f} [{r['d_lo']:+.4f}, {r['d_hi']:+.4f}] | "
              f"{r['kl_mean']:.4f} / {r['kl_max']:.3f} | {r['top1']:.1%} | {r['passed']}/{r['total']} "
              f"({r['pass_lo']:.0%}) | {r['ms']:.0f} | {r['misses']:.1f} | {'yes' if r['frontier'] else ''} |")
    print("\nms is the teacher-forced decode step on one process (indicative: alternate arms and confirm a win through"
          " the server path). 'inside noise' needs a `noise: true` arm; default-on stays Hamed's call.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--arms", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--ref")
    r.add_argument("--quick", action="store_true", help="48 positions a text and only tasks marked quick")
    s = sub.add_parser("sweep")
    s.add_argument("arms")
    s.add_argument("--out", required=True)
    s.add_argument("--quick", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("out")
    p.add_argument("--arms", required=True)
    args = ap.parse_args()
    {"run": cmd_run, "sweep": cmd_sweep, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
