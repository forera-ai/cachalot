"""
Teacher-forced Farsi leak probe: does the decode miss budget make DeepSeek put English words into Farsi?
(docs/v1017-farsi-leak/RECORD.md)

Live Hermes sessions under `CACHALOT_DECODE_MISS_BUDGET=0` wrote one stray English word into 2 of 6 Farsi story translations
("mattered", "heard"). This probe takes the six recorded replies from the local server dumps (their exact prompt and reply token ids,
`CACHALOT_SERVER_DUMP` reply rows), prefills the prompt and the reply up to its Farsi section once, snapshots that state, and then
teacher-forces the recorded Farsi tokens one decode step at a time under each arm: exact (no budget) and budget 0, restoring the
snapshot between them. Budget 0 goes first in every section, so it meets the expert cache the prefill left (as the live decode
did); exact decode reads every expert, so its output does not depend on the cache the first arm leaves.

Per forced position it records: the probability mass on tokens that carry two or more ASCII letters (the "leak" mass: a Latin word
starting or continuing), the log-probability of the recorded token, the arm's top 1,024 log-probabilities (for KL between arms), and
the experts the budget dropped. Positions whose recorded token is itself Latin (the stray words) are listed separately.

No server may run (one runtime). The dumps hold personal file names and stay outside the repository; sections are pinned by dump file,
row and the sha256 of their token ids.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    benchmarks/guarded_run.sh --budget-gib 48 --force --tag farsi-probe -- env PYTHONPATH=src \\
        ~/venvs/deepseek-v41/bin/python benchmarks/farsi_leak_probe.py run --out benchmarks/results/farsi-leak
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/farsi_leak_probe.py report benchmarks/results/farsi-leak
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TOP = 1024
PERSIAN = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]{2,}")
# (label, dump file in $HOME, row index of the reply row, sha256[:16] of prompt + reply token ids). Rows found by
# docs/v1017-farsi-leak/RECORD.md section 3: the six story replies with a Farsi section; listings (file names) are excluded.
SECTIONS = [
    ("F1", "cachalot-dump-toolexact.jsonl", 21, "bf7641658ba8f629"),  # 2026-11-02, fish story, stray "mattered"
    ("F2", "cachalot-dump-1017.jsonl", 33, "bd3cabbe69303ffb"),       # 2026-11-01, hummingbird, stray "heard"
    ("F3", "cachalot-dump-1017.jsonl", 57, "ef23483664c4c555"),       # 2026-11-01, fish story, clean
    ("F4", "cachalot-dump-1017.jsonl", 59, "c1374c29d7ca25a5"),       # 2026-11-01, hummingbird, clean
    ("F5", "cachalot-dump-di2.jsonl", 40, "ac6cc9d3529d111c"),        # 2026-11-02, fish story, clean
    ("F6", "cachalot-dump-di2.jsonl", 42, "e76294c357933e29"),        # 2026-11-02, hummingbird, clean
]
ARMS = {"exact": None, "budget0": 0}


def latin_mask(tokenizer) -> np.ndarray:
    """True for vocabulary ids whose text holds two or more consecutive ASCII letters."""
    n = len(tokenizer)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if LATIN.search(tokenizer.decode([i])):
            mask[i] = True
    return mask


def farsi_start(tokenizer, reply: list[int]) -> int | None:
    """Index of the first reply token on the line where Persian script first appears (None without any)."""
    if not PERSIAN.search(tokenizer.decode(reply)):
        return None
    lo, hi = 0, len(reply)
    while lo < hi:
        m = (lo + hi) // 2
        if PERSIAN.search(tokenizer.decode(reply[: m + 1])):
            hi = m
        else:
            lo = m + 1
    s = lo
    while s > 0 and "\n" not in tokenizer.decode(reply[s - 1 : s]):
        s -= 1
    return s


def load_section(name: str, row: int, expect: str | None = None) -> tuple[list[int], list[int], str]:
    rows = Path.home().joinpath(name).read_text().splitlines()
    rec = json.loads(rows[row])["reply"]
    prompt, reply = list(rec["prompt_tokens"]), list(rec["reply_tokens"])
    sha = hashlib.sha256(json.dumps([prompt, reply]).encode()).hexdigest()[:16]
    if expect is not None and sha != expect:
        raise SystemExit(f"{name} row {row}: token sha {sha}, expected {expect}")
    return prompt, reply, sha


def force(rt, logits, targets: list[int], mask: np.ndarray, store) -> dict:
    import mlx.core as mx

    cols: dict[str, list] = {k: [] for k in ("leak", "target_logp", "ids", "logp", "skipped", "ms")}
    for t in targets:
        lp = logits.astype(mx.float32)
        lp = np.array(lp - mx.logsumexp(lp))
        p = np.exp(lp)
        top = np.argpartition(-lp, TOP)[:TOP]
        top = top[np.argsort(-lp[top])]
        cols["leak"].append(float(p[: len(mask)][mask].sum()))
        cols["target_logp"].append(float(lp[t]))
        cols["ids"].append(top.astype(np.int32))
        cols["logp"].append(lp[top].astype(np.float16))
        before = store.skipped_experts
        t0 = time.perf_counter()
        logits = rt.decode_token(t).logits
        mx.eval(logits)
        cols["ms"].append(1000 * (time.perf_counter() - t0))
        cols["skipped"].append(store.skipped_experts - before)
    return {k: np.array(v) for k, v in cols.items()}


def cmd_run(args) -> None:
    sys.path.insert(0, str(REPO / "benchmarks"))
    from pareto import git_state, serve_defaults

    os.environ.setdefault("CACHALOT_PREFILL_ENCODER_ONLY", "1")
    os.environ.setdefault("CACHALOT_PREFILL_DECODER_KEEP", "140")
    serve_defaults()
    from cachalot.model.api import V41Model
    from cachalot.model.text_decode_runtime import IMAGE_TOKEN_ID

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = [s for s in SECTIONS if not args.only or s[0] in args.only.split(",")]
    with V41Model.from_pretrained(os.environ["CACHALOT_MODEL_PATH"], max_seq_len=args.max_seq_len,
                                  expert_cache_budget_bytes=int(args.budget_gib * 2**30)) as model:
        rt = model.runtime
        store = rt.expert_store
        mask = latin_mask(rt.tokenizer)
        print(f"[probe] latin tokens {int(mask.sum())} of {len(mask)}", file=sys.stderr, flush=True)
        manifest = {"git": git_state(), "budget_gib": args.budget_gib, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "env": {k: v for k, v in os.environ.items() if k.startswith("CACHALOT_")}, "sections": {}}
        for si, (label, name, row, expect) in enumerate(chosen):
            prompt, reply, sha = load_section(name, row, expect)
            s = farsi_start(rt.tokenizer, reply)
            if s is None:
                raise SystemExit(f"{label}: no Farsi in {name} row {row}")
            targets = reply[s:]
            if args.limit:
                targets = targets[: args.limit]
            # The conversations' one image is not rebuilt (no vision tower here): its placeholder tokens are removed, in both arms.
            context = [t for t in prompt + reply[:s] if t != IMAGE_TOKEN_ID]
            stray = [i for i, t in enumerate(targets) if LATIN.search(rt.tokenizer.decode([t]))]
            t0 = time.perf_counter()
            rt.reset()
            res = None
            for a in range(0, len(context), 4096):
                res = rt.prefill_tokens(context[a : a + 4096], encoder_only=True)
            res = rt.finish_decoder()
            snap = rt.snapshot(logits=res.logits)
            prefill_s = time.perf_counter() - t0
            # budget 0 first, always: it then meets the expert cache the prefill left, as the live decode did; exact reads every
            # expert, so its output does not depend on what the first arm left resident (the smoke run, exact first, saw 0 drops)
            order = ["budget0", "exact"]
            print(f"[probe] {label} {name}:{row} sha {sha}: context {len(context):,} prefilled in {prefill_s:.0f} s, "
                  f"{len(targets)} Farsi tokens, stray at {stray}, order {order}", file=sys.stderr, flush=True)
            info = {"dump": name, "row": row, "sha": sha, "context": len(context),
                    "image_tokens_removed": len(prompt) + s - len(context), "farsi_tokens": len(targets),
                    "stray_positions": stray, "stray_text": [rt.tokenizer.decode([targets[i]]) for i in stray],
                    "order": order, "prefill_s": round(prefill_s, 1)}
            for arm in order:
                rt.restore(snap)
                model.set_decode_miss_budget(ARMS[arm])
                t1 = time.perf_counter()
                cols = force(rt, snap.logits, targets, mask, store)
                np.savez_compressed(out / f"{label}-{arm}.npz", **cols)
                info[arm] = {"seconds": round(time.perf_counter() - t1, 1), "leak_mean": float(cols["leak"].mean()),
                             "skipped_per_token": float(cols["skipped"].mean())}
                print(f"[probe] {label} {arm}: leak mean {cols['leak'].mean():.4%}, skipped/token {cols['skipped'].mean():.1f}, "
                      f"{info[arm]['seconds']} s", file=sys.stderr, flush=True)
            model.set_decode_miss_budget(None)
            manifest["sections"][label] = info
            (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        manifest["complete"] = True
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def kl_on_union(ids_p, lp_p, ids_q, lp_q) -> float:
    """KL(P || Q) over P's top ids; Q's log-probability outside its own top set is floored at its smallest kept value."""
    q = dict(zip(ids_q.tolist(), lp_q.astype(np.float64).tolist()))
    floor = float(lp_q.min())
    lpp = lp_p.astype(np.float64)
    lq = np.array([q.get(i, floor) for i in ids_p.tolist()])
    return float((np.exp(lpp) * (lpp - lq)).sum())


def cmd_report(args) -> None:
    d = Path(args.out)
    man = json.loads((d / "manifest.json").read_text())
    print(f"git {man['git']}, budget {man['budget_gib']} GiB, complete {man.get('complete', False)}")
    print("| section | tokens | order | leak exact | leak budget 0 | ratio | dropped/token (b0) | KL mean | stray: logp exact / budget 0 |")
    print("|---|---|---|---|---|---|---|---|---|")
    pooled = {"exact": [], "budget0": []}
    higher = 0
    for label, info in man["sections"].items():
        if "exact" not in info or "budget0" not in info:
            continue
        a = dict(np.load(d / f"{label}-exact.npz"))
        b = dict(np.load(d / f"{label}-budget0.npz"))
        pooled["exact"].append(a["leak"])
        pooled["budget0"].append(b["leak"])
        la, lb = a["leak"].mean(), b["leak"].mean()
        higher += lb > la
        kl = np.mean([kl_on_union(a["ids"][i], a["logp"][i], b["ids"][i], b["logp"][i]) for i in range(len(a["leak"]))])
        stray = "; ".join(f"{t!r} {a['target_logp'][i]:.2f} / {b['target_logp'][i]:.2f}"
                          for i, t in zip(info["stray_positions"], info["stray_text"]))
        print(f"| {label} | {info['farsi_tokens']} | {'/'.join(info['order'])} | {la:.4%} | {lb:.4%} | {lb / max(la, 1e-12):.2f} | "
              f"{b['skipped'].mean():.1f} | {kl:.4f} | {stray or '-'} |")
    ea, eb = np.concatenate(pooled["exact"]), np.concatenate(pooled["budget0"])
    # a block bootstrap over 64-position blocks (fewer for a smoke run) within the pooled sequence, for the ratio of means
    rng = np.random.default_rng(0)
    bs = 64 if len(ea) >= 256 else max(1, len(ea) // 4)
    nb = len(ea) // bs
    ratios = []
    for _ in range(2000):
        pick = rng.integers(0, nb, nb)
        ia = np.concatenate([np.arange(k * bs, k * bs + bs) for k in pick])
        ratios.append(eb[ia].mean() / max(ea[ia].mean(), 1e-12))
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    print(f"pooled: leak exact {ea.mean():.4%}, budget 0 {eb.mean():.4%}, ratio {eb.mean() / ea.mean():.2f} [95 % {lo:.2f}, {hi:.2f}], "
          f"budget 0 higher in {higher} of {len(pooled['exact'])} sections; positions with leak > 5 %: exact {(ea > 0.05).sum()}, "
          f"budget 0 {(eb > 0.05).sum()} of {len(ea)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--budget-gib", type=float, default=48.0)
    r.add_argument("--max-seq-len", type=int, default=49152)
    r.add_argument("--only", default="", help="comma list of section labels (F1..F6)")
    r.add_argument("--limit", type=int, default=0, help="force at most this many Farsi tokens a section (smoke)")
    p = sub.add_parser("report")
    p.add_argument("out")
    args = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
