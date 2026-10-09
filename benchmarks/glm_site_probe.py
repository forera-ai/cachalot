"""
GLM same-site break probe (HANDOFF 18.111/18.112): what does GLM predict at the position where 15 of 24 compile-panel replies
broke (`else if (c == delimiter)` written as `else` + a stray `delimiter`)?

For replies of the compile panel that contain the site, the chat prompt plus the reply text up to just after `else\\n` is
teacher-forced, then the next few tokens of the reply (the indentation and the stray `delimiter`) are forced one at a time.
Arms, all on the same tokens and the same prefilled context (everything before the last WINDOW tokens):

    prefill   the last WINDOW + forced tokens in one chunk (GLM's prefill mode)
    exact     the same tokens one at a time through the decode path, no miss budget
    b2        the same, decode with a miss budget of 2

At each forced position the log-probability of the token the reply actually wrote, its rank, and the top three candidates are
saved. Output: results.jsonl (one line per reply and arm) and a printed table. The machine is Hamed's working machine: the expert
budget follows the available memory, the model's own memory fit runs every 16 tokens, nothing is closed or moved.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/glm_site_probe.py --out benchmarks/results/glm-site-probe
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/glm_site_probe.py --site else --replies 4 --out benchmarks/results/glm-site-probe-else
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CACHALOT_PAGE_CACHE", "1")
os.environ.setdefault("CACHALOT_MLX_WIRED_LIMIT_GIB", "80")
os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
os.environ.setdefault("CACHALOT_GLM_PREDICT_TOPK", "0")
BANK = "/Volumes/X10Pro/models/GLM-5.3-Flash-bank"
if "CACHALOT_GLM_BANK" not in os.environ and os.path.exists(f"{BANK}/bank.json"):
    os.environ["CACHALOT_GLM_BANK"] = BANK

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as rm  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("CACHALOT_MODEL_PATH", "/Volumes/X10Pro/models/GLM-5.3-Flash-MLX-4bit-MTP")
PANEL = REPO / "benchmarks/results/glm-compile-panel"
SITE = re.compile(r"else[ \t]*\n([ \t]*)delimiter")


def pick_budget_gib() -> float:
    from cachalot.glm.model import host_available

    avail = host_available() / 2**30
    for gib, need in ((46.0, 70.0), (44.0, 62.0), (40.0, 54.0)):
        if avail >= need:
            return gib
    return 36.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--replies", type=int, default=6, help="replies with the site to probe (half of each arm)")
    ap.add_argument("--window", type=int, default=40, help="tokens before the site fed in decode mode")
    ap.add_argument("--forced", type=int, default=6, help="reply tokens after the site that are scored")
    ap.add_argument("--site", choices=("newline", "else"), default="newline",
                    help="where the scored tokens start: after `else\\n` (the stray word's position) or right after `else` (where ` if` belongs)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from cachalot.glm.model import GlmModel

    rows = [json.loads(l) for l in (PANEL / "replies.jsonl").read_text().splitlines()]
    hit = [r for r in rows if SITE.search(r["content"] or "")]
    pick = []
    for arm in ("exact", "b2"):
        pick += [r for r in hit if r["arm"] == arm][: args.replies // 2]
    print(f"{len(hit)} of {len(rows)} replies have the site; probing {[(r['arm'], r['k']) for r in pick]}", flush=True)

    budget_gib = pick_budget_gib()
    print(f"expert budget {budget_gib} GiB (chosen from host availability)", flush=True)
    man = rm.collect(workload="W4:glm-site-probe", budget_gib=budget_gib, instrument="glm_site_probe",
                     workload_text=(PANEL / "prompt.txt").read_text())
    mid = rm.write_manifest(out, man)
    t_start = time.perf_counter()
    m = GlmModel(MODEL, expert_budget_gib=budget_gib, heartbeat_seconds=0)
    store, tok = m.store, m.tokenizer
    prompt = (PANEL / "prompt.txt").read_text()
    text = m.render_chat([{"role": "user", "content": prompt}], tools=None, thinking=False, effort=None, add_generation_prompt=True)
    prompt_ids = list(tok.encode(text, add_special_tokens=False))
    print(f"prompt tokens {len(prompt_ids)} (the panel's usage said 162)", flush=True)

    def dec(i):
        return tok.decode([int(i)])

    results = []
    for ri, r in enumerate(pick):
        content = r["content"]
        mt = SITE.search(content)
        site_char = mt.start(1) if args.site == "newline" else mt.start() + len("else")  # after `else\n`, or right after `else`
        tail = mt.end()  # end of `delimiter`
        full = content[:tail + 3]
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = list(enc["input_ids"]), list(enc["offset_mapping"])
        s = sum(1 for (a, b) in offs if b <= site_char)  # tokens fully before the site
        forced = ids[s:s + args.forced]
        w = max(0, s - args.window)
        ctx = prompt_ids + ids[:w]
        mid_toks = ids[w:s]
        seq = mid_toks + forced  # scored: logits that predict forced[0..]
        arms = ["prefill", "exact", "b2"]
        arms = arms[ri % 3:] + arms[:ri % 3]
        print(f"\n== reply {r['arm']} k={r['k']}: site at token {s} of {len(ids)}, window {len(mid_toks)}, forced {[dec(t) for t in forced]}", flush=True)

        m.set_decode_miss_budget(None)
        m._fit_prefill(len(ctx))
        cache = m.new_cache()
        logits = m.prefill(ctx, cache)
        mx.eval(logits)
        snap = m.snapshot(ctx, cache, None)
        del cache
        n_mid = len(mid_toks)
        for arm in arms:
            t0 = time.perf_counter()
            cache = m.restore(snap)
            lps = []  # log-prob rows predicting forced[0..]
            if arm == "prefill":
                m.set_decode_miss_budget(None)
                o = m.model(mx.array(seq[:-1], dtype=mx.int32)[None], cache=cache)
                full_l = getattr(o, "logits", o)[0].astype(mx.float32)
                lp = full_l - mx.logsumexp(full_l, axis=-1, keepdims=True)
                mx.eval(lp)
                lps = [np.array(lp[i]) for i in range(n_mid - 1, lp.shape[0])] if n_mid else []
                store.release_prefill()
            else:
                m.set_decode_miss_budget(2 if arm == "b2" else None)
                last = logits  # predicts seq[0] when n_mid == 0
                for j, tkn in enumerate(seq[:-1]):
                    step = m._forward([tkn], cache).astype(mx.float32)
                    lp = step - mx.logsumexp(step, axis=-1, keepdims=True)
                    mx.eval(lp)
                    if j >= n_mid - 1:
                        lps.append(np.array(lp[0]))
                    if j % 16 == 15:
                        m._fit_memory()
                m.set_decode_miss_budget(None)
            dt = time.perf_counter() - t0
            rec = {"arm": arm, "reply": [r["arm"], r["k"]], "seconds": round(dt, 1), "positions": []}
            for pi, row in enumerate(lps):
                tgt = forced[pi]
                order = np.argsort(-row)
                rec["positions"].append({"actual": dec(tgt), "lp": float(row[tgt]), "rank": int(np.where(order == tgt)[0][0]),
                                         "top3": [[dec(i), round(float(np.exp(row[i])), 4)] for i in order[:3]]})
            results.append(rec)
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{(time.perf_counter() - t_start) / 60:5.1f} min] {arm:8s} {dt:5.0f} s", flush=True)
            for p in rec["positions"]:
                print(f"    actual {p['actual']!r:16} lp {p['lp']:7.3f} rank {p['rank']:4d}  top3 {p['top3']}", flush=True)
            del cache
        del snap
    m.close()
    print("\nSUMMARY: mean log-prob of the reply's own next token and share of forced positions where it is the top-1, by arm")
    for arm in ("prefill", "exact", "b2"):
        ps = [p for rec in results if rec["arm"] == arm for p in rec["positions"]]
        print(f"  {arm:8s} mean lp {np.mean([p['lp'] for p in ps]):7.3f}  top-1 share {np.mean([p['rank'] == 0 for p in ps]):.2f}  (n={len(ps)})")


if __name__ == "__main__":
    main()
