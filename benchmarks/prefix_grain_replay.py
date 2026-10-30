"""DI-4 price (docs/DEEPINFER-LEVER-LANE.md): would token-granular (128) prefix reuse have prefilled fewer tokens than the snapshot cache did?

Replays a `CACHALOT_SERVER_DUMP` request dump through the real `prepare_prompt` and `PrefixCache` exactly as `prefix_pin_replay.py` does (no model is
loaded), and for every request also finds the longest common prefix with any earlier prompt (each followed by its recorded reply). A 128-grain
reuse would hit at the largest multiple of 128 inside that prefix minus one 128-token replay window (DeepInfer's rule: only global KV is stored at
multiples of 128 and the sliding window is rebuilt by replaying the last 128 tokens). It prints, per request, the real reuse, the shared prefix
and the grain hit, and totals: tokens the cache prefilled, tokens a grain would have prefilled, and how many requests diverged inside a chunk
(grain hit above the real reuse). Registered falsifier (lane file): 80 % or more of requests diverging exactly at a snapshot point or at the end
of the prompt means the lever is not worth building.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/prefix_grain_replay.py benchmarks/results/hermes-live-di4/requests.jsonl [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GRAIN = 128


def common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def grain_hit(shared: int, grain: int = GRAIN) -> int:
    """Largest multiple of `grain` inside a shared prefix, minus one replay window; 0 below two grains."""
    return max(0, (shared // grain) * grain - grain)


def summarize(rows: list[dict]) -> dict:
    """rows: per request {prompt, reused, shared}. `extra` is the tokens a grain hit would have saved over the cache's reuse."""
    out = []
    for r in rows:
        g = grain_hit(r["shared"])
        out.append({**r, "grain_hit": g, "extra": max(0, g - r["reused"]), "diverged_in_chunk": g > r["reused"]})
    n = len(out)
    return {"requests": n, "real_prefilled": sum(r["prompt"] - r["reused"] for r in out),
            "grain_prefilled": sum(r["prompt"] - max(r["grain_hit"], r["reused"]) for r in out),
            "diverged_in_chunk": sum(r["diverged_in_chunk"] for r in out),
            "diverged_share": (sum(r["diverged_in_chunk"] for r in out) / n) if n else None,
            "extra_tokens": sorted((r["extra"] for r in out if r["extra"]), reverse=True), "rows": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--gib", type=float, default=1.5)
    ap.add_argument("--entries", type=int, default=64, help="PrefixCache max_entries (the server's prefix_cache_entries)")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer
    from transformers.utils import logging

    import prefix_pin_replay as P
    from cachalot.model.generation import load_official_encoding, prepare_prompt
    from cachalot.model.prefix_cache import PrefixCache
    from cachalot.server.engine import Engine
    from reply_splice_replay import MODEL_PATH, chat_request

    logging.set_verbosity_error()
    e = Engine.__new__(Engine)
    e.encoding = load_official_encoding(MODEL_PATH)
    e.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    e._own_replies = deque(maxlen=64)
    e.replies_spliced = 0
    rows_in = [json.loads(line) for line in open(args.dump)]
    replies = [r["reply"] for r in rows_in if "reply" in r]
    bodies = [r for r in rows_in if "body" in r]
    cache = PrefixCache(max_entries=args.entries, max_pinned=64, max_bytes=int(args.gib * 1024**3))
    rt = P.TokenRuntime(cache)
    unused, seen, rows = list(replies), [], []
    for row in bodies:
        req = chat_request(row["body"])
        tokens, images = e._encode_chat(req)
        tokens = e._splice_own_replies(tokens, req.thinking_mode)
        boundary = 0 if images else e.system_prefix_len(req, tokens)
        _, reused = prepare_prompt(rt, tokens, boundaries=(boundary,) if boundary else ())
        rows.append({"prompt": len(tokens), "reused": reused, "shared": max((common_prefix(s, tokens) for s in seen), default=0), "images": len(images)})
        seen.append(list(tokens))
        reply = next((r for r in unused if r["prompt_tokens"] == tokens), None)
        if reply is not None:
            unused.remove(reply)
            rt.tokens.extend(reply["reply_tokens"])
            rt.position += len(reply["reply_tokens"])
            cache.add(rt.snapshot())
            e._remember_reply(tokens, reply["reply_tokens"], e.tokenizer.decode(reply["reply_tokens"], skip_special_tokens=False), req.thinking_mode)
            seen[-1] = seen[-1] + list(reply["reply_tokens"])
    res = summarize(rows)
    for i, r in enumerate(res["rows"]):
        print(f"req{i:3d} prompt={r['prompt']:6d} reused={r['reused']:6d} shared={r['shared']:6d} grain_hit={r['grain_hit']:6d} extra={r['extra']:5d}")
    print(f"requests {res['requests']}: cache prefilled {res['real_prefilled']}, grain would prefill {res['grain_prefilled']}, "
          f"diverged inside a chunk {res['diverged_in_chunk']} ({res['diverged_share']:.0%})")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
