"""DI-1 equality test (docs/DI1-BUILD-RECORD.md): does an encoder-only prefill with an exact decoder tail end in the state a full prefill ends in?

One mode per process (the runtime is one process at a time):

    full      prefill the prompt in `prepare_prompt`'s 4,096-token calls the usual way, then greedy-decode `--decode` tokens
    deferred  the same calls with `encoder_only=True`, then `finish_decoder()`, then the same greedy decode

and each writes `--out DIR` (final-position logits as float32, the sliding-window rings of every layer, the layer-20 compressed KV and index keys up to the
prompt, the shared candidates and top-k, the decoded token ids and their logits). `compare A B` prints the differences:

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    benchmarks/guarded_run.sh --budget-gib 34 --max-seconds 1500 --tag di1eq -- env PYTHONPATH=src CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \
        CACHALOT_EXPERT_BANK=$HOME/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=80 MLX_METAL_FAST_SYNCH=1 \
        ~/venvs/deepseek-v41/bin/python benchmarks/di1_tail_equality.py full --tokens 6100 --out /tmp/di1-eq-full

The rounding band of a regrouped MoE is the lane's Q-NOISE (KL mean 0.0144); an exact-arithmetic equivalence is expected to land far inside it on the logits and
to make the windows and the layer-20 caches equal to rounding, not bit-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def softmax64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(p || q) of two logit vectors, float64."""
    p, q = softmax64(p_logits), softmax64(q_logits)
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], 1e-300)))))


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a.astype(np.float32), b.astype(np.float32)
    d = np.abs(a - b)
    scale = float(np.abs(a).max()) or 1.0
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "rel_max": float(d.max() / scale), "equal": bool(np.array_equal(a, b))}


def run(mode: str, n_tokens: int, out: Path, decode: int) -> int:
    import mlx.core as mx

    from _common import MODEL_PATH
    from cachalot.model.generation import load_official_encoding, prefill_chunks
    from cachalot.model.text_decode_runtime import TextDecodeRuntime
    from trace_routing import build_prompt, prompt_sources

    out.mkdir(parents=True, exist_ok=True)
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=max(8192, n_tokens + decode + 8)) as rt:
        enc = load_official_encoding(MODEL_PATH)
        ids = build_prompt(rt, enc, prompt_sources()[0][1], n_tokens)
        rt.reset()
        chunks = prefill_chunks(0, len(ids))
        t0 = perf_counter()
        for i, (a, b) in enumerate(chunks):
            if mode == "full":
                res = rt.prefill_tokens(ids[a:b])
            else:
                res = rt.prefill_tokens(ids[a:b], encoder_only=True)
        if mode == "deferred":
            res = rt.finish_decoder()
        prefill_s = perf_counter() - t0
        logits = np.array(res.logits.astype(mx.float32))
        state = {"logits": logits}
        for layer, w in rt.windows.items():
            state[f"window_{layer}"] = np.array(w.astype(mx.float32))
        n = len(ids)
        state["compressed_20"] = np.array(rt.compressed_caches[20][:n].astype(mx.float32))
        state["index_k_20"] = np.array(rt.indexer_states[20].k_cache[:n].astype(mx.float32))
        for layer in (2, 8, 14):
            state[f"compressed_{layer}"] = np.array(rt.compressed_caches[layer][: n // 2].astype(mx.float32))
        if rt.shared_attn.candidates is not None:
            state["candidates"] = np.array(rt.shared_attn.candidates.astype(mx.int32))
        if rt.shared_attn.topk_idxs is not None:
            state["topk"] = np.array(rt.shared_attn.topk_idxs.astype(mx.int32))
        toks, dlog = [], []
        cur = int(np.argmax(logits))
        for _ in range(decode):
            toks.append(cur)
            r = rt.decode_token(cur)
            lg = np.array(r.logits.astype(mx.float32))
            dlog.append(lg)
            cur = int(np.argmax(lg))
        state["decode_tokens"] = np.array(toks, dtype=np.int64)
        state["decode_logits"] = np.stack(dlog) if dlog else np.zeros((0,))
        np.savez(out / "state.npz", **state)
        (out / "meta.json").write_text(json.dumps({"mode": mode, "prompt_tokens": n, "chunks": chunks, "decode": decode, "position": rt.position, "prefill_s": prefill_s}))
    print(f"{mode}: prompt {n} tokens in {len(chunks)} calls in {prefill_s:.1f} s, decoded {decode}, wrote {out}")
    return 0


def compare(a_dir: Path, b_dir: Path) -> int:
    A, B = np.load(a_dir / "state.npz"), np.load(b_dir / "state.npz")
    res: dict = {"keys_only_in_a": sorted(set(A.files) - set(B.files)), "keys_only_in_b": sorted(set(B.files) - set(A.files))}
    res["logits_kl_a_to_b"] = kl(A["logits"], B["logits"])
    res["logits_top1_equal"] = bool(np.argmax(A["logits"]) == np.argmax(B["logits"]))
    res["logits"] = compare_arrays(A["logits"], B["logits"])
    res["decode_tokens_equal"] = bool(np.array_equal(A["decode_tokens"], B["decode_tokens"]))
    res["decode_tokens_first_difference"] = next((i for i, (x, y) in enumerate(zip(A["decode_tokens"], B["decode_tokens"], strict=False)) if x != y), None)
    if A["decode_logits"].size and A["decode_logits"].shape == B["decode_logits"].shape:
        res["decode_kl_max"] = float(max(kl(x, y) for x, y in zip(A["decode_logits"], B["decode_logits"], strict=True)))
    for k in sorted(set(A.files) & set(B.files)):
        if k.startswith(("window_", "compressed_", "index_k_")):
            res[k] = compare_arrays(A[k], B[k])
        elif k in ("candidates", "topk"):
            res[k] = {"equal": bool(np.array_equal(A[k], B[k]))}
    windows = [v for k, v in res.items() if k.startswith("window_")]
    res["windows_max_rel"] = max((v["rel_max"] for v in windows), default=None)
    res["windows_bit_equal"] = all(v["equal"] for v in windows) if windows else None
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for m in ("full", "deferred"):
        p = sub.add_parser(m)
        p.add_argument("--tokens", type=int, default=6100)
        p.add_argument("--decode", type=int, default=16)
        p.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        return compare(Path(args.a), Path(args.b))
    return run(args.cmd, args.tokens, Path(args.out), args.decode)


if __name__ == "__main__":
    sys.exit(main())
