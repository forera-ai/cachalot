"""DI-1 first measurement (docs/DI1-SPLIT-RECORD.md): how a cold prefill chunk's time splits between the encoder half (layers 0 to 19) and
the decoder half (layers 20 to 39) of the causal encoder-decoder.

It reuses the per-layer marks of `profile_prefill_timeline.py` (importing it installs them: it patches the MoE call, the block functions,
the SSD read hook and the Engram phase) and runs `rt.prefill_tokens` on one prompt of N tokens, `repeat` times (the first run is cold,
later ones warm). From the marks it computes each layer's time (the interval between consecutive `moe_end` marks; layer 0 starts at the
call), the MoE part of each layer (`moe_start` to `moe_end`), the SSD reads that started in each layer and the encoder and decoder sums.
It changes nothing in the runtime and adds no synchronisation (the marks only record host times; `CACHALOT_PROFILE_SYNC` must stay
unset), so the layer times are host-observed intervals of a pipelined run: they attribute wall time to the layer in which the host waited,
which is what the saving of skipping a layer would remove for the host-bound parts and only approximately for GPU work that overlaps the
next layer.

Run under the guardian (the machine gate and the sysctl are Hamed's; see the record):
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    benchmarks/guarded_run.sh --budget-gib 44 --max-seconds 1200 --tag di1 -- env PYTHONPATH=src \
        ~/venvs/deepseek-v41/bin/python benchmarks/prefill_half_split.py 4096 2 --json /tmp/di1-split.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

N_LAYERS, SPLIT = 40, 20


def layer_times(marks: list[tuple[float, str]], t0: float, n_layers: int = N_LAYERS) -> dict:
    """Per-layer seconds from the `moe_start Lk` / `moe_end Lk` marks of one prefill call. `total_s` of layer k is the interval from the
    previous layer's `moe_end` (the call's start for layer 0) to its own `moe_end`; the tail after the last `moe_end` is reported apart."""
    starts, ends = {}, {}
    for t, lab in marks:
        if lab.startswith("moe_start"):
            starts[int(lab.split()[1][1:])] = t
        elif lab.startswith("moe_end"):
            ends[int(lab.split()[1][1:])] = t
    layers = []
    prev = t0
    for k in range(n_layers):
        if k not in ends or k not in starts:
            layers.append({"layer": k, "total_s": None, "moe_s": None})
            continue
        layers.append({"layer": k, "total_s": ends[k] - prev, "moe_s": ends[k] - starts[k]})
        prev = ends[k]
    last_end = max(ends.values()) if ends else t0
    return {"layers": layers, "tail_s": None, "last_moe_end": last_end}


def halves(layers: list[dict], split: int = SPLIT) -> dict:
    """Encoder (layers below `split`) and decoder sums; None if any layer is missing."""
    if any(x["total_s"] is None for x in layers):
        return {"complete": False}
    enc = sum(x["total_s"] for x in layers[:split])
    dec = sum(x["total_s"] for x in layers[split:])
    enc_moe = sum(x["moe_s"] for x in layers[:split])
    dec_moe = sum(x["moe_s"] for x in layers[split:])
    tot = enc + dec
    return {"complete": True, "encoder_s": enc, "decoder_s": dec, "decoder_share": dec / tot if tot > 0 else None,
            "encoder_moe_s": enc_moe, "decoder_moe_s": dec_moe, "encoder_other_s": enc - enc_moe, "decoder_other_s": dec - dec_moe}


def reads_per_half(reads: list[tuple[float, float]], marks: list[tuple[float, str]], split: int = SPLIT) -> dict:
    """SSD reads that started before and after the end of layer `split - 1`'s MoE."""
    cut = next((t for t, lab in marks if lab == f"moe_end L{split - 1}"), None)
    if cut is None:
        return {"complete": False}
    enc = sum(1 for s, _ in reads if s < cut)
    return {"complete": True, "encoder_reads": enc, "decoder_reads": len(reads) - enc}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n_tokens", type=int, nargs="?", default=4096)
    ap.add_argument("repeat", type=int, nargs="?", default=2)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    import os

    if os.environ.get("CACHALOT_PROFILE_SYNC") == "1":
        print("STOP: CACHALOT_PROFILE_SYNC=1 changes overlap; unset it for the split", file=sys.stderr)
        return 2
    import profile_prefill_timeline as ppt
    from _common import MODEL_PATH
    from cachalot.model.generation import load_official_encoding
    from cachalot.model.text_decode_runtime import TextDecodeRuntime
    from trace_routing import build_prompt, prompt_sources

    out = {"n_tokens": args.n_tokens, "runs": []}
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=max(4096, args.n_tokens)) as rt:
        enc = load_official_encoding(MODEL_PATH)
        ids = build_prompt(rt, enc, prompt_sources()[0][1], args.n_tokens)
        out["prompt_tokens"] = len(ids)
        for run in range(args.repeat):
            ppt.READS.clear()
            ppt.MARKS.clear()
            rt.reset()
            t0 = perf_counter()
            rt.prefill_tokens(ids)
            wall = perf_counter() - t0
            lt = layer_times(list(ppt.MARKS), t0)
            rec = {"run": run, "kind": "cold" if run == 0 else "warm", "wall_s": wall, "halves": halves(lt["layers"]),
                   "reads": reads_per_half(list(ppt.READS), list(ppt.MARKS)), "n_reads": len(ppt.READS),
                   "layers": lt["layers"], "after_last_layer_s": (t0 + wall) - lt["last_moe_end"]}
            out["runs"].append(rec)
            h = rec["halves"]
            if h.get("complete"):
                print(f"run {run} ({rec['kind']}): wall {wall:.2f}s | encoder {h['encoder_s']:.2f}s | decoder {h['decoder_s']:.2f}s "
                      f"(share {h['decoder_share']:.3f}) | after the last layer {rec['after_last_layer_s']:.2f}s | reads {rec['reads']}")
            else:
                print(f"run {run}: layer marks incomplete")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
