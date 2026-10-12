"""
Raw per-layer rows of the DeepSeek decode token (docs/L4B-RAW-LAYER-RECORD.md, HANDOFF 18.136): runs the mixed workload of
`token_trace_price.py` through `Engine.chat` with the tracer recording every decode token and saves each turn's raw columns
(`TokenTrace.drain()`), which the per-token price run never kept.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    benchmarks/guarded_run.sh --budget-gib 34 --max-seconds 3000 --tag l4braw -- \
        env PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/token_trace_raw.py --budget-gib 34 --out benchmarks/results/l4b-raw

One runtime at a time (it loads the model).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))
from token_trace_price import MIXED, SYSTEM  # noqa: E402
from server_miss_budget_ab import USER as HERMES_USER  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget-gib", type=float, default=34.0)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--dump", type=Path, default=None, help="a dumped Hermes request file: its first body supplies the 22k-token system prompt and 25 tools; the prompts become the 12 Hermes user turns (docs/L4B-RAW-LAYER-RECORD.md arm 7)")
    ap.add_argument("--max-seq-len", type=int, default=16384)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import pareto

    pareto.serve_defaults()
    from cachalot.metrics import token_trace as tt
    from cachalot.model.api import V41Model
    from cachalot.model.generation import SamplingParams
    from cachalot.server.engine import ChatRequest, Engine

    model = V41Model.from_pretrained(os.environ["CACHALOT_MODEL_PATH"], max_seq_len=a.max_seq_len,
                                     expert_cache_budget_bytes=int(a.budget_gib * 2**30))
    from cachalot.cli import _decode_miss_budget_from_env  # the server applies this knob in cli.py, not in the model

    budget = _decode_miss_budget_from_env(os.environ)
    if budget is not None:
        model.set_decode_miss_budget(budget)
    print(f"decode miss budget: {budget}", flush=True)
    engine = Engine(model)
    rt = model.runtime
    trace = tt.TokenTrace(every=1)
    manifest: list[dict] = []
    if a.dump is not None:
        body = json.loads(a.dump.read_text().splitlines()[0])["body"]
        system, tools, prompts = body["messages"][0], body["tools"], HERMES_USER
    else:
        system, tools, prompts = {"role": "system", "content": SYSTEM}, None, MIXED

    def turn(idx: int, prompt: str, record: bool) -> None:
        rt.set_token_trace(trace if record else None)
        if record:
            trace.drain()
        req = ChatRequest(messages=[system, {"role": "user", "content": prompt}], tools=tools,
                          params=SamplingParams(max_new_tokens=a.tokens, temperature=0.0))
        m0 = rt.expert_store.cache_misses
        res = engine.chat(req)
        n = max(res.completion_tokens, 1)
        row = {"turn": idx, "recorded": record, "tokens": res.completion_tokens, "ms_tok": round(1000 * res.decode_seconds / n, 2),
               "misses_tok": round((rt.expert_store.cache_misses - m0) / n, 1)}
        if record:
            d = trace.drain()
            np.savez_compressed(a.out / f"turn{idx:03d}.npz", **{k: v for k, v in d.items() if k != "dropped"})
            row["dropped"] = int(d["dropped"])
            row["events"] = int(len(d["ts"]))
        manifest.append(row)
        print(row, flush=True)
        (a.out / "manifest.json").write_text(json.dumps({"budget_gib": a.budget_gib, "tokens": a.tokens, "rows": manifest}))

    for i in range(a.warm):
        turn(-1 - i, prompts[i % len(prompts)], False)
    for i in range(a.turns):
        turn(i, prompts[i % len(prompts)], True)
    rt.set_token_trace(None)
    model.close()


if __name__ == "__main__":
    main()
