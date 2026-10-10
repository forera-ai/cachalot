"""E3e (docs/E3E-INPROC-KERNELS-RECORD.md): the DeepSeek decode kernels in-process, without the server, for the energy arms.

Same phases and phase log as benchmarks/energy_arms_driver.py (A idle, B all-resident decode, D idle) so benchmarks/energy_e3.py reads it, but the
model is driven by TextDecodeRuntime directly: no HTTP, no tokenizer streaming, no snapshot store, no stats. Phase B repeats the pass of
benchmarks/decode_resident.py (reset, prefill a short prompt, greedy decode N tokens), which requests the same experts every time, so after the
first pass the decode is all-resident. Per-pass decode intervals are logged so the prefill share of the phase is known.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && E_TAG=e3e ./benchmarks/inproc_arms.sh            (the energy run, sampler beside)
    ... benchmarks/decode_power_inproc.py --smoke                                                          (short, no energy logger)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, StoreSnapshot  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(rec: dict) -> None:
    print(json.dumps(rec), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=16)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--idle", type=float, default=90.0)
    ap.add_argument("--load", type=float, default=120.0)
    ap.add_argument("--prefill-phase", type=float, default=45.0)
    ap.add_argument("--smoke", action="store_true", help="warm passes and 10 resident passes only, no idle phases, no logger")
    ap.add_argument("--logger-out", default="", help="start benchmarks/energy_logger.py --full to this file once the runtime is ready")
    a = ap.parse_args()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        _, text = prompt_sources()[0]
        ids = build_prompt(rt, enc, text, a.prompt_tokens)
        log({"event": "ready", "clock": now(), "budget_gib": rt.expert_cache_budget_bytes / 2**30, "wired_gib": rt.mlx_wired_limit_bytes / 2**30})

        def one_pass() -> dict:
            rt.reset()
            before = StoreSnapshot.take(rt)
            t0 = perf_counter()
            r = rt.prefill_tokens(ids)
            mx.eval(r.logits)
            p1 = perf_counter()
            tok = int(r.logits.argmax().item())
            out = []
            for _ in range(a.decode_tokens):
                step = rt.decode_token(tok)
                tok = int(step.logits.argmax().item())
                out.append(tok)
            t1 = perf_counter()
            d = before.delta(StoreSnapshot.take(rt))
            return {"prefill_s": p1 - t0, "decode_s": t1 - p1, "tokens": out, "misses": d.cache_misses, "gib_read": d.ssd_bytes_read / 2**30}

        first = one_pass()["tokens"]
        for _ in range(2):  # warm: the first pass loaded the experts, the next two settle kernels and caches
            r = one_pass()
            if r["tokens"] != first:
                log({"event": "WARNING", "msg": "a pass decoded different tokens; the expert set is not repeating"})
        log({"event": "warm", "clock": now(), "ms_per_token": r["decode_s"] / a.decode_tokens * 1e3, "prefill_s": r["prefill_s"]})

        if a.smoke:
            for _ in range(10):
                r = one_pass()
                log({"event": "smoke_pass", "ms_per_token": r["decode_s"] / a.decode_tokens * 1e3, "prefill_s": r["prefill_s"], "same": r["tokens"] == first, "misses_per_token": r["misses"] / a.decode_tokens, "gib_read": r["gib_read"]})
            log({"event": "mlx_gib", "active": mx.get_active_memory() / 2**30, "peak": mx.get_peak_memory() / 2**30, "cache": mx.get_cache_memory() / 2**30})
            return 0

        lp = None
        if a.logger_out:
            lp = subprocess.Popen([sys.executable, str(Path(__file__).with_name("energy_logger.py")), "--out", a.logger_out, "--full"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
        try:
            s = now(); time.sleep(a.idle); log({"event": "phase", "name": "A idle-process", "start": s, "end": now(), "tokens": 0})
            s = now(); t = time.time(); tok = 0; dec = 0.0; pre = 0.0; passes = 0; diff = 0; miss = 0; gib = 0.0
            while time.time() - t < a.load:
                r = one_pass(); tok += a.decode_tokens; dec += r["decode_s"]; pre += r["prefill_s"]; passes += 1; diff += r["tokens"] != first; miss += r["misses"]; gib += r["gib_read"]
            log({"event": "phase", "name": "B all-resident", "start": s, "end": now(), "tokens": tok, "passes": passes, "decode_s": dec, "prefill_s": pre,
                 "ms_per_token": dec / tok * 1e3, "different_passes": diff, "misses": miss, "gib_read": gib})
            # P: prefill alone (reset + the same short prefill, no decode), so the prefill share of phase B can be taken out of B's power
            s = now(); t = time.time(); n = 0; pt = 0.0
            while time.time() - t < a.prefill_phase:
                rt.reset(); t0 = perf_counter(); r = rt.prefill_tokens(ids); mx.eval(r.logits); pt += perf_counter() - t0; n += 1
            log({"event": "phase", "name": "P prefill-only", "start": s, "end": now(), "tokens": 0, "prefills": n, "prefill_s": pt})
            s = now(); time.sleep(a.idle); log({"event": "phase", "name": "D idle-process", "start": s, "end": now(), "tokens": 0})
            log({"event": "mlx_gib", "active": mx.get_active_memory() / 2**30, "peak": mx.get_peak_memory() / 2**30, "cache": mx.get_cache_memory() / 2**30})
        finally:
            if lp is not None:
                lp.terminate()
                lp.wait(timeout=10)
        log({"event": "done", "clock": now()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
