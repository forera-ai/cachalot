# E3e record: do the model's kernels alone, without the server, draw the unexplained decode power? (charter L5b, follows `docs/E3D-NO-PREFETCH-RECORD.md`)

Fields 1-4 written 2026-10-10, before the run, against runtime 0.62.39. Result and explanation are appended after the run.

## 1. Baseline

E3d (HANDOFF 18.126): through the server with `CACHALOT_PREDICT_TOPK=0`, the all-resident decode read 0.00 GB/s from the drive and the component fit still under-predicted `PSTR` by **+10.9 W** (PSTR 71.3 W, 83 ms a token, 5.90 J a token, fans about 1,020 RPM, idle 28.2 W); E3b with the prefetch on, +12.6 W. About 11 W of decode power is therefore not storage, not a read pattern, not the prediction's work. What is left: the model's kernels (the GPU work of the quantized matrix products and attention, with the CPU work that issues them), the server's own CPU work (HTTP handling, tokenizer and stream plumbing, the stats thread, the engine loop), and the 62 GiB of wired, resident memory. Smoke check of this session's instrument (no sampler, `SMOKE=1`): the in-process all-resident decode runs at 76-87 ms a token (median 80) with 0.0 misses a token and 0 bytes read, MLX active memory 62.5 GiB (the server showed 63.0 GiB), a pass of 16 prompt tokens plus 32 decode tokens reproduces the same tokens every time, and its short prefill takes 0.52 s against 2.6 s of decode (about 17 % of a pass).

## 2. Hypotheses and predictions (tags: prediction)

Intervention in one line: the same all-resident decode driven by `TextDecodeRuntime` directly in one process (no HTTP server, no tokenizer streaming, no snapshot store, no stats), `CACHALOT_PREDICT_TOPK=0`, 48 GiB budget and the serve.sh environment, sampled with the same instruments as E3d. A fourth phase (P, prefill alone) measures the power of the pass's prefill so its share can be taken out of phase B.

- **H-kernels.** The excess belongs to the model's kernels and the resident memory, so it stays when the server is removed. Probability 50 %. Predictions: the decode-only fit error (B corrected for the prefill share) **+10.5 W (7.5 to 13.5)**; decode-only system power **71 W (65 to 77)**; seconds a token **80 ms (76 to 86)**; system energy a token **5.7 J (5.0 to 6.5)**.
- **H-server.** The server's CPU work explains a large part of it. Probability 35 %. Predicts a decode-only fit error **at or below 6 W**, the CPU channel lower than E3d's 4.3 W in phase B, and PSTR lower by about the same amount.
- **H-other.** Something neither switch touches, or a hotter session. Probability 15 %. Predicts a decode-only fit error **above 13.5 W**, or an error that follows the idle watts instead of the arm.
- Phase P (prefill alone): its fit error is reported; no prediction on its size (it is a correction term, not a hypothesis).
- Preconditions: every pass in B reproduces the first pass's tokens; misses a token below 0.5 and the drive below 0.05 GB/s (the arm is all-resident); MLX active memory within 1.5 GiB of the server's 63.0; idle phases A and D with fit error within +-3 W; the prefill share of B below 25 % of its wall time.

## 3. Falsifier

H-kernels is falsified if the decode-only fit error is **at or below 6 W** (the server's work was a large part) or **above 13.5 W**. H-server is falsified by an error above 8 W. An error between 6 and 7.5 W or between 13.5 and 14.5 W is reported as unresolved against the fit's 1.3 to 3 W noise, not as a result. The prefill correction assumes the average power of B is the time-weighted mean of decode and prefill; if the corrected value moves the error by more than 3 W from the uncorrected one, both are reported and the conclusion is drawn only if both fall in the same band. Void: a failed precondition, a sampler not running (frozen counters), an idle phase off by more than 3 W.

## 4. Intervention

None to the runtime. New: `benchmarks/decode_power_inproc.py` (phases A idle 90 s, B all-resident 120 s, P prefill only 45 s, D idle 90 s; per-pass decode and prefill seconds, misses and bytes read; it starts `energy_logger.py --full` itself after the runtime is up) and `benchmarks/inproc_arms.sh` (the serve.sh environment with `CACHALOT_PREDICT_TOPK=0`, `E_TAG` prefix, `SMOKE=1` precheck). Cachalot Lab closed, display as it is. Commands for Hamed, two terminals, the sampler first:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 700 -o benchmarks/results/energy/e3e-sampler.txt
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && E_TAG=e3e ./benchmarks/inproc_arms.sh > benchmarks/results/energy/e3e-run.log 2>&1
```

About 8 minutes of run (model load about 20 s, warm-up about 20 s, 90 + 120 + 45 + 90 s of phases); the sampler's 700 samples cover it with margin. Analysis: `benchmarks/energy_e3.py --log benchmarks/results/energy/e3e-e3b-arms.jsonl --driver benchmarks/results/energy/e3e-arms-driver.jsonl` and the prefill correction by hand from phases B and P.
