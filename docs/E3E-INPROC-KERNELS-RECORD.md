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

## 5. Result (measured; 2026-10-11, run 00:03:28-00:09:54, Cachalot Lab closed, fans 1,004-1,121 RPM (mean 1,033); `benchmarks/results/energy/e3e-e3b-arms.jsonl`, `e3e-arms-driver.jsonl`, sampler `e3e-sampler.txt`; analysis `benchmarks/energy_e3.py` and the prefill correction below)

Per phase (IOReport watts, `PSTR`, the E1b fit, error = `PSTR` minus fit):

| phase | s | PSTR W (sd) | fit W | error W | cpu / gpu / dram / dcs+amcc W |
|---|---:|---:|---:|---:|---|
| A idle process | 90 | 28.6 (8.1) | 28.3 | +0.2 | 3.0 / 0.5 / 1.5 / 4.8 |
| B all-resident (decode + prefill) | 122 | 76.9 (8.6) | 66.1 | **+10.9** | 5.3 / 18.3 / 10.0 / 14.3 |
| P prefill only | 45 | 95.4 (2.2) | 87.6 | **+7.8** | 6.5 / 33.3 / 9.5 / 15.8 |
| D idle process | 90 | 24.1 (6.3) | 26.5 | -2.3 | 2.2 / 0.1 / 1.2 / 4.5 |

Phase B ran 40 passes of 16 prompt tokens and 32 decode tokens: decode 101.47 s (1,280 tokens, **79.27 ms a token**), prefill 21.09 s, so prefill was 17.2 % of the phase; all 40 passes reproduced the first pass's tokens, 0 misses, 0 GiB read; MLX active memory 62.47 GiB (the server: 63.0). **Prefill correction** (power is a time-weighted mean, the fit is linear in the components, so the error is too): decode-only error = (10.9 - 0.172 x 7.8) / 0.828 = **+11.5 W**; decode-only PSTR = (76.9 - 0.172 x 95.4) / 0.828 = **73.1 W**; system energy a token = 73.1 W x 79.27 ms = **5.79 J**. The uncorrected and corrected errors (10.9 and 11.5 W) fall in the same band.

Against the predictions: decode-only error 10.5 W (7.5 to 13.5): **held** (+11.5). Decode-only system power 71 W (65 to 77): **held** (73.1). Seconds a token 80 ms (76 to 86): **held** (79.3). Energy a token 5.7 J (5.0 to 6.5): **held** (5.79). Preconditions: identical tokens in every pass, 0 misses, 0 bytes read, MLX memory within 0.6 GiB of the server's, idle A within +-3 W (+0.2), idle D -2.3 W (inside +-3), prefill share 17 % (below 25 %): all met. Falsifiers: error at or below 6 W (server work): **not fired**; above 13.5 W: **not fired**. H-server (error at or below 6 W, falsified above 8 W): **falsified**.

## 6. Explanation

Taking the server away does not change the excess: the in-process decode draws +11.5 W above the component fit against the server's +10.9 W (E3d), at the same speed (79.3 against 83 ms a token, the server's 3.7 ms being its own CPU and plumbing) and the same system energy a token (5.79 against 5.90 J). The server's work is not where the 11 W is; it costs a few percent of a token in time and nothing resolvable in power. Prefill alone, which has no drive reads and no per-token synchronization, also sits 7.8 W above the fit (95.4 W against 87.6), and the fit is off for the same reason there, so the excess is not specific to decode's token-by-token structure: it follows the model's real GPU work. What this run cannot separate is the kernels from the resident 62 GiB: both stay. The E1b fit was made on a synthetic matmul and streaming loads (errors -6.5 to +4.2 W there); the model's quantized kernels, with a different mix of ALU, cache and memory activity, draw more at the system rail than the same IOReport components imply. That is the simplest reading of every result from E3b on, but it is a reading, not a test.

## 7. Generalization

Hardware-specific (this Mac, DeepSeek 2-bit bank, one session, cool state). The transferable statement: a component power fit calibrated on synthetic single-source loads under-predicts a real quantized-MoE workload by about 11 W at the system rail (15 % of the draw), in decode and in prefill, with no server, drive, prediction or read-pattern involved; joules-per-token figures have to come from the system rail.

## 8. Bottleneck after

Speed: unchanged (nothing shipped). Energy: about 11 W (decode) and 8 W (prefill) of system power that the IOReport components do not account for in a real model run, whose cause (the kernels' own behavior against the 62 GiB of resident memory) is one test away: (iii) the same in-process decode at a smaller expert budget, with the working set unchanged.

## 9. Kind

Research (predicted, measured, explained in part): every numeric prediction held, H-server was falsified, H-kernels survived its falsifier; the split between the kernels and the resident memory is not made.
