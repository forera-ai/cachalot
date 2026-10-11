# E3f record: do the model's own quantized kernels, run alone, draw the unexplained decode power? (charter L5b, follows `docs/E3E-INPROC-KERNELS-RECORD.md`)

Fields 1-4 written 2026-10-11, before the run, against runtime 0.62.40. Result and explanation are appended after the run.

## 1. Baseline

E3d and E3e (HANDOFF 18.126, 18.127): the all-resident DeepSeek decode draws **+10.9 W** (through the server) and **+11.5 W** (in-process, no server, prefill share removed) above the E1b component fit, at 83 and 79.3 ms a token (PSTR 71 to 73 W, about 94 GB/s of weights); prefill alone +7.8 W. Excluded: the drive, the prefetch loads, the prediction's work, the read pattern, the server's CPU work, fans. Not separated: the model's quantized kernels, the rest of the model's work (attention, hyper-connections, the router, the Python and CPU side) and the 62 GiB of resident, wired memory. Test (iii), a smaller expert budget, cannot be made all-resident below about 44 GiB (smoke checks this session: 40 GiB still misses 1.7 experts a token, 28 GiB 4.5 to 11.6, because a prefill chunk's expert set alone is several GiB), and 8 GiB of 62 would move a proportional memory effect by only about 1.4 W, below the fit's noise; so it was dropped for this test (Hamed's choice, 2026-10-11). The E1b fit was made on a synthetic bf16 matmul, `mx.sum` streams and CPU loads; its errors on those were -6.5 to +4.2 W (rms 3.0 W).

## 2. Hypotheses and predictions (tags: prediction)

No model is loaded. The two kernels that read nearly all of a token's weight bytes are run on distinct random buffers in the model's own formats: the trunk's FP8 GEMV (`fp8_linear_quantized`, 8 MiB launches, 5.04 GiB a token) and the routed experts' 2-bit group-128 `mx.quantized_matmul` (3 MiB launches, 2.36 GiB a token), read once per token over 43 layer groups with a host synchronization after each group, as decode does. Conditions, 40 s each, two passes in opposite order, idle between passes' ends: **TOK** (both kernels, sync per layer, as fast as the GPU goes, about 240 GB/s), **TOK-80** (the same padded to 80 ms a token, about 95 GB/s, decode's cadence and weight rate), **TOK-NS** (the whole token in one evaluation, about 400 GB/s), **FP8** (trunk kernel alone), **Q2** (expert kernel alone), **idle**.

- **H-real-kernels.** The model's real quantized kernels, at decode's cadence, already draw the excess that the fit (made on bf16 and streaming loads) does not name. Probability 30 %. Predicts TOK-80's fit error **at least +8 W**.
- **H-rest-of-model.** The kernels alone sit near the fit; the excess comes from what a real model run has and these loops lack (attention and the other kernels, the CPU and Python side with 40 syncs and threads, the resident 62 GiB). Probability 50 %. Predicts TOK-80's fit error **below +5 W**.
- Between (+5 to +8 W): unresolved, 20 %.
- Point prediction for TOK-80: **+4 W (0 to 9)**. TOK (240 GB/s): **+3 W (-3 to 8)**; TOK-NS: within **3 W** of TOK; FP8 alone **+2 W (-4 to 8)**; Q2 alone **+2 W (-4 to 8)**.
- Preconditions: TOK-80's rate **94 GB/s (86 to 100)**; idle steps' fit error within +-3 W; the sampler running (DRAM counter advancing); fans within 1,000 to 1,300 RPM at the start (a cool session, comparable to E3d and E3e).

## 3. Falsifier

H-real-kernels is falsified if TOK-80's fit error is **below +5 W** (then H-rest-of-model); H-rest-of-model is falsified by an error **at or above +8 W**. A value between +5 and +8 W is reported as unresolved against the fit's 1.3 to 3 W noise. Because the fit is linear but real power need not be, the error is reported for every condition (TOK at 240 GB/s and TOK-NS at 400 GB/s load the GPU 2.5 to 4 times harder than decode; they are context, not the comparison; TOK-80 is). A run with a frozen DRAM counter, an idle step off by more than 3 W, or TOK-80's rate outside 80 to 105 GB/s is void.

## 4. Intervention

None to the runtime. New: `benchmarks/kernel_power.py` (full-channel logging, `--analyze` re-reads a run, `--no-check` is the smoke test; smoke-tested this session without the sampler: buffers built in 5 s, rates TOK 230 to 245, TOK-80 95, TOK-NS 377 to 415, FP8 187 to 221, Q2 103 to 112 GB/s). Cachalot Lab closed, no other GPU user. About 9 minutes of run (2 passes x 6 conditions x 40 s plus set-up); the sampler's 700 samples cover it with margin. Commands for Hamed, two terminals, the sampler first:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 700 -o benchmarks/results/energy/e3f-sampler.txt
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/kernel_power.py > benchmarks/results/energy/e3f-run.log 2>&1
```
