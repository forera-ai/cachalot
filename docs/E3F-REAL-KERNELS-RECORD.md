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

## 5. Result (measured; 2026-10-11, run 11:28-11:38, Cachalot Lab closed, no model loaded; `benchmarks/results/energy/e3f-kernels.jsonl` and `.header.json`, sampler `e3f-sampler.txt`; analysis `kernel_power.py --analyze`)

Per step (achieved GB/s of weights, PSTR W, the E1b fit's error against PSTR W, fan RPM; pass A then pass B in the opposite order): idle 0.0, 31.0, +1.6, 1,009 / 0.0, 34.5, +0.8, 2,382; **TOK-80** 95.0, 59.2, **+8.3**, 1,011 / 95.2, 62.3, **+9.7**, 2,315; TOK 239.3, 112.7, +16.7 / 229.3, 116.8, +17.5; TOK-NS 404.9, 163.8, +23.8 / 434.2, 171.4, +25.1; FP8 198.8, 106.8, +14.5 / 218.3, 107.3, +15.2; Q2 105.1, 83.1, +11.7 / 106.4, 82.4, +11.9. Means by condition: idle +1.2; **TOK-80 +9.0 W** (PSTR 60.8 W); TOK +17.1; TOK-NS +24.5; FP8 +14.9; Q2 +11.8. Decode for comparison: +10.9 (server, 71.3 W) and +11.5 W (in-process, 73.1 W). Fans rose from 1,005 to 2,402 RPM through the run (the run's own heat): the B pass reads 0.2 to 1.4 W above the A pass for the same condition (TOK-80 +8.3 then +9.7).

Against the predictions: TOK-80 at least +8 W (H-real-kernels): **held** (+9.0, both passes above 8); TOK-80 point prediction +4 W (0 to 9): **missed** (at the edge, on the side of the other hypothesis); TOK +3 W (-3 to 8): **missed** (+17.1); TOK-NS within 3 W of TOK: **missed** (+24.5, 7.4 W above); FP8 +2 W (-4 to 8): **missed** (+14.9); Q2 +2 W (-4 to 8): **missed** (+11.8). Preconditions: TOK-80 rate 95.0 / 95.2 GB/s (86 to 100): met; idle steps +1.6 and +0.8 (within +-3): met; sampler running: met; fans at the start 1,007 to 1,011 RPM: met. Falsifiers: TOK-80 below +5 W (H-real-kernels): **not fired**; at or above +8 W (H-rest-of-model): **fired**, so H-rest-of-model is falsified.

## 6. Explanation

The model's own FP8 and 2-bit kernels, run alone on random buffers with no model, no attention, no router, no Python decode loop beyond a launch and a sync per layer, no server, no drive and only 7.4 GB of resident buffers, at the weight rate and cadence of decode, draw **+9.0 W** above the component fit, against +10.9 and +11.5 W for the real decode. The kernels explain most of the unexplained decode power (about 80 %: 9.0 of 11.2 W); the remaining 2 W are inside the thermal drift of this run (+1.4 W between passes) and the fit's noise. The earlier E3b to E3e exclusions (storage, prefetch, prediction, read pattern, server) all stand, and now the resident 62 GiB and the rest of the model's work are not needed to explain the excess either.

A regularity in the data, seen after the numbers were in and therefore a derived hypothesis, not a prediction: the fit's error is a nearly constant share of PSTR for every real-kernel load. Error over PSTR: TOK-80 14.8 %, TOK 14.9 %, TOK-NS 14.6 %, FP8 13.9 %, Q2 14.3 %, real decode 15.3 % (server) and 15.7 % (in-process). That is what a multiplicative miss looks like (a coefficient too low, or a conversion loss the IOReport components do not carry for these kernels), not an additive power from some component. The E1b fit was calibrated on a bf16 matmul, `mx.sum` streams and CPU loads, where its errors were -7 % to +4 %; the quantized kernels (a dequantize inside the multiply-accumulate) differ from those in what the GPU energy channel reports against what the supply delivers. Untested: a bf16 GEMV at the same rate on the same buffers' scale (does the error vanish with the format?), and the same ratio on GLM's and MiniMax's kernels.

## 7. Generalization

Hardware- and kernel-format-specific (M3 Ultra, MLX's FP8 and 2-bit affine quantized kernels, one cool-to-warm session). The transferable statement: a per-component power model calibrated on dense bf16 kernels under-reports a quantized-weight workload at the system rail by about 15 % of the draw here; the system-rail measurement (SMC `PSTR`) is the total, and a model of a token's energy by component should carry a kernel-format term (about +15 % of PSTR for these formats) or be calibrated on the real kernels.

## 8. Bottleneck after

Speed: unchanged (nothing shipped). Energy: the split of a token's system energy by component stays unavailable beyond the PSTR total, but the unexplained decode power is now explained to about 80 % by the kernels' own draw (a fit-calibration gap, about 15 % of PSTR) and not by anything in the runtime. Open: whether the gap follows the quantized format (a bf16 control) and whether it carries to GLM and MiniMax (their E3 arms with the prediction "error about 15 % of PSTR").

## 9. Kind

Research (predicted, measured, explained in part): the headline hypothesis held and its alternative was falsified; four of the five secondary point predictions missed and are recorded as missed; the 15 % regularity is a post-hoc hypothesis.
