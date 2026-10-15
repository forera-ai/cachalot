# E3i record: does the component fit's decode gap follow the weight format? A bf16 control (charter L5b, follows `docs/E3F-REAL-KERNELS-RECORD.md`)

Fields 1-4 written 2026-10-15, before the run, against runtime 0.66.0. Result and explanation are appended after the run.

## 1. Baseline

E3f (HANDOFF 18.128): the model's own quantized kernels (FP8 GEMV, 2-bit `quantized_matmul`) on random buffers at decode's cadence (TOK-80, 95 GB/s) draw **+9.0 W** above the E1b component fit (PSTR 60.8 W), 14.8 % of PSTR; every real-kernel load showed 13.9-15.3 % of PSTR; real decode +10.9 / +11.5 W (15.3 / 15.7 %). The E1b fit was calibrated on a bf16 matmul, `mx.sum` streams and CPU loads (errors -7 % to +4 %). E3f's own limit: "no bf16 control".

## 2. Hypotheses and predictions (tags: prediction)

New conditions in `benchmarks/kernel_power.py`: **BF** (dense bf16 GEMVs `w @ x`, the same launch sizes, 15 x 8 MiB and 19 x 3 MiB per layer group, 43 groups, a host sync per group, as fast as the GPU goes) and **BF-80** (the same padded to 80 ms a token). Same bytes a token as TOK (7.43 GiB of buffers, about 95 GB/s of weights at 80 ms), only the format differs. Note the confound: bf16 kernels move the same bytes faster, so the GPU is busy a smaller share of the 80 ms in BF-80 than in TOK-80; the comparison at equal bytes and cadence is the right one for "format", and BF (unpadded, higher rate) is context.

- **H-format.** The gap is a property of the quantized formats (dequantization inside the multiply-accumulate draws power the component fit does not carry). Predicts BF-80's fit error **below +4 W** and below **7 % of PSTR**.
- **H-not-format.** The gap is a property of sustained GPU work at this cadence whatever the format (the fit under-reports GPU draw in this operating region). Predicts BF-80's fit error **at or above +6.5 W**.
- Between (+4 to +6.5 W): unresolved.
- Point predictions (probability H-format 55 %): BF-80 **+3 W (-1 to +8)**, PSTR about 55 W; BF **+5 W (-1 to +12)**; idle steps within +-3 W; all other conditions repeat E3f within 3 W (TOK-80 +9.0, Q2 +11.8, FP8 +14.9).
- Preconditions: BF-80's rate **94-96 GB/s** (7.43 GiB in 80 ms); sampler running (DRAM counter advancing, the script checks); fans near 1,000-1,300 RPM at the start; idle steps within +-3 W; otherwise void.

## 3. Falsifier

H-format is falsified if BF-80's fit error is **at or above +6.5 W**; H-not-format is falsified if it is **below +4 W** in both passes. A mean between is reported as unresolved. Both passes (A then B in opposite order) must agree on the side, else unresolved.

## 4. Intervention

None to the runtime. `benchmarks/kernel_power.py` gains the BF and BF-80 steps and a 7.43 GiB bf16 buffer set (smoke-tested without the sampler this session: steps run in both passes). Run time: 8 conditions x 2 passes x 40 s = about 11 minutes plus about 20 s of set-up; the sampler's 900 samples cover it. Commands, the sampler first:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 900 -o benchmarks/results/energy/e3i-sampler.txt
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/kernel_power.py --tag e3i > benchmarks/results/energy/e3i-run.log 2>&1
```

## 5. Result (measured; 2026-10-15, run 23:48-23:58, after a reboot, no model loaded, sampler `e3i-sampler.txt`; `benchmarks/results/energy/e3f-e3i.jsonl`, local)

Per condition, mean of the two passes (A then B in opposite order): rate GB/s, PSTR W, fit error W (error / PSTR): idle 0.0, 23.8, -2.0; **BF-80 95.6, 40.0, +4.0 (3.9, 4.0; 10.0 %)**; TOK-80 93.7, 46.4, +5.3 (4.4, 6.1; 11.4 %); BF 283.2, 87.2, +13.8 (15.9 %); FP8 264.2, 105.4, +13.6 (12.9 %); Q2 122.4, 76.7, +10.4 (13.6 %); TOK 273.7, 114.5, +16.4 (14.3 %); TOK-NS 491.5, 162.7, +24.7 (15.2 %). Fans 995 RPM at the start, 2,109 at the end. Decode for comparison: +10.9 / +11.5 W at PSTR 71-73 W.

## 6. Against the prediction

- BF-80 point prediction +3 W (-1 to +8): **held** (+4.0). Rate 95.6 GB/s (94-96): met. Idle -2.0 and within +-3: met. Sampler running: met. Fans near 1,000: met.
- H-format (BF-80 below +4 W and below 7 % of PSTR) vs H-not-format (at or above +6.5): BF-80 is +3.9 / +4.0 W and 10.0 % of PSTR, between the thresholds, so by the registered rule the headline is **unresolved**. H-format's two conditions both fail narrowly (+4.0 is not below +4, 10 % is not below 7 %); H-not-format's +6.5 is not reached.
- BF +5 W (-1 to +12): **missed** (+13.8).
- All other conditions repeat E3f within 3 W: **missed for TOK-80** (+5.3 against +9.0; PSTR 46.4 against 60.8 W; idle also lower, 23.8 against 31-34 W, a cooler or quieter machine after the reboot); held for TOK (+16.4 against +17.1), TOK-NS (+24.7 against +24.5), FP8 (+13.6 against +14.9) and Q2 (+10.4 against +11.8).

## 7. Explanation (hypothesis, not tested further)

At equal bytes and cadence the quantized kernels draw only about **1.3 W** more fit error than bf16 (TOK-80 +5.3 against BF-80 +4.0), and at high rate bf16 (BF, 283 GB/s, +13.8 W) matches the FP8 trunk kernel (264 GB/s, +13.6 W). So most of the gap that E3f attributed to "the model's quantized kernels" appears with dense bf16 GEMVs as well. The better reading is that the E1b component fit under-reports sustained GPU streaming load in general, by about 10-15 % of PSTR, and that the weight format contributes at most a small part (about 1-2 W of decode's ~11 W). E3f's statement that the quantized kernels explain about 80 % of the unexplained power is therefore **weakened to "sustained GPU streaming at decode's cadence"**; what E3f showed (it is the GPU work, not the drive, prefetch, server or pattern) stands.

## 8. Limits

Two passes, one session; the machine state differs from E3f's (PSTR lower at the same bytes), so the E3f-to-E3i comparison of TOK-80 is itself not controlled; the bf16 GEMV is `w @ x` through MLX's own kernel, not a hand-written one, and its GPU utilisation at 80 ms is lower than the quantized kernels' (3.5 against 7 W GPU), the confound named in the prediction.

## 9. Bottleneck after

Speed: unchanged (nothing shipped). Energy: the decode gap is a property of sustained GPU load in the fit, not of the quantized format; a fit calibrated on high-GPU-utilization bf16 loads, not just on idle-to-moderate ones, is the open fix.

## 10. Kind

Research (predicted, measured, explained as hypothesis): the headline is unresolved by the rule, one point prediction held, two missed; the supporting comparison (BF against FP8) points against H-format.
