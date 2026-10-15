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
