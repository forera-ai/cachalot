# E3k record: a resident GLM energy arm (charter L5b, follows `docs/E3G-GLM-MINIMAX-RECORD.md` and `docs/E3I-BF16-CONTROL-RECORD.md`)

Fields 1-4 written 2026-10-16, before the run, against runtime 0.66.3. Result and explanation are appended after the run. Hamed: "go for it" (the open item "a resident GLM energy arm" of v208).

## 1. Baseline

E3g (HANDOFF 18.130): GLM's phase B ran read-bound, not resident (112.9 misses a token, 0.97 s a token, GPU 0.5 W, 36.4 J a token at 37.6 W, fit error +3.5 W = 9.3 % of PSTR), because the shipped decode miss budget 2 drops non-resident experts and so a repeated prompt never becomes resident. E3i (18.158): on DeepSeek the fit's decode gap is mostly the fit under-reporting sustained GPU streaming load, 10-16 % of PSTR whatever the format; the model's weight format explains 1-2 W. GLM's resident token has never been measured with energy; the recorded fit `GLM 92 ms + 14.70 ms x reads` puts it near 0.09 s.

## 2. Hypotheses and predictions (tags: prediction)

Intervention: the DeepSeek energy arms on `./serve-glm.sh --expert-budget-gib 46` with **`CACHALOT_GLM_DECODE_MISS_BUDGET=off`** (exact decode, prefetch off as shipped), the repeated prompt at 48 tokens so that, after the two warm-up requests, it is resident. Phases: A idle server 60 s, B repeated prompt 120 s (resident), C skipped, D idle 60 s. Hamed's root sampler beside, the logger recording all channels, `CACHALOT_ENERGY=1`.

- **Resident.** Phase B reaches at most **2 misses a token** (hit rate above 98 %) after the warm-up. Probability 75 %; if not, the arm is void as a resident arm (recorded as read-bound again).
- **Seconds a token** in B: **0.10 s (0.07 to 0.16)**.
- **System energy a token** (PSTR times seconds): **6 J (3.5 to 11)**, against 36.4 J read-bound in E3g (ratio at least 4).
- **PSTR in B:** **55 W (42 to 75)**; GPU W (IOReport) **at least 5 W**.
- **H-streaming (E3i's reading).** The fit error over PSTR in B is **8 to 18 %**. Probability 55 %.
- **H-activity.** The shortfall scales with GPU activity, so GLM's resident token shows a share close to DeepSeek's BF-80 (10 %), not GLM's read-bound 9.3 % at 0.5 W GPU: indistinguishable in share, but the absolute error follows PSTR. Predicts the error in watts at least **+3 W** and below **+8 W**.
- Idle phases A and D within +-3 W of the fit; the meter's system joules agree with the logger's integral to a median under 4 % on windows of at least 10 s.
- Preconditions: the sampler running (DRAM counter moving); fans near 1,000 RPM at the start; no other GPU user; the GLM budget 46 GiB accepted without a guardian kill; `CACHALOT_GLM_DECODE_MISS_BUDGET=off` visible in the server log's first lines.

## 3. Falsifier

Resident claim falsified if phase B shows more than **5 misses a token** (then void as resident). H-streaming is falsified if the error over PSTR in B is outside **5 to 25 %**; H-activity if the error in watts is below +3 W or above +8 W while the share is in 8-18 %. Energy a token outside 3.5-11 J is a missed prediction, reported. Void: a frozen DRAM counter (for the fit test), an idle phase off by more than 3 W, a guardian kill.

## 4. Intervention

None to the runtime. Time estimate: model load about 3 minutes, the two warm-up requests (cold first pass about 1.5 s a token for 48 tokens), 120 s of phase B and two idle minutes: **about 12 minutes**. Commands, the sampler first, one model at a time:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 1000 -o benchmarks/results/energy/e3k-sampler.txt
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && E_TAG=e3k-glm E_FULL=1 CACHALOT_ENERGY=1 CACHALOT_GLM_DECODE_MISS_BUDGET=off E_SERVE=./serve-glm.sh E_SERVE_ARGS="--expert-budget-gib 46" E_MODEL=glm-5.3-flash E_MAX_TOKENS=48 E_A=60 E_B=120 E_C=-1 E_D=60 ./benchmarks/energy_arms.sh > benchmarks/results/energy/e3k-glm-run.log 2>&1
```
