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

## 5. Result (measured; 2026-10-16, run 00:36-00:44, `./serve-glm.sh --expert-budget-gib 46` with `CACHALOT_GLM_DECODE_MISS_BUDGET=off`, sampler `e3k-sampler.txt`, a second root sampler (E3j's) still finishing beside it; files `benchmarks/results/energy/e3k-glm-*`, local; `energy_e3.py`)

**The arm was not resident: void as a resident arm by its own falsifier.** The three phase-B requests (48 tokens, identical text each time) each missed **84.2 experts a token** (hit rate 75 %, read 43.4 ms a token in the server's line), not 5 or fewer; 1.32 s a token (0.76 tok/s), 35.1 W (sd 3.5) system power, GPU 0.5 W, 0.91 GB/s from the drive. Phase B: **46.45 J a token** (meter figures 46.1-46.5 J, 35.0-35.3 W); fit error +2.0 W (5.7 % of PSTR); idle phases A and D -2.6 and -2.3 W (within +-3).

**Against the prediction.** Resident (at most 2 misses a token, 75 %): **missed**, falsifier (more than 5) fired. Seconds a token 0.10 (0.07-0.16): missed (1.32). Energy 6 J (3.5-11): missed (46.45). PSTR 55 W: missed (35.1); GPU at least 5 W: missed (0.5). H-streaming (error 8-18 % of PSTR) and H-activity (error +3 to +8 W): **not tested**, the phase was read-bound again (error 5.7 % / +2.0 W). Idle phases within +-3 W: held. Meter against logger: not re-scored here.

## 6. Explanation (hypothesis)

The three passes missed exactly the same number of experts (84.2, 84.2, 84.2) with identical text, so the misses are not a sampling effect: the same prompt asks for the same experts each pass and still misses a fixed 84 a token. That is what a cyclic access over a set larger than the expert cache looks like under LRU (each pass evicts what the next pass needs first): 48 tokens x 43 layers x 8 experts is 16,500 expert uses, and the distinct experts of a 48-token reply plus the prefill's are more than the 46 GiB cache (about 3,400 expert slots) holds. With the budget off the token is 1.32 s, against 0.97 s with the shipped budget 2 (E3g): the 26-27 % saving of the budget, here measured in energy as well (36.4 against 46.45 J a token). A resident GLM arm therefore needs a reply short enough to fit (about 12 tokens would touch a quarter of the experts), a different design from DeepSeek's 48-token arms.

## 7. Limits

One run, three phase-B requests, same prompt; the cache-fit explanation is derived from the miss counts and not tested.

## 8. Bottleneck after

Unchanged for speed. Energy: GLM's token is 36-47 J at 35 W, waiting on the drive in every configuration measured; GLM's resident token (the GPU-bound share) remains unmeasured.

## 9. Kind

Research (predicted, measured, explained as hypothesis): the registered resident precondition failed and the arm is void as designed; the by-product (budget 2 against off, 36.4 against 46.45 J a token) is an observation, not a scored claim.

## Addendum E3l (2026-10-16, runtime 0.66.4): the resident arm with a 12-token reply

Written before the run. Hamed: "Yes, rerun the GLM resident arm with E_MAX_TOKENS=12".

**Baseline.** The 48-token arm above was void: 84.2 misses a token in each of three identical passes, 1.32 s and 46.45 J a token at 35.1 W, hypothesised cause the reply's distinct experts exceeding the 46 GiB cache under cyclic LRU.

**Intervention.** Same server and flags (`./serve-glm.sh --expert-budget-gib 46`, `CACHALOT_GLM_DECODE_MISS_BUDGET=off`, `CACHALOT_ENERGY=1`), `E_MAX_TOKENS=12`, `E_A=60 E_B=120 E_C=-1 E_D=60`; phase B then holds many short requests (each about 1.5-20 s of decode, so the meter's per-request windows are mostly under 10 s and are not scored; the phase figures come from the logger's PSTR over the phase). About 12 minutes.

**Prediction.**
- Resident: phase B at most **5 misses a token** after the warm-up (a quarter of the experts of the 48-token reply; the cache-fit hypothesis predicts it). Probability 60 %. Misses falling over the first requests of the phase and then flat is expected.
- Seconds a token **0.12 s (0.08 to 0.35)** if resident; system power **50 W (38 to 70)**, GPU at least **4 W**; energy **6 J (3.5 to 15)** a token.
- Fit error over PSTR: **8 to 18 %** if resident (H-streaming), in watts **+3 to +8 W** (H-activity), the same numbers as in the first part of this record; not scored if the arm is not resident.
- Idle phases within +-3 W.

**Falsifier.** More than **5 misses a token** in phase B: void as resident and the cache-fit hypothesis is falsified for 12 tokens as well (then a 4-token reply or a larger budget would be the next design, Hamed's call). A resident phase with the error outside 5-25 % of PSTR falsifies H-streaming; outside +3 to +8 W falsifies H-activity.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 1000 -o benchmarks/results/energy/e3l-sampler.txt
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && E_TAG=e3l-glm E_FULL=1 CACHALOT_ENERGY=1 CACHALOT_GLM_DECODE_MISS_BUDGET=off E_SERVE=./serve-glm.sh E_SERVE_ARGS="--expert-budget-gib 46" E_MODEL=glm-5.3-flash E_MAX_TOKENS=12 E_A=60 E_B=120 E_C=-1 E_D=60 ./benchmarks/energy_arms.sh > benchmarks/results/energy/e3l-glm-run.log 2>&1
```

**Result (measured; 2026-10-16, run 08:05-08:11, sampler `e3l-sampler.txt`; files `benchmarks/results/energy/e3l-glm-*`, local; `energy_e3.py`).** The arm was **resident**: after the two warm-up requests (84.5 and 127.1 misses a token, 13 and 21 s of decode) every one of the 153 phase-B requests of 12 tokens missed **0.0 experts a token** (hit 100 %), decode 0.74-0.82 s a request (**about 63 ms a token, 15-16 tok/s**; 65 ms a token over the whole phase including gaps), 1,836 tokens in 120 s. Phase B: PSTR **75.8 W** (sd 4.4), fit 64.2 W, **error +11.7 W = 15.4 % of PSTR**, IOReport cpu 5.3 / gpu 16.5 / dram 9.9 / dcs+amcc 14.7 W, drive 0.00 GB/s, **system energy 4.96 J a token** (package only 1.42 J). Idle phases A and D: -2.3 and -1.7 W. The meter's first requests ramp from 2.0 to 5.2 J (its 1 Hz sampler warming; windows under 1 s, not scored).

**Against the prediction.** Held: resident (at most 5 misses; 0.0), GPU at least 4 W (16.5), energy 6 J (3.5-15; 4.96), H-streaming (error 8-18 % of PSTR; 15.4 %), idle phases within +-3 W. Missed: seconds a token 0.12 (0.08-0.35; 0.063, below the range), system power 50 W (38-70; 75.8, above). **Falsified:** H-activity (error +3 to +8 W; +11.7 W). The cache-fit hypothesis of the first part is supported: 12 tokens fit, 48 did not.

## 6b. Explanation (hypothesis)

A resident GLM-5.3-Flash token is 63 ms and 5 J at 76 W, faster and no more costly than DeepSeek's resident token (74-80 ms; 6-7 J), and the component fit under-reports it by **15.4 % of PSTR**, the same share as DeepSeek's decode (15.3 / 15.7 %) and as every sustained GPU streaming load of E3f and E3i (13-16 %). That is a second architecture, a different quantization and a different kernel mix at the same share: the share follows sustained GPU streaming activity, not the model or the format (E3i's reading), and the watts follow PSTR (+11.7 W here at 75.8 W against +10.9 W at 71 W). The first part's read-bound figures (9.3 % and 5.7 %, GPU 0.5 W) fit the same rule from the other side: the share falls when the GPU is idle. GLM's resident token is 7-9 times cheaper than its read-bound token (4.96 against 36-47 J), so for energy the drive wait is the cost.

## 7b. Limits

Phase-B requests are 12-token replies of one repeated prompt (decode windows under 1 s, so the meter-against-logger comparison was not scored on them); one run; PSTR 75.8 W is higher than the E3i bf16-at-cadence runs (40 W) because the GPU is busier here (16.5 W); the share is one point on a curve whose other points are 10-16 %.

## 8b. Bottleneck after

Speed unchanged. Energy: GLM resident 4.96 J a token at 15-16 tok/s; the energy of a GLM token is set by the drive wait, as DeepSeek's is.

## 9b. Kind

Research (predicted, measured, explained as hypothesis): the resident precondition held this time, H-streaming held and H-activity was falsified, two numeric predictions (seconds, power) missed.
