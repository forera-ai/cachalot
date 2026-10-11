# E3g record: the energy arms on GLM-5.3-Flash and MiniMax-M3, and the meter's first live read (charter L5b, follows `docs/E3F-REAL-KERNELS-RECORD.md`)

Fields 1-4 written 2026-10-11, before the run, against runtime 0.63.0. Result and explanation are appended after the runs.

## 1. Baseline

DeepSeek (HANDOFF 18.122-18.128): system energy a token (`PSTR` times time) 5.9 to 6.8 J all-resident and 10.7 to 11.9 J read-bound; the E1b component fit under-predicts `PSTR` by 10.9 to 14.0 W, **13.9 to 19.1 % of `PSTR`** in the decode phases (E3d: 15.3 % resident, 19.1 % read-bound; E3b: 16.1 %, 17.0 %), 13.9 to 14.9 % for the model's FP8 and 2-bit kernels run alone and 14.8 % at decode's cadence (E3f; `ENERGY-KERNELS-ALONE` in `docs/LEDGER.md`). Derived after the fact (post hoc, not yet predicted): the fit is about 15 % short of `PSTR` for quantized kernels. GLM-5.3-Flash (4-bit, mixed experts; recorded token 1.5 to 1.6 s read-bound, 91 % of it waiting on the X10Pro at 0.95 GB/s, LEDGER GLM-*; decode miss budget 2 shipped) and MiniMax-M3 (3-bit, coded bank; floor 44 ms, read-bound token about 115 ms, 68 GiB budget) have not been run with energy instruments. The server now carries its own meter (0.63.0, `CACHALOT_ENERGY=1`, HANDOFF 18.129); it has been exercised once live, system energy only, on a 7-token request.

## 2. Hypotheses and predictions (tags: prediction)

Intervention in one line: the DeepSeek energy arms (phases A idle server, B repeated prompt = all-resident, C fresh prompts = read-bound, D idle server) on each of the two other servers, with `CACHALOT_ENERGY=1`, Hamed's root `powermetrics` beside, and the logger's full-channel recording; the shipped configuration of each server (GLM: `./serve-glm.sh --expert-budget-gib 46` with its decode miss budget 2, prefetch off; MiniMax: `./serve-minimax.sh`, substitution on).

- **H-format.** The fit's shortfall follows the quantized weights in general, so it is the same share of `PSTR` on GLM and MiniMax. Probability 40 %. Prediction: the fit error over `PSTR` is **10 to 20 %** in phases B and C of both models, where the GPU is working (the test is made on each phase separately).
- **H-model-specific.** The shortfall is a property of DeepSeek's kernel mix (FP8 trunk, 2-bit experts), not of quantized weights in general. Probability 35 %. Prediction: the error over `PSTR` outside **6 to 25 %** in at least one phase of one model.
- **H-activity.** The shortfall scales with the GPU's activity, not with the whole `PSTR`; GLM, which waits on its drive for 91 % of a token, shows a much smaller share (under 6 %) while MiniMax stays near 15 %. Probability 25 %. Prediction: GLM's error over `PSTR` below **6 %** in phase C, MiniMax within 10 to 20 %.
- **Energy a token** (system, `PSTR` times seconds a token; point and interval, the meter's decode figure and the logger's agree): GLM resident-style phase B **15 J (6 to 35)** at an assumed 0.15 s a token; GLM read-bound phase C **60 J (35 to 100)** (1.0 to 1.6 s a token at 45 to 60 W); MiniMax phase B **3.5 J (2.5 to 5)** (44 to 50 ms at 70 to 80 W); MiniMax phase C **8 J (6 to 11)**. The ratio C over B is **at least 3** for GLM and **1.6 to 2.6** for MiniMax.
- **The meter against the logger.** For every request with a decode window of at least 10 s, the server's `decode.system_joules` agrees with the logger's `PSTR` integral over the same window within **8 %** (median under 4 %). Below 10 s the meter's 1 Hz sampling error can reach one interval at each end; those are reported but not scored. The meter's component joules, when available, agree with the logger's IOReport differences within **8 %**.
- **Meter availability.** With the sampler up, `components.available` is true in `/v1/stats` throughout the run; in the idle phases it is true too (the DRAM counter keeps moving).
- Preconditions: the sampler running (frozen counters make the fit-error test void; the system-energy part survives); idle phases A and D with a fit error within +-3 W; fans at the start at or below 1,300 RPM; no other GPU user (Cachalot Lab closed); the GLM budget 46 GiB without a guardian kill.

## 3. Falsifier

H-format is falsified if the fit error over `PSTR` is outside **6 to 25 %** in any loaded phase of either model (then it is H-model-specific for that model) or below 6 % for GLM's phase C with MiniMax in range (H-activity). H-activity is falsified if GLM's phase C is at or above 10 %. A ratio between the bands (6 to 10 % or 20 to 25 %) is reported as unresolved. The meter claim is falsified if the median disagreement with the logger over decode windows of at least 10 s exceeds 8 %, or if the meter's `components.available` is false while the DRAM counter in the logger moves. Void: a frozen DRAM counter (for the fit test), an idle phase off by more than 3 W, a server killed by the memory guardian. Phase C of GLM is long per token; if fewer than 40 decode tokens fit, its ratio is reported with that count and not scored against the energy interval.

## 4. Intervention

None to the runtime beyond the meter shipped in 0.63.0. `benchmarks/energy_arms_driver.py` and `benchmarks/energy_arms.sh` take `E_URL`, `E_MODEL`, `E_MAX_TOKENS`, `E_A`..`E_D`, `E_SERVE` and `E_SERVE_ARGS` from the environment (defaults are the DeepSeek arms) and the driver logs each request's `usage.cachalot.energy`. Cachalot Lab closed, display as it is. Commands for Hamed, two terminals each, the sampler first, **one model at a time** (the one-runtime rule). Estimates: GLM about 14 minutes (server load, a cold first pass of the repeated prompt at about 1.5 s a token, then 6 minutes of phases), MiniMax about 10 minutes.

GLM:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 1100 -o benchmarks/results/energy/e3g-glm-sampler.txt
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && E_TAG=e3g-glm E_FULL=1 CACHALOT_ENERGY=1 E_SERVE=./serve-glm.sh E_SERVE_ARGS="--expert-budget-gib 46" E_MODEL=glm-5.3-flash E_MAX_TOKENS=48 E_A=60 E_B=90 E_C=150 E_D=60 ./benchmarks/energy_arms.sh > benchmarks/results/energy/e3g-glm-run.log 2>&1
```

MiniMax (after the GLM run has stopped and its sampler has finished or been stopped):

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 900 -o benchmarks/results/energy/e3g-mm-sampler.txt
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && E_TAG=e3g-mm E_FULL=1 CACHALOT_ENERGY=1 E_SERVE=./serve-minimax.sh E_MODEL=minimax-m3 E_MAX_TOKENS=120 E_A=60 E_B=120 E_C=150 E_D=60 ./benchmarks/energy_arms.sh > benchmarks/results/energy/e3g-mm-run.log 2>&1
```

Analysis: `benchmarks/energy_e3.py --log benchmarks/results/energy/<tag>-e3b-arms.jsonl --driver benchmarks/results/energy/<tag>-arms-driver.jsonl`, and the meter against the logger by `benchmarks/energy_requests.py` (written after the runs, from the driver's request events and the logger's rows; its definition is fixed above).

## 5. Result, GLM (measured; 2026-10-11, run 12:19:55-12:30:01, `./serve-glm.sh --expert-budget-gib 46`, decode miss budget 2, prefetch off, Cachalot Lab closed, fans 1,004 to 1,017 RPM throughout; `benchmarks/results/energy/e3g-glm-e3b-arms.jsonl`, `e3g-glm-arms-driver.jsonl`, sampler `e3g-glm-sampler.txt`; analysis `energy_e3.py` and `energy_requests.py`)

Per phase (IOReport watts, `PSTR`, the E1b fit, error = `PSTR` minus fit):

| phase | s | tokens | s/token | cpu / gpu / dram / dcs+amcc W | PSTR W (sd) | fit W | error W | error / PSTR | drive GB/s | misses/token | system J/token |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| A idle server | 60 | 0 | - | 2.1 / 0.2 / 1.2 / 4.2 | 24.5 (6.3) | 26.1 | -1.6 | - | 0 | - | - |
| B repeated prompt | 93 | 96 | 0.97 | 3.6 / 0.5 / 3.0 / 9.2 | 37.6 (3.6) | 34.1 | **+3.5** | **9.3 %** | 0.87 | 112.9 | **36.4** |
| C fresh prompts | 186 | 96 | 1.94 | 3.5 / 0.4 / 2.7 / 8.8 | 36.0 (3.6) | 33.3 | **+2.6** | **7.2 %** | 0.89 | 237.4 | **69.7** |
| D idle server | 60 | 0 | - | 2.0 / 0.2 / 1.2 / 4.1 | 24.1 (6.2) | 25.8 | -1.7 | - | 0 | - | - |

**Phase B was not all-resident.** The repeated prompt kept missing (112.9 misses a token, hit rate 72 % over the run, 10,132 experts dropped by the miss budget): a decode miss budget of 2 drops the non-resident experts beyond two a layer instead of loading them, so a repeated prompt's dropped experts never become resident. Phase B therefore ran read-bound at the X10Pro's 0.87 GB/s, 0.97 s a token, GPU 0.5 W; phase C (fresh prompts, whose seconds a token include their cold prefills) 1.94 s a token. Each phase held two requests of 48 tokens (96 tokens; above the 40-token floor of the falsifier section). The model's GPU is nearly idle throughout (0.4 to 0.5 W), the machine waiting on the drive.

**The meter against the logger (the part that is a clean test).** Seven requests, six with decode windows of at least 10 s (46 to 52 s): the server's `decode.system_joules` against the logger's `PSTR` integral over the same window: median **0.2 %**, maximum **0.4 %** (+-2.3 % on the one 8.8 s window, not scored); meter joules a token 35.9 to 39.9 against the logger's 36.1 to 39.9. The meter's DRAM and GPU component joules against the logger's IOReport differences: median **0.6 %**, maximum 6.9 % (one 4 J GPU figure on the short request; the six scored requests are within 1 %). The server's `[request]` line carried the same figures (`J/tok=38.45 W=36.8`, the meter's 38.454). `/v1/stats.energy.components.available` was true at the end of the run (sampler up).

Against the predictions. Meter: median disagreement under 4 % and every scored request within 8 %: **held** (0.2 %, 0.4 %). Meter components within 8 %: **held** (max 6.9 % on a 4 J figure). Components available throughout with the sampler up: **held** (checked at the end of the run). Energy a token, phase B 15 J (6 to 35): **missed** (36.4 J, because B was read-bound); phase C 60 J (35 to 100): **held** (69.7 J). C over B at least 3: **missed** (1.91, same cause). Idle phases within +-3 W: met (-1.6, -1.7). Fans at the start at or below 1,300 RPM: met (1,010). **Hypotheses:** H-format (10 to 20 % in B and C): B 9.3 %, C 7.2 %, both **just below the band**, in the unresolved band of the falsifier section; H-model-specific (outside 6 to 25 % in a loaded phase): **not met** (both inside 6 to 25 %); H-activity (GLM's C below 6 %): **not met** (7.2 %); its falsifier (C at or above 10 %) not met either. As registered, none of the three is falsified for GLM and the result is **unresolved**.

## 6. Explanation, GLM

For GLM the fit's shortfall is 2.6 to 3.5 W of 36 to 38 W, and a drive is reading at 0.87 to 0.89 GB/s. The component fit does not name the X10Pro, and E2 (HANDOFF 18.121) measured it at +2.71 W (se 0.85) at 0.99 GB/s, about 2.4 W at 0.88 GB/s. Subtracting that **derived** drive term (post hoc; it carries E2's +-0.9 W) leaves about **+1.1 W in B and +0.2 W in C**, that is 3 % and 0.6 % of `PSTR`, inside the fit's noise. So on GLM, whose GPU draws 0.4 to 0.5 W because the machine waits on its drive, there is no kernel shortfall to see, which agrees with H-activity (the shortfall follows the GPU's work, as in E3f where the same fit error grew with the kernels' load) and not with a share of `PSTR`; but the registered test, as predicted, did not decide between H-format and H-activity, and the decision rests on a drive term taken from another run. MiniMax's phases, where the GPU works, are the test that separates them.

The result for the energy lane that stands without that caveat: GLM costs 36 to 70 J a token at 36 to 38 W of system power (idle 24 W), nearly all of it waiting; 3 to 4 times DeepSeek's resident token and more than 3 times its read-bound token.

## 7. Generalization (GLM)

Model- and workload-specific. The meter works on the GLM server (the one engine MiniMax shares), agrees with an independent integral to under half a percent over 46 to 52 s windows, and reports its components when the sampler is up. A shipped GLM decode miss budget of 2 keeps a repeated prompt from becoming resident, so a "resident" GLM arm needs the budget off.

## 8. Result, MiniMax (measured; 2026-10-11, run 12:31:56-12:38:45, `./serve-minimax.sh` shipped configuration (68 GiB, substitution on), Cachalot Lab closed, fans 1,015 RPM at the start rising to 2,075 (mean 1,409) as the run warmed the machine; `benchmarks/results/energy/e3g-mm-e3b-arms.jsonl`, `e3g-mm-arms-driver.jsonl`, sampler `e3g-mm-sampler.txt`)

| phase | s | tokens | ms/token | cpu / gpu / dram / dcs+amcc W | PSTR W (sd) | fit W | error W | error / PSTR | drive GB/s | misses/token | system J/token |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| A idle server | 60 | 0 | - | 2.0 / 0.2 / 1.2 / 4.2 | 24.2 (5.3) | 25.8 | -1.7 | - | 0 | - | - |
| B repeated prompt (resident) | 121 | 2,880 | 42 | 9.3 / 41.9 / 16.2 / 19.9 | 127.9 (7.1) | 108.4 | **+19.5** | **15.2 %** | 0.67 | 1.2 | **5.38** |
| C fresh prompts (read-bound) | 125 | 1,200 | 104 | 7.2 / 20.4 / 10.4 / 15.6 | 89.6 (16.3) | 72.6 | **+17.0** | **19.0 %** | 3.26 | 15.8 | **9.34** |
| D idle server | 61 | 0 | - | 4.1 / 0.1 / 1.7 / 6.6 | 31.1 (8.6) | 31.2 | -0.1 | - | 0 | - | - |

MiniMax's resident token draws 128 W, far above DeepSeek's 71 to 77 W (GPU 42 W against 15). Hit rate 97.5 % over the run, no experts skipped, MLX active 74 GiB. Components were available throughout (checked at the end of the run).

**The meter against the logger.** Thirty-nine requests; the six with decode windows of at least 10 s (10.7 to 13.2 s, phase C): system joules median **0.6 %**, maximum **5.6 %** (the prediction: median under 4 %, all within 8 %: **held**); 32 shorter windows (4.4 to 9.2 s) not scored, with the system figure within +-1.5 % for 29 and 3.6 to 7.3 % for the other three. **The meter's component joules missed the 8 % prediction**: against the logger's IOReport differences, median 7.3 %, maximum 32.4 %, with the meter systematically below the logger (for example DRAM 77 against 78 to 89 J, GPU 195 against 219 J on 5 s windows; the six scored C windows 4.7 to 9.0 % low). Cause found: the meter extended a window that ended after its last sample with the system figure at its last watts but held the components at the last sample's value, so a window lost up to one interval of component energy at its end (10 % of a 5 s window; 1 % of GLM's 50 s windows, hence GLM's 0.6 %). Fixed in 0.63.1 (each component continues at its last observed rate for up to two intervals, as the system figure does), with a unit test. **Offline check of the fix, not a live re-run:** replaying the logger's own once-a-second rows through the meter, truncated at each request's end, the old rule reproduces the live bias (median error -7.9 % over 72 DRAM and GPU figures), the fixed rule has median error 0.0 %, median absolute 0.5 %, maximum absolute 13.6 %. The replay uses the logger's samples as the meter's, so it tests the arithmetic, not the two processes' sampling phases against each other.

Against the predictions. Energy a token, phase B 3.5 J (2.5 to 5): **missed** (5.38 J: the machine draws 128 W, not the 70 to 80 W assumed); phase C 8 J (6 to 11): **held** (9.34). C over B 1.6 to 2.6: **held** (1.74). Idle phases within +-3 W: met (-1.7, -0.1). Fans at the start at or below 1,300: met (1,015); the run warmed to 2,075 RPM, which can move a condition by about 1 W (E3f). Hypotheses (MiniMax part): error over PSTR 10 to 20 % in B and C: **held** (15.2 %, 19.0 %), the band the DeepSeek decode phases sit in (13.9 to 19.1 %).

## 9. Result summed over the two models, and the verdict on the hypotheses

| model | phase | GPU W | error W | error / PSTR | drive GB/s |
|---|---|---:|---:|---:|---:|
| DeepSeek (E3d/E3e) | resident / read-bound | 15.5 / 11.3 | +10.9 / +14.0 | 15.3 % / 19.1 % | 0 / 2.1 |
| MiniMax | resident / read-bound | 41.9 / 20.4 | +19.5 / +17.0 | 15.2 % / 19.0 % | 0.7 / 3.3 |
| GLM | B / C (both read-bound) | 0.5 / 0.4 | +3.5 / +2.6 | 9.3 % / 7.2 % | 0.87 / 0.89 |

As registered: H-format (10 to 20 % in B and C of both): MiniMax inside the band, GLM B 9.3 % and C 7.2 % just below it, in the unresolved band; H-model-specific (outside 6 to 25 % somewhere): **not met**, falsified; H-activity (GLM C under 6 %, MiniMax in 10 to 20 %): MiniMax met, GLM **missed narrowly** (7.2 %), and its falsifier (GLM C at or above 10 %) not fired. So the pre-registered criteria leave H-format and H-activity both partly met, neither falsified, H-model-specific falsified. What the data say beyond the criteria (derived, post hoc): on two different quantized model families (DeepSeek's FP8/2-bit, MiniMax's 3-bit coded experts) wherever the GPU works hard the component fit is 14 to 19 % of PSTR short, whether the phase is resident or read-bound and at GPU powers from 11 W to 42 W; on GLM, whose GPU draws 0.5 W, it is 7 to 9 % raw and, after the derived X10Pro term (E2: about 2.4 W at 0.88 GB/s), 0.6 to 3 %. The shortfall therefore follows the GPU's quantized-kernel work (the mechanism of H-activity, E3f) and its size is about 15 to 19 % of the draw when the GPU is busy; it is not specific to DeepSeek.

## 10. Generalization

Model-family-general across DeepSeek and MiniMax on this Mac (two quantizations, two kernel mixes), with GLM as the idle-GPU control; hardware-specific. The energy lane's rule stands and is now supported on three models: take the total from the SMC system rail; a component sum from the IOReport fit under-reports a quantized-weight workload by about 15 to 19 % of the draw when the GPU is busy. The meter (0.63.0 and the 0.63.1 tail fix) agrees with an independent system integral to under 1 % on windows of 10 s or more on both servers; its component joules are only as good as 1 Hz sampling (one interval of error at each end of a window).

## 11. Bottleneck after

Speed unchanged. Energy: the instruments exist; the open items are the meter's cost on a decode token (no server A/B), a live re-run to confirm the tail fix with two processes sampling independently, and showing the figures in Cachalot Lab (Hamed's and Lab's side).

## 12. Kind

Research (predicted, measured, explained in part): the meter claim held for system energy and missed for components, with the cause found and fixed; the shortfall hypotheses are partly decided (model-specific falsified, format and activity both partly met); GLM's phase B was not the all-resident arm it was named for and is recorded as read-bound.
