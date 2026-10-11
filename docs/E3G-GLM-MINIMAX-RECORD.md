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
