# E-2 and E-3: GLM's read-bound decode beside the system model, and the power of the system model beside the GPU (experiment record; predictions before any build or run)

Written 2026-10-30 against runtime 0.69.3, **before the drivers for these runs are written or anything is run**. Charter
section 7.2 fields 1 to 3; results go in `docs/APPLE-FM-E2-E3-RESULTS.md`. A prediction edited after its run is void.
Hamed asked for both on 2026-10-30: "Go for GLM and power measurement steps. Do the remaining steps."

## E-2: GLM-5.3-Flash read-bound decode, with and without auxiliary classification beside it

**Baseline (dated, measured).** GLM's experts stream from the X10Pro (~1 GB/s): a mixed token 1.52 s exact with K = 5
prefetch (0.61.1), 1.61-1.73 s in a live Hermes replay at 46 GiB (0.61.13; prefetch off since 0.62.14); the shipped decode
miss budget 2 cuts a token ~31 % (0.62.20); ~70 % of a token is waiting on the store. E-1 on DeepSeek's all-resident decode
moved a token by under ~1 ms either way (0.69.3). The system model's own weights are not read from the X10Pro.

**Hypothesis (H-e2).** A read-bound GLM token is mostly drive wait, which the Neural Engine does not touch; the system
model can only reach it through the GPU part (~30 %) and the host, so the relative effect is smaller than on DeepSeek's
all-resident token.

**Method.** One fresh `./serve-glm.sh --expert-budget-gib 46` (shipped defaults otherwise: bank on the X10Pro, budget 2,
prefetch off) under `guarded_run.sh --budget-gib 46 --force`, with `CACHALOT_GLM_SNAPSHOT_DIR=` (empty). The E-1 driver
(`benchmarks/apple_fm_e1.py`), with **a different prompt for every reply** so each stays read-bound (a repeated prompt
would become resident): 18 short questions of the same form, two warm-ups, then A B B A x 4 (8 replies an arm),
`temperature 0`, `thinking: false`, 32 tokens. Per reply: decode ms a token, misses a token and the mean drive-read time
from the server's `[request]` line. Workload W7 on W1, read-bound.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| E2-1 | arm A decode | median 0.8 to 1.4 s a token (0.55) | below 0.5 or above 2.0 s |
| E2-2 | B against A | per-cycle B/A of mean s a token, median between 0.97 and 1.03 (0.55) | above 1.08 or below 0.92 |
| E2-3 | misses a token | arm means within 10 % of each other (0.65) | more than 25 % apart |
| E2-4 | drive-read time | mean ms a read within 5 % between arms (0.55) | more than 15 % apart |
| E2-5 | auxiliary beside GLM | classification median +0 to +10 % over its alone baselines (0.55) | more than +25 % |
| E2-6 | completion | no guardian kill, swap growth under 512 MB (0.70) | a kill |

## E-3: power of the system model, alone and beside the GPU

**Baseline (dated, measured).** Idle CPU+GPU+ANE 1.4-3.7 W (0.62.28, powermetrics); idle system (SMC `PSTR`) 25-30 W
(0.62.35); DeepSeek's decode 6.4 J a token all-resident by `PSTR` (0.62.35); the GPU GEMV stand-in at 659 GB/s and the
system model at 52.9 tok/s (0.69.0); no power number exists for the system model.

**Hypothesis (H-e3).** The Neural Engine runs the system model at a few watts; its energy a token is two orders of
magnitude below DeepSeek's; GPU and ANE power add.

**Method.** `benchmarks/apple_fm_bench.py power` (new arm): 30 s cells in the order idle, fm, gemv, gemv+fm, gemv+fm, gemv,
fm, idle, gemm, gemm+fm, gemm+fm, gemm, classify, classify (fm: continuous generation; classify: back-to-back structured
classification; gemv/gemm: the 0.69.0 MLX stand-ins), each in its own process, cell windows logged by wall clock.
`benchmarks/energy_logger.py` beside the whole arm (SMC `PSTR` every second without root; IOReport components only while a
root `powermetrics` runs), and Hamed's `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000` writing a file
for CPU, GPU and ANE watts. Medians over each cell's samples, the first 3 s of a cell dropped. No runtime process.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| E3-1 | ANE power | below 0.1 W in idle and GPU-only cells; 1.5 to 6 W median in fm cells (0.50) | fm cells below 0.5 W or above 10 W |
| E3-2 | system power, fm minus idle (`PSTR`) | +3 to +10 W (0.50) | below +1 or above +20 W |
| E3-3 | energy a system-model token (E3-2 / tok/s, derived) | 0.06 to 0.20 J (0.50) | above 0.5 J |
| E3-4 | energy a classification call (classify minus idle, x latency, derived) | 2 to 8 J (0.45) | above 15 J |
| E3-5 | additivity | gemv+fm `PSTR` within 3 W of gemv + fm - idle (0.50) | more than 6 W off |
| E3-6 | GPU power beside fm | powermetrics GPU W in gemv+fm within 5 % of gemv (0.50) | more than 12 % off |
| E3-7 | GEMV rate beside fm | slowed 1 to 4 % (replicates 2.1 %) (0.60) | sped up, or slowed more than 8 % |

**What each outcome means.** E3-1 to E3-3 held: the system model is a low-power side engine (a few watts, ~0.1 J a token
against DeepSeek's ~6 J), and auxiliary work is cheap in energy as well as time. E3-5 or E3-6 falsified: the package
budget is shared (the GPU clocks down beside the ANE), which would also explain part of the stand-in's slowdown.

## Cost and stop rules

E-2: settle 1-3 min, aux baselines 1 min, server start 1-3 min, 18 replies of ~45-90 s (cold prefill of a short prompt
from the X10Pro plus 32 read-bound tokens) ≈ 15-27 min: **~25-35 minutes**. E-3: 14 cells of ~45 s with process start
≈ **~11-13 minutes**, plus Hamed starting and stopping `powermetrics`. Stop on a guardian kill or a server error. The two
runs are sequential, E-3 first (it needs no runtime and Hamed's terminal), then E-2.
