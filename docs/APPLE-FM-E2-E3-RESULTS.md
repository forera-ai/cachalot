# E-2 and E-3 results, and lever D priced: GLM's read-bound decode does not notice the system model; the system model costs ~18 W alone and ~8 W beside a busy GPU; no cache policy has anything to win

Written 2026-10-30 against runtime 0.69.3 plus the drivers of 97b40f9. Record: `docs/APPLE-FM-E2-E3-RECORD.md` (E2-1 to
E2-6, E3-1 to E3-7, committed first in a8749fe). Raw files (local): `benchmarks/results/apple-fm-e1/20261030T105018Z/`
(E-2), `benchmarks/results/apple-fm/power-20261030T111349Z.jsonl` and `power-logger-20261030T111349Z.jsonl` (E-3).
Numbers measured unless tagged derived.

## 0. Deviations from the record

* **Order.** Hamed confirmed "powermetrics is recording", but no `powermetrics` process existed and its output file was
  never created (checked by process list and path; the Terminal panel showed only a prompt). E-2, which does not need it,
  ran first; E-3 ran after with SMC `PSTR` only (no root needed) and the powermetrics join left in place, so it would have
  picked up any samples. None arrived: **E3-1 (ANE watts) and E3-6 (GPU watts) are not scored**, and the IOReport
  component counters stayed frozen (they advance only beside a root `powermetrics`), so they are not used.
* A 5 s smoke run of four E-3 cells (idle, fm, gemv+fm, classify) ran before the measured run, after the record was
  committed; it showed fm about +22 W over idle. It is not counted.
* E-2's guarded server ran 1,269 s (21 minutes) against the 25-35 estimated; E-3 about 12 minutes as estimated.

## 1. E-2: GLM-5.3-Flash read-bound, beside the system model

`./serve-glm.sh --expert-budget-gib 46` under `guarded_run.sh --force` (bank on the X10Pro, decode miss budget 2,
prefetch off, no snapshot directory), ready in 14 s; 18 distinct prompts (two warm-ups, then A B B A x 4), 32 tokens,
`temperature 0`. Peak wired 74.1 GiB, swap flat (3,512 MB), pressure 1.

| id | result | score |
|---|---|---|
| E2-1 | arm A median 0.909 s a token (0.860-0.995) | held |
| E2-2 | B/A by cycle 0.912, 0.971, 0.972, 1.051; **median 0.971** (B slower in 1 of 4) | held (at the interval's edge) |
| E2-3 | misses a token 57.9 (A) and 56.3 (B), 2.8 % apart | held |
| E2-4 | drive-read time 24.45 and 24.31 ms a read, 0.6 % apart | held |
| E2-5 | classification 0.831 s beside GLM (621 calls) against 0.825 s alone (73): +0.7 % | held |
| E2-6 | no kill, swap flat | held |

**Six of six held.** Each reply's cold prefill of a short new prompt took 36-37 s (the X10Pro reads every layer's
experts for it); decode 0.83-0.99 s a token at 52-65 misses a token, 80-84 % hit. **A read-bound GLM token does not notice
the system model** (B/A 0.97 median, the cycles scattering ±5 % around it on different prompts), and the system model does
not notice GLM (+0.7 %). The drive reads are untouched (24.3-24.5 ms each). The 16 auxiliary errors were the known refusal
of fixture 29, once in each pass of the 40 fixtures.

## 2. E-3: the system model's power (SMC `PSTR`, whole system, 1 Hz)

Fourteen 30 s cells, each the median of ~27 one-second samples after a 3 s trim; each cell kind ran twice.

| cell | system W (two cells) | mean | rate |
|---|---|---:|---|
| idle | 19.1, 18.7 | 18.9 | |
| fm (system model generating) | 36.4, 37.3 | 36.9 | 52.5 tok/s |
| classify (back-to-back calls) | 37.2, 37.4 | 37.3 | 0.817 s a call |
| gemv | 128.6, 132.2 | 130.4 | 653.9 GB/s |
| gemv + fm | 137.8, 138.7 | 138.3 | 643.2 GB/s, 51.1 tok/s |
| gemm | 179.9, 176.5 | 178.2 | 17.77 TFLOP/s |
| gemm + fm | 192.2, 194.2 | 193.2 | 18.12 TFLOP/s, 52.3 tok/s |

| id | result | score |
|---|---|---|
| E3-1 | ANE watts | not scored (no powermetrics) |
| E3-2 | fm minus idle **+18.0 W** | outside (+3 to +10), under the +20 falsifier |
| E3-3 | energy a system-model token, alone: 18.0 W / 52.5 tok/s = **0.34 J** (derived) | outside (0.06-0.20), under the 0.5 falsifier |
| E3-4 | energy a classification call: 18.4 W x 0.817 s = **15.1 J** (derived) | **falsified** (above 15 J, by 0.1) |
| E3-5 | gemv+fm against gemv + fm - idle: **-10.1 W**; gemm: -3.0 W | **falsified** (more than 6 W off, sub-additive) |
| E3-6 | GPU watts beside fm | not scored (no powermetrics) |
| E3-7 | GEMV 653.9 to 643.2 GB/s beside the system model: -1.6 % | held |

**What it shows.**

1. **Alone, the system model raises whole-system power by about 18 W** (alone at idle: 0.34 J a token, 15 J a
   classification call). That is far more than the Neural Engine itself is likely to draw (hypothesis: most of it is
   waking the shared fabric, memory and CPU cluster out of their idle states, which `PSTR` sees and an ANE counter
   would not; the missing powermetrics cell would have separated them).
2. **Beside a busy GPU it costs much less: +7.8 W beside the GEMV, +15.0 W beside the GEMM** (derived as both minus
   GPU-alone), so the increments do not add (sub-additive by 10.1 W and 3.0 W). Read as the hypothesis above: the fixed
   cost of an awake machine is paid once, by whichever engine wakes it first. Beside the GEMV the system model's
   marginal energy is about 7.8 W / 51.1 tok/s = **0.15 J a token** (derived).
3. **For scale (derived, different models and work):** DeepSeek's all-resident decode measured about 6.4-6.8 J a token by
   `PSTR` (0.62.35-0.62.36); a classification call beside a running target costs on the order of the energy of one or two
   target tokens.
4. The GEMV slowdown replicates 0.69.0 (-1.6 % here, -2.1 % then).

## 3. Lever D priced on a real Hermes session (offline; a pricing step, not a registered prediction)

`benchmarks/prefix_grain_replay.py` (DI-4's replay through the real `prepare_prompt` and `PrefixCache`, no model loaded)
on the 142-request Hermes session `benchmarks/results/hermes-live-di4/requests.jsonl`, at several cache budgets
(`--gib`, new `--entries`):

| prefix-cache budget | tokens prefilled over the session |
|---|---:|
| 0.25 GiB | 350,541 |
| 0.5 GiB | 314,111 |
| 1.5 GiB, 8 entries | 313,920 |
| **1.5 GiB, 64 entries (shipped)** | **279,056** |
| 2 GiB | 279,247 |
| 3 GiB | 266,959 |
| 6 GiB, or unbounded | 266,913 |

An unbounded cache saves 12,143 tokens over the shipped one (4.4 % of the session's prefill, about 2 minutes at 99 tok/s,
derived), and **one request carries almost all of it** (request 106: a 12,288-token system block evicted and prefilled
again; two others differ by 46 and 191 tokens). Everything above 266,913 is new content no cache can hold. So on this
session **no eviction policy, semantic or not, can save more than one system block**, which a 3 GiB budget (or a pin for
that block) already saves. An auxiliary model's labels have nothing to add: lever D is closed on the measured session
(reopen only if a session shows eviction losses a budget cannot fix). The 2 GiB row is slightly worse than 1.5 GiB (an
LRU ordering effect, not chased).

## 4. Limits

E-2: one GLM server, 8 replies an arm on 16 different prompts, 32 tokens each. E-3: whole-system power only, one run,
Hamed's applications open (5-minute load ~5), no component breakdown. Lever D: one Hermes session.

## 5. Next cells (not registered)

E-3's two unscored cells with a root `powermetrics` actually recording (the same command; ~12 minutes); a prefill arm
(a cold 4-8k DeepSeek prompt with the auxiliary load beside it).
