# E-1 results: beside the system model, DeepSeek's server decode moves by less than about 1 ms a token, sign unresolved (+0.57 ms under budget 0, -0.97 ms on the exact path)

Written 2026-10-29 against runtime 0.69.1. Record: `docs/APPLE-FM-E1-RECORD.md` (E1-1 to E1-7, committed first in
0ff4d76); driver built in bf29cc7 before the run. Raw files (local): `benchmarks/results/apple-fm-e1/20261029T194147Z/`
(replies, auxiliary call logs, summary) and the guardian's `apple-fm-e1_*` output and samples. Numbers measured unless
tagged derived.

## 1. What ran

`benchmarks/apple_fm_e1.sh`, one attempt that ran to completion: `settle.sh`, 30 s of auxiliary calls alone, one fresh
`./serve.sh` (DeepSeek V4.1 Flash, internal 2-bit bank, 48 GiB expert budget, wired limit 80, shipped decode miss budget
0, `CACHALOT_SNAPSHOT_DIR=`) under `guarded_run.sh --force` (ready in 17.5 s), two warm-up replies, 24 measured replies in
A B B A order six times, the server stopped, 30 s of auxiliary calls alone again. Each reply: the same 36-token prompt,
`temperature 0`, `thinking: false`, 128 tokens (every reply hit the cap). Machine: sysctl 88064, screensaver off,
memory pressure 1 throughout, swap 4,486 MB at start and peak (stale, flat), peak wired 73.4 GiB, min free 0.1 GiB,
Hamed's applications open (5-minute load 3.2-3.5). Run time of the guarded server 304 s.

**Deviation from the record, before the measured run:** the first start was stopped during `settle.sh`, which waited
for the 77 GiB that a 48 GiB arm's formula asks for while the machine offered ~70 and would have timed out; the gate was
changed to `settle.sh --budget-gib 36` (its pressure, swap and quiet checks unchanged) and the arm restarted. The arm
itself ran at 48 GiB under `--force`, as every 48 GiB arm since 18.76 has.

## 2. Scores

| id | result | score |
|---|---|---|
| E1-1 | **24 distinct texts** among the 24 measured replies (and the two warm-ups), in both arms | **falsified** |
| E1-2 | arm A median 78.12 ms a token (77.6-79.8) | held |
| E1-3 | B - A, median of the six cycles: **+0.57 ms a token** (+0.27, +0.50, +0.52, +0.62, +1.22, +1.83) | held |
| E1-4 | B slower in 6 of 6 cycles | held |
| E1-5 | misses a token: A 1.43, B 1.24 (difference 0.19) | held (by 0.01) |
| E1-6 | classification call 0.820 s alone (74 calls, before and after), 0.862 s beside the decode (121 calls inside B replies): +5.2 % | held |
| E1-7 | no guardian kill, swap growth 0 MB, pressure 1 | held |

**Six held, E1-1 falsified.**

| arm | replies | ms a token, median | range | reply wall, median |
|---|---:|---:|---|---:|
| A, auxiliary idle | 12 | 78.12 | 77.6-79.8 | 10.00 s |
| B, auxiliary classifying back to back | 12 | 79.26 | 77.7-80.7 | 10.15 s |

(The two arms' medians differ by 1.1 ms; the registered statistic, the median of the per-cycle differences, is 0.57 ms.)

## 3. What it shows

1. **The system model beside DeepSeek's decode costs about 0.6 ms of an ~78 ms token (+0.7 %), consistently** (every
   cycle, +0.3 to +1.8 ms). The auxiliary side paid +5 % (0.82 to 0.86 s a call) and ran about one call a second, so
   the target was sharing the machine with a nearly continuous auxiliary load.
2. **The stand-in over-predicted it.** The GEMV stand-in lost 2.1 %; DeepSeek's token lost 0.7 %, about 1.1 % of its
   ~53 ms of GPU work if all of it lands there (derived). Part of a real token is not bandwidth-bound (17.9 ms of host
   work between syncs, kernel launches below the bandwidth ceiling), and classification calls are bursts with output of
   a few dozen tokens, not continuous generation (hypothesis for the difference).
3. **For a scheduler:** at this cost there is nothing to schedule around on the short all-resident decode; auxiliary
   classification can run during decode. Not measured: a read-bound decode, long contexts, prefill, GLM or MiniMax.

**Why E1-1 failed (hypothesis, not tested).** The record assumed the repeated greedy prompt would be all-resident after
two warm-ups. It was not: 0.5-2.8 misses a token remained (hit 99-100 %), and under the shipped miss budget 0 a miss is a
dropped expert, so the routing and then the text change from reply to reply as the cache moves. The A replies differ
among themselves as much as from the B replies, so the auxiliary load is not what changed the text; the target never
reads anything the auxiliary touches. An exact arm (`CACHALOT_DECODE_MISS_BUDGET=off`) would be the identity check.

## 4. Limits

One server process, one prompt, 128 tokens, 12 replies an arm, Hamed's applications open. One auxiliary request type.
The texts differ between replies, so each reply's routing differs a little (misses 0.5-2.8 a token in both arms; the arm
means agree within 0.2).

## 5. Next cells (not registered)

The exact-path replicate (budget off) to confirm text identity under load; a GLM read-bound arm (E-2); the power cells
with Hamed's `powermetrics` (E-3); a prefill arm (a cold 4-8k prompt with the auxiliary load beside it).

## 6. The exact-path replicate (addendum 1 of the record, E1R-1 to E1R-6, committed first in 47d6358)

Same script, `CACHALOT_DECODE_MISS_BUDGET=off` in the server's environment (the script printed `server decode miss
budget: off`), one attempt, run directory `benchmarks/results/apple-fm-e1/20261029T195225Z/`. Settle ready after 20 s
(72.4 GiB available), server healthy after 20 s, guardian 297 s, peak wired 73.5 GiB, pressure 1, swap 4,486 MB at
start and 3,259 MB peak afterwards (macOS shrank it during the run).

| id | result | score |
|---|---|---|
| E1R-1 | all 26 replies (warm-ups included) byte-identical, one hash | held |
| E1R-2 | misses a token 0.0 in every measured reply, both arms (the first, cold reply 23.0) | held |
| E1R-3 | arm A median 75.95 ms a token (74.9-77.3) | outside (78-84), above the 75 ms falsifier |
| E1R-4 | B - A by cycle: +0.98, -1.25, -1.07, -0.87, -0.13, -1.35; **median -0.97 ms** | outside (+0.2 to +2.0), short of the -1 ms falsifier |
| E1R-5 | B slower in **1 of 6** cycles | **falsified** |
| E1R-6 | no kill, pressure 1, swap did not grow | held |

| arm (exact path) | ms a token, median | range |
|---|---:|---|
| A, auxiliary idle | 75.95 | 74.9-77.3 |
| B, auxiliary classifying back to back | 75.33 | 73.9-77.5 |

Classification calls: 0.846 s beside the decode (119 calls) against 0.825 s alone (72): +2.6 %. Mean drive-read time of
the predicted loads 0.84 ms (A) against 0.89 ms (B); in the budget-0 run 1.61 against 1.68 ms.

**What the two runs say together (measured, then derived).** Under budget 0 the auxiliary arm was slower in 6 of 6
cycles (+0.57 ms); on the exact path it was faster in 5 of 6 (-0.97 ms). Each run's cycles agreed with each other and the
runs disagree in sign, so the effect is smaller than the run-to-run difference of the method: **the system model's
classification beside DeepSeek's all-resident decode moves a token by less than about 1 ms (about 1.3 %), with no
resolved sign.** Section 3's "+0.57 ms a token, consistently" holds for that run only and is withdrawn as the general
statement. The practical conclusion does not change: auxiliary classification can run during this decode.

**Hypotheses for the negative gap (not tested):** a continuously busy auxiliary client and system service keep the CPU
cluster out of deep idle, which shortens the host's wake-ups between the token's 44 synchronizations (the "slow decode
window" effects of HANDOFF 15.9 and the 0.13.0 focused-app role point the same way); under budget 0 the token also waits
on in-flight predicted loads, whose reads ran ~4 % slower in B in both runs.

**The exact floor moved (derived).** 75.95 ms against 18.76's 79.2-80.3 ms on 0.60.10: the FP8 GEMV decode, on by default
since 0.66.0 and priced at ~3.3 ms a token in a microbenchmark, is the likeliest reason; E1R-3's interval used the stale
constant. Not separately tested here.
