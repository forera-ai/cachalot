# E-1 results: DeepSeek's server decode beside the system model costs +0.57 ms a token (+0.7 %)

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
