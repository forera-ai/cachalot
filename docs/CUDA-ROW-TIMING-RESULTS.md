# CUDA lane: per-row and per-copy timing of a miss (results; RTX 5090, one instance)

Run 2026-10-20 under `docs/CUDA-ROW-TIMING-RECORD.md` (seven predictions committed in 6502a50 before any code; erratum 21b65ac before the build; section 8 erratum for a harness defect found in these runs). One RTX 5090 instance of the fast population (median source-record copy 0.194 ms), the new multi-lead trace, 17 % resident, 400 decode tokens a cell. Two full runs (A, B; 24 cells each, 264 s each) and one fix run (`f-fix`, 8 cells, 161 s), all nine gates true in every run including the new `rowsum` and `copyevents` (and `floor` in the fix run). Scoring: `benchmarks/cuda/row_timing.py RUN_A RUN_B --fix FIX`, output `docs/cuda-results/row-timing-2026-10-20.json`. Labels follow plan section 0: **measured** on this card on this day; the explanations are **hypothesis**.

## 1. A defect of mine, and what it voided

The plan revisited C = 1.0 ms after 0.3 ms and the stand-in was not re-tuned: the level-6 and level-4 cells at C = 1.0 in runs A and B ran at 0.3 ms (floors 12.6 ms against 40 ms). I caught it from the floors, which no gate checked. The fix run repeats exactly those cells with the fixed harness (floors 40.03 and 40.00 ms). Cells at C = 1.0 for levels 6 and 4 therefore have one run, not two; everything else has two (replicate agreement below).

## 2. Scoring the registered predictions

| id | measured | verdict |
|---|---|---|
| RT-1 (idle-queue demand copy 0.17 to 0.22 ms) | median 0.2036 ms, p05 0.190, p95 0.207 (5,147 batches, `D`, level 6, C = 1.0) | **held** |
| RT-2 (first copy of a batch minus the later ones, -0.03 to +0.06 ms) | +0.014 ms | **held** |
| RT-3 (share of `P0L` miss rows that wait behind prefetch: at most 0.20 at 1.0 ms; 0.40 to 0.75 at 0.3 ms) | 0.005 at 1.0 ms; 0.30 at 0.3 ms | **missed** (the 1.0 half held; the 0.3 half is below its interval, above the 0.25 falsifier) |
| RT-4 (main claim: stall per left-behind miss 10 to 25 % above `D`'s at C = 1.0, at least 70 % of it queue wait) | **+1.3 % (level 3) and +1.5 % (level 6)** at C = 1.0 (0.188 against 0.1855 ms); queue wait in 0.5 % of rows | **falsified** (within 5 % of `D`) |
| RT-5 (a prefetch copy 0 to +8 % slower under compute) | +1.0 % at 0.3 ms (10,969 under, 9,018 after); +0.3 % at 1.0 ms (only 48 copies complete after the row's end) | **held** |
| RT-6 (per-copy events cost at most 0.5 ms a token; closure at least 0.90) | level 6 minus level 3: -0.06 / -0.05 ms (1.0 ms, `D` / `P0L`), +0.025 / +0.062 ms (0.3 ms); closure 1.00 | **held** |
| RT-7 (queue-aware row model: r at least 0.8, token-level hidden-share error at most 0.05; falsifier r below 0.6 or error above 0.08) | r = 0.945 (1.0 ms), 0.991 (0.3 ms); hidden share 0.533 against the level-4 measured 0.423 (error 0.110) and 0.321 against 0.232 (error 0.089) | **falsified** (token error past 0.08 at both C; the correlation held) |

**4 held, 1 missed, 2 falsified.** The adoption rule (RT-4 and RT-7 together, RT-6 not fired) is not met: **no queue-aware row model is adopted and the copy-count screen stays.**

## 3. What was measured

* **Queue wait is the cost, and only when the window is short (measured).** At C = 0.3 ms a left-behind miss costs 0.246 ms against `D`'s 0.1855 (+33 %, level 3; +33 %, level 6), and 90 % of the excess is time the demand batch waited behind earlier prefetch copies (`q_ms`); 30 % of miss rows wait. At C = 1.0 ms there is no excess (+1.3 %): the previous row's compute window finishes the prefetch first. The 0.22 against 0.187 ms per miss of the earlier fits was an average over cells; it is the 0.3 ms cells' queue wait. This is not what RT-4 registered (it asked for the effect at 1.0 ms), and it is reported as a measured result and as hypothesis, not as a held prediction.
* **A copy is a copy (measured).** An idle-queue demand copy costs 0.2036 ms with no batch overhead (+0.014 ms for the first of a batch), and a prefetch copy under a compute kernel costs 1 % more than one with compute idle. The per-miss difference is not a slower copy.
* **The row model's stalls are right and its token total is not (measured, derived).** The queue-aware row model predicts each row's wait with r = 0.99 (0.3 ms) and 0.945 (1.0 ms) on the replicate. Summed per token, the row stalls give hidden shares of 0.501 (1.0 ms) and 0.323 (0.3 ms) against 0.406 and 0.211 from walls (level 3). The remainder is compute-stream time: the stand-in's busy time is **0.607 ms a token longer in `P0L` than in `D` at C = 1.0 (+1.5 %) and 0.77 ms longer at 0.3 (+6.3 %), identical in runs A and B to 0.006 ms**. That is about the size of the model's token error (0.96 ms at 1.0, 0.78 ms at 0.3).
* **Per-copy events are cheap here (measured).** No cost is visible at level 6 against level 3 at 1.0 ms (negative), and 0.03 to 0.06 ms a token at 0.3 ms.
* **Reproducibility (measured).** The compute busy times of the level 3 cells agree between runs A and B to 0.006 ms; the stall per miss was scored on run B only (RT-4) and not compared across runs. The single-run cells are level 6 and level 4 at C = 1.0 (the fix run).

## 4. What it means, and does not

**Hypothesis, not tested:** a prefetch copy stream running beside the stand-in compute stretches the compute kernels by 1.5 % of the token at a 1 ms layer and 6 % at 0.3 ms (cross-stream memory contention), which is the term a row model needs besides the queue wait; adding it (a per-copy compute slowdown, fitted on one run and scored on the other) is the next offline step and needs no GPU. The earlier N3 finding that copies slow compute by 0.3 to 1.1 % at 1 ms and above (stage A, an interval) is consistent with the 1.5 % here.

**Not shown:** anything about another card or instance (the copy rate varies by instance; this one is the fast class), the real model, or a mechanism of the slowdown. The stand-in compute is a bandwidth-bound reduction; the model's kernels may contend differently.

## 5. Decision

* No model adopted; the screen stays at an error of about 0.1. The offline follow-up (a compute-slowdown term fitted on the recorded busy times) is the one item the data names; it needs no GPU and is Hamed's call.
* The other open lane items (fewer and larger copies, a native issue path, ROCm, V100 or L40S, instance population) are unchanged and need his go and a GPU. Fewer and larger copies is weakened (no per-copy overhead was found: RT-2).
