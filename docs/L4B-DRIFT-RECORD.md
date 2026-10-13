# How much does one configuration drift between processes? (record opened 2026-10-13, before the runs)

Hamed's go: "go, measure the drift by repeating one config" (2026-10-13), after 18.147 showed two budget-0 recordings differing by 5.9 % in decode and 9.3 ms in the router-eval window, against the 1-2 % bound this series had used (one replicate pair, +1.3 %).

## Baseline
Configuration = arm 1 of `docs/L4B-RAW-LAYER-RECORD.md`: exact decode, prefetch on, 34 GiB, 24 mixed turns of 128 tokens, tracer on every token, `token_trace_raw.py` under `guarded_run.sh`. Arm 1 (2026-10-11 evening): decode 151.6 ms a token, 43.0 misses, window 63.1, wait-free floor (sum of per-layer 10th percentiles) 48.9, sum of per-layer medians 64.3, blocked 59.2, busy 101.8, hidden share 43.3 %. The void replicate (`l4b-raw-replicate`, also 2026-10-11): 153.6 ms, window 62.8, blocked 63.4, busy 105.6. Budget 0 twice: 112.2 and 105.6 ms (5.9 %), windows 66.1 and 56.8 (the text differs between those two, because budget-0 drops depend on timing). The exact configuration has deterministic text and a deterministic miss count, so it isolates machine and process state.

## Method
`benchmarks/drift_repeat.sh 3 34`: three new recordings of the same configuration, one after another with `settle.sh` between them (about 15-20 minutes each: load 3, four warm turns, 24 turns), writing `benchmarks/results/drift-34gib/run{1,2,3}`. A state file per run (time, load average, thermal notes from `pmset -g therm`, swap, screensaver, VMware resident size, the eight busiest processes by CPU before and after) and the guard's per-second memory samples. With the two earlier recordings of the identical configuration (arm 1 and the replicate) that makes five samples, over about two days. Quantities per run: decode ms a token, window mean, sum of medians, wait-free floor, blocked, read busy, misses a token, hidden share, tokens, text identity.

## Prediction (written before the runs)
1. Text and token counts identical across all three new runs and arm 1 (greedy, exact); misses a token equal to within 0.3 (43.0).
2. Decode ms a token across the five samples: range 5-12 ms (3-8 % of 151), standard deviation 2-5 ms; the new three within 145-160 ms.
3. Window mean: range 3-9 ms; the wait-free floor (sum of 10th percentiles) range under 2 ms (the floor is stable, as in the two budget-0 runs: 47.8 and 49.2).
4. The movement sits in the upper part of the distribution: the range of the sum of medians is at least half the range of the window mean; the blocked time moves less than the window.
5. Some of the state files will show a difference that lines up with the slow runs (a busier machine: higher load average, an extra heavy process); the match is a hypothesis, not tested by this design.

## Falsifier
Decode range under 3 ms (under 2 %): the exact configuration is stable and the budget-0 variation (5.9 %) is specific to budget 0, not machine drift. Decode range over 15 ms (10 %): machine state matters more than the 6 % the series now assumes. A floor range above 3 ms: the floor is not stable and the "wait-free" reading of 18.142 is weakened.

## Limits
Three new samples plus two earlier ones, over two days (the earlier two ran at other times of day and in other load states); the sample is too small for a standard deviation with a tight interval (three to five values: a range and an sd to be read as a rough bound); background load is whatever the machine had (WindowServer 44 %, iStatistica, Codex, Claude, VMware Fusion running idle at 115 MB resident, a Python process at 24 % CPU were seen in a snapshot before launch), not controlled; the state capture records it but cannot say which part matters.

## Result (measured, 2026-10-13)

Three new recordings (`benchmarks/results/drift-34gib/run{1,2,3}`, local; about 8.5 minutes each, faster than the estimate of 15-20) of arm 1's configuration, with settle steps between, and the two earlier recordings of it. 24 mixed turns, 2,452 tokens, 43.0 misses a token and 0 dropped events in all five; token counts per turn identical in all five.

| sample | decode ms a token | window | wait-free floor | sum of medians | blocked | read busy | hidden |
|---|---|---|---|---|---|---|---|
| arm 1 (10-11 evening) | 151.6 | 63.1 | 48.9 | 64.3 | 59.2 | 101.8 | 43.3 % |
| replicate (10-11) | 153.6 | 62.8 | 48.7 | 65.0 | 63.4 | 105.6 | 41.3 |
| run 1 (10-13 19:16) | **156.8** | 62.8 | 48.8 | 64.5 | 65.6 | 108.3 | 40.8 |
| run 2 (10-13 19:25) | 151.5 | 62.4 | 48.6 | 63.7 | 61.2 | 103.1 | 42.1 |
| run 3 (10-13 19:34) | 151.3 | 62.4 | 48.5 | 63.2 | 61.4 | 103.0 | 41.8 |
| range | **5.5 ms (3.6 %)** | 0.7 | 0.4 | 1.8 | 6.5 | 6.5 | |
| sd | 2.3 ms (1.5 %) | 0.3 | | | | | |

State files (`drift-34gib/run*.state.txt`): no screensaver in any run; the machine was busy throughout (load average 2.6-6.8; WindowServer 28-49 % CPU, Codex service and renderer 6-14 %); a VM was running in VMware Fusion (`vmware-vmx`, 10 GB virtual, 2.5-3.4 GB resident); the load average rose from 3.1 to 6.8 during run 1 (the slowest) and stayed at 4.6-6.1 around runs 2 and 3.

## Against the prediction

1. **Held.** Token counts identical, misses 43.0 in all five (text identity was not compared, the manifests hold counts).
2. **Held, at the low end.** Decode range 5.5 ms (predicted 5-12), sd 2.3 (2-5); the new three at 156.8, 151.5, 151.3 (145-160).
3. **Floor held, window missed.** The wait-free floor range 0.4 ms (under 2); the window range 0.7 ms against the predicted 3-9: the window is far more stable here than in the two budget-0 recordings (9.3 ms).
4. **Missed.** The movement is in the blocked time (range 6.5 ms) and read busy time (6.5), not in the window (0.7) or in the sum of medians (1.8): the opposite of "the blocked time moves less than the window".
5. **Weak.** The slowest run (run 1, 156.8 ms) is the one that began with the VM resident size at its largest (3.4 GB, falling to 2.6) and a load average that climbed from 3.1 to 6.8 during the run; runs 2 and 3 also had load averages of 4.6-6.1 and were fast. A hypothesis, not a finding; the state capture cannot isolate a cause from five samples.

No falsifier fired: the range (5.5 ms, 3.6 %) is neither under 3 ms nor over 15 ms, and the floor is stable.

## Explanation (hypothesis)

For an exact configuration with a deterministic text and miss count, decode time varies by about 1.5 % (sd, five runs over two days; range 3.6 %), and the variation is in how long the main thread waits on the drive (blocked and read busy time, range 6.5 ms), not in the GPU-side window (range 0.7 ms) or the floor (0.4 ms). That points at the storage side of the machine (the internal drive's response under background activity, a VM and the system's other disk traffic), not at the GPU clock. The budget-0 pair of 18.147 differed by 5.9 % in decode and 9.3 ms in the window; that is far outside this range for the window, and budget 0's text and drop pattern differ run to run (4 of 24 turns of different length), so **the budget-0 variation is largely budget 0's own nondeterminism (what is dropped depends on what is resident or in flight at that moment), not machine drift**. 18.147 over-corrected: it attributed all of the budget-0 difference to the machine.

## Consequences (corrections to 18.147)

- The between-process drift bound for an **exact** configuration is about 1.5 % sd, range 3.6 % (5.5 ms) in decode, and 0.7 ms in the router-eval window; the earlier "1-2 %" was nearer the truth for exact arms; "6 %" in 18.147 is withdrawn as a general bound. For **budget-0 arms** run-to-run differences of 6 % in decode and 9 ms in the window have been seen and are not drift of the machine alone.
- Cross-process decode gaps in the earlier sections, judged against a range of 3.6 %: the prefetch-off penalty at 34 GiB (+10.4 %), the 46 GiB pair (+7.7 %), the 28-34 and 22-28 GiB steps (+9.4 %, +7.4 %) exceed it and stand; arm 7's prefetch gap (+3.3 %) is within it (unresolved by that pair; the alternated pair of 18.145 resolves it at +5.5 ms, +4.1 %); the 5.9 % budget-0 pair is within the range of exact drift only if one counts range, not sd, so it is borderline for the drift explanation and is better explained by budget 0's variability.
- The "excess" of 18.142 and 18.146 stays withdrawn (the second budget-0 recording did not reproduce it); the reason is budget 0's own variability, not machine drift.

## Decision

No change in `src/`. Ledger row DS-DRIFT. A tighter bound needs more runs (three more would cost about 25 minutes); not worth it unless a decision depends on a gap under 4 %.
