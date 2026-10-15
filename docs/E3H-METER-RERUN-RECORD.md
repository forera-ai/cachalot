# E3h record: does the 0.63.1 component-tail fix hold live, and what does the energy meter cost a decode token? (charter L5b, follows `docs/E3G-GLM-MINIMAX-RECORD.md`)

Fields 1-4 written 2026-10-11, before the run, against runtime 0.63.1. Result and explanation are appended after the run.

## 1. Baseline

E3g (HANDOFF 18.130): on MiniMax's mostly 5 s decode windows the meter's DRAM and GPU component joules read a median 7.3 % low against the logger (signed median about -8 %); cause found in `EnergyMeter._cum_at` (components held, system extended past the last sample); fixed in 0.63.1 and replayed offline over 72 figures from the logger's own rows (median error -7.9 % before, 0.0 % after, maximum absolute 13.6 %). Not yet seen live with the meter and the logger sampling independently. The meter's cost on a decode token has not been measured: by construction it adds two `time.time()` readings and a ring lookup after the last token and a 1 Hz thread that makes two ctypes calls (an SMC read and an IOReport snapshot) per second; the DeepSeek all-resident token is 79-83 ms through the server, with a run-to-run drift of several milliseconds on longer scales (HANDOFF 18.82). The meter's sampler thread contends for the GIL with the decode loop in the same process, once a second.

## 2. Hypotheses and predictions (tags: prediction)

Intervention in one line: four fresh DeepSeek server arms in the order **off, on, on, off** (ABBA; `CACHALOT_ENERGY` unset or 1; exact decode, `serve.sh`, 48 GiB, internal bank, `CACHALOT_PREDICT_TOPK` default), each with the repeated all-resident prompt at 160 tokens for 130 s (about 9 requests of about 13 s of decode), Hamed's root `powermetrics` and the logger beside throughout, so the "on" arms give live meter-against-logger pairs.

- **Cost.** Pooled over the two arms of each kind, the meter-on decode time a token minus the meter-off is **+0.1 ms (-1.5 to +1.5)**, that is within +-1.9 % of 80 ms, the resolution of this design; the meter's work is two clock readings per request plus a once-a-second thread. Probability that the true cost is under 0.5 ms: 85 %. The design cannot resolve less than about +-1 ms against drift; a smaller cost is reported as "below the resolution", not as zero.
- **Drift.** The A/A differences (first off against last off, first on against second on) are each **under 3 ms** in absolute value; if either is larger the arms drifted too much for the cost to be read and it is reported as void.
- **The tail fix, live.** Over all "on"-arm requests with decode windows of at least 8 s (about 18), component joules (DRAM and GPU) against the logger's IOReport differences: **median absolute difference under 3 %, signed median within +-2 %, maximum absolute under 10 %**; system joules: median under 1.5 %, maximum under 6 %.
- Preconditions: the sampler running during all four arms (frozen counters make the component comparison void, not the cost); idle server errors not required (no idle phases); each arm's phase B at 0.0 misses a token on the server's counters (the arm is all-resident), 1.5 to 2.5 GB/s of prefetch loads from the drive is expected as in E3 and is the same in every arm; fans at the start at or below 1,300 RPM; no other GPU user (Cachalot Lab closed).

## 3. Falsifier

The cost claim is falsified if the pooled meter-on minus meter-off exceeds **+1.9 ms (+2.4 %)** and both ABBA pair differences (on1 minus off1 neighbour, on2 minus off2 neighbour) point the same way; it is void if an A/A difference is 3 ms or more. A pooled difference between 1.5 and 1.9 ms is unresolved. The tail fix is falsified if the components' median absolute difference is **5 % or more**, or the signed median is **-4 % or lower** (the old bias was -7.9 %); between those and the prediction it is unresolved. Void: a frozen DRAM counter during an "on" arm (for the component comparison only), a server killed by the memory guardian, an arm with misses a token above 0.5 in phase B.

## 4. Intervention

None to the runtime. The energy driver now logs every request's server-side decode time (and the meter's report when present); `E_C=-1` skips its read-bound phase. New `benchmarks/meter_ab.sh` (the four arms) and `benchmarks/meter_ab_analyze.py` (cost, drift and the tail check). About 16 minutes (four arms of about 3.7 minutes: server load, warm-up, 130 s of load, stop; 20 s between arms). Cachalot Lab closed, display as it is. Commands for Hamed, two terminals, the sampler first:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 1300 -o benchmarks/results/energy/e3h-sampler.txt
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./benchmarks/meter_ab.sh > benchmarks/results/energy/e3h-run.log 2>&1
```

## 5. Result (measured; 2026-10-11, arms 12:43:42-12:58:47, Cachalot Lab closed; `benchmarks/results/energy/e3h-{1-off,2-on,3-on,4-off}-*`, sampler `e3h-sampler.txt`; analysis `benchmarks/meter_ab_analyze.py`)

**The all-resident precondition failed in every arm.** The repeated prompt at 160 tokens was not all-resident: each arm's requests missed **10.9 experts a token** (hit rate 95 %, 20.3 misses on the first pass) pass after pass, because a 160-token reply touches more experts than the 48 GiB cache keeps between passes and the same experts are evicted and re-read each time (the 120-token arms of E3-E3e had 0.0 misses). I raised the reply from 120 to 160 tokens to get decode windows of at least 8 s without checking that the arm stayed resident; the smoke run that would have shown it was not made. Per the void list in section 3 (an arm with misses a token above 0.5), **all four arms are void as registered**, the cost reading and the tail check alike. What follows is what was observed, with that stated.

**Cost (void, observed only).** Server-side decode time a token, per arm (9, 9, 8, 8 requests of 160 tokens): off1 median 99.58 ms (sd 0.36), on2 99.95 (0.72), on3 100.67 (mean 101.85, sd 3.47, last third 106.4), off4 114.47 (sd 1.15). The drift check failed hard: off1 minus off4 is -14.7 ms (limit 3), on2 minus on3 -2.1 ms; arm 4 ran hot (fans 1,770 to 1,830 RPM against 1,056 to 1,238 in arms 1 and 2; the drive-influenced token drifts with temperature, HANDOFF 18.82). The pooled on-minus-off (-5.9 ms mean, -0.08 ms median) is meaningless under that drift. The one pair taken in the same cool state (arm 1 against arm 2, similar fans) differs by +0.07 ms (means 99.66 and 99.72; sds 0.4 and 0.7 ms), which is consistent with a cost under about 1 ms and is **not** a measurement of it. The meter's cost on a decode token is **unmeasured**; the prediction (+0.1 ms, -1.5 to +1.5) is neither held nor missed.

**Tail fix, live (observed; the miss precondition was not met).** Over the two "on" arms, requests with decode windows of at least 8 s (decode 15.9 to 18.4 s): system joules, n 15, median absolute difference from the logger **0.2 %**, maximum **0.6 %**; component joules (DRAM and GPU), n 30, median absolute **1.1 %**, signed median **-0.2 %**, maximum **8.5 %**. Against the predictions (components: median under 3 %, signed within +-2 %, maximum under 10 %; system: median under 1.5 %, maximum under 6 %): all five numbers fall inside. The failure of the miss precondition does not bear on this comparison (the meter and the logger measure the same power whatever the workload; the precondition was written for the cost claim), but it was registered in the void list, so a strict reading voids this too, and the windows here are 16 to 18 s, longer than the 5 s windows that showed the bug; a one-interval error is 6 % of 16 s, so the longer windows test the fix less sharply than MiniMax's did.

## 6. Explanation

The old bias (-7.9 % on components) has gone: the signed median is -0.2 % on windows where the held tail would have cost about 3 to 6 %. The maximum of 8.5 % is a single figure; the 1 Hz sampling of two processes at independent phases leaves up to one interval of error at each end of a window, as the Lab brief says. The cost question needs a corrected design: the 120-token reply that stays resident (0.0 misses, 79 to 83 ms), arms short enough not to drift (the same machine drifted 15 % over eleven minutes in 18.82), and an ABBA order repeated or interleaved so that drift cancels. The expected effect (two clock readings per request, one thread tick a second) is far below what such a design resolves, about +-1 ms.

## 7. Generalization

The meter's component and system joules match an independent logger on DeepSeek, GLM and MiniMax servers (E3g, E3h) to about 1 % median on windows of 8 s or more. No claim about the meter's cost beyond "no difference visible at about 1 ms in the one cool pair".

## 8. Bottleneck after

Speed unchanged. Energy: the instruments are verified; what is open is only a clean cost price (a corrected ABBA with resident arms) and Lab's use of the figures.

## 9. Kind

Research (predicted, measured): the tail-fix predictions held; the cost question was not answered because the arm's design failed its own precondition (my error, recorded), and the drift check, designed for this, caught it.

## Addendum E3j (2026-10-16, runtime 0.66.2): the corrected cost design

Written before the run, Hamed's go ("go for energy-meter A/B e3h-sampler"). Section 6 of this record named the fix; this is it.

**Baseline.** The meter-on minus meter-off cost is unmeasured (E3h was void: 160-token replies missed 10.9 experts a token, arm 4 ran 15 ms hot).

**Intervention.** `benchmarks/meter_ab2.sh`: six fresh-server arms in the order **off, on, on, off, off, on** (a linear drift cancels), 120-token replies (all-resident, 0.0 misses a token), 60 s of load an arm, then `benchmarks/meter_ab_analyze.py` (tag `e3j`). Estimate: about 3 minutes an arm, **about 20 minutes** in all; the sampler `-n 1500` covers it. Memory is free after the reboot (guard 80 GiB available, sysctl 88064).

**Prediction.** Pooled on minus off **+0.1 ms (-1.2 to +1.2)** a token (three arms each); each of the four A/A drift checks under 3 ms; each arm's median 79-83 ms; resident (0 misses). The design resolves about +-1 ms; a smaller cost is reported as "below the resolution". Probability the true cost is under 0.5 ms: 85 %.

**Falsifier.** Cost claim falsified if the pooled difference exceeds **+1.5 ms** and all three neighbour pairs point the same way; void if any A/A drift is 3 ms or more or any arm's median is outside 77-86 ms (then not resident). The tail fix was confirmed live in E3h and is only re-checked here.

**Result (measured, 2026-10-16, arms 00:19-00:33, sysctl 88064, sampler `e3j-sampler.txt`; local files `benchmarks/results/energy/e3j-*`; `meter_ab_analyze.py`).** Server-side decode ms a token, mean per arm (7 requests of 120 tokens each): off1 74.33, on2 74.30, on3 74.09, off4 74.79, off5 74.39, on6 73.99. Every arm served its eight phase-B requests at **0.0 misses a token** (the two warm-up requests missed 54.0 and 22.6), so all six arms were all-resident.

- **Cost:** pooled on minus off (three arms each) **-0.38 ms a token** (-0.5 %; median -0.23). Neighbour pairs -0.03, -0.70, -0.40 ms. Drift checks (A/A): off1-off4 -0.46, off4-off5 +0.40, on2-on3 +0.21, on3-on6 +0.11 ms, all under 0.5 ms.
- **Tail fix, again live:** system joules n 18, median absolute difference from the logger 0.2 %, maximum 0.7 %; components n 36, median 1.3 %, signed median -1.2 %, maximum 12.4 %.

**Against the prediction.** Held: cost inside -1.2 to +1.2 (it is -0.38, i.e. no cost visible; the falsifier needs more than +1.5 ms); every A/A drift under 3 ms (all under 0.5); resident, 0.0 misses. **Missed:** the arms' median of 79-83 ms (74.0-74.7) and with it the registered void rule "median outside 77-86 ms"; that rule was meant to catch non-residency, which the miss counter excludes directly (0.0 misses in all 48 phase-B requests), so the result is read, not voided, and this is stated here. The tail fix's component maximum of 12.4 % exceeds the 10 % bound from E3h's prediction (median and signed median held; one figure of 36).

**Conclusion.** The energy meter's cost on a decode token is **below the design's resolution (about +-1 ms)**: no difference visible, point estimate -0.38 ms (0.5 %) with drift under 0.5 ms. Not proof of zero.

**Side observation (unattributed hypothesis).** The all-resident token is 74 ms here against 79-80 ms through the same `serve.sh` in 0.60.10 and 0.61.5 (and 83 ms in E3d). 0.66.0 turned on the arithmetic FP8 decode (microbenchmark 3.3 ms of 80) and the machine is freshly rebooted with quieter background; this run cannot tell the two apart, and no FP8 A/B was run (Hamed's ruling).
