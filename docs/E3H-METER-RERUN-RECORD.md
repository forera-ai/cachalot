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
