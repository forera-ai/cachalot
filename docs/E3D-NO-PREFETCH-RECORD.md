# E3d record: does the unexplained decode power stay when the drive reads are switched off? (charter L5b, follows `docs/E2I-INTERNAL-SSD-RECORD.md`)

Fields 1-4 written 2026-10-10, before the run, against runtime 0.62.38. Result and explanation are appended after the run.

## 1. Baseline

E3b (HANDOFF 18.123), DeepSeek exact decode through `serve.sh`, internal bank, 48 GiB budget: the E1b component fit under-predicts `PSTR` by **+12.6 W** in the all-resident phase (PSTR 78.1 W, 87.5 ms a token, about 6.8 J a token) and **+13.9 W** in the read-bound phase (81.9 W, 145 ms, 11.9 J). The internal drive read 1.82 GB/s in the all-resident phase even though the phase had 0.0 misses a token: those reads are speculative prefetch loads from the next-layer predictor (`CACHALOT_PREDICT_TOPK`, default on). E2i (HANDOFF 18.125) priced the internal drive at about 1.1 W per GB/s, so the drive explains about 2.8 W of the all-resident excess and about 4.7 W of the read-bound one, which leaves about **9.8 W and 9.2 W unexplained**, almost equal in the two phases although the read rate doubled. E3c ruled out a synthetic read pattern, E3b the fans and a missing Energy Model channel.

## 2. Hypotheses and predictions (tags: prediction)

Intervention in one line: the same energy arms with `CACHALOT_PREDICT_TOPK=0` (no next-layer prediction, hence no speculative loads; demand misses still read). In the all-resident phase B the experts of the repeated prompt are resident, so the drive should be idle.

- **H-kernels-server.** The remaining power is the model's own kernels and the server's CPU-side work, which exist with or without the drive. Probability 50 %. Predictions for phase B:
  - Drive read rate from the store counters: **below 0.1 GB/s** (a precondition; above 0.3 GB/s the arm did not remove the reads and is void).
  - The component fit's error against `PSTR`: **+9.8 W (7.0 to 12.5)**, that is the E3b +12.6 W less the drive's 2.8 W.
  - Seconds a token: **76 ms (72 to 82)** (the prediction costs about 5 ms of an all-resident token, LEDGER DS-PRED-FLOOR: 72.8-76.8 ms with `PREDICT_TOPK=0` against 79-80 shipped).
  - System energy a token: **5.5 J (4.7 to 6.5)** at the thermal state of this session; E3 and E3b differed by 6 to 11 % between sessions, so the comparison with 6.4 and 6.8 J is made with that margin.
- **H-prediction-work.** The unexplained power is mostly the prediction itself (the extra router and norm evaluation on the GPU, the host thread that ranks and submits loads, the reader threads' wake-ups), not the bytes. Probability 30 %. It predicts a fit error in phase B **below 6 W**.
- **H-other.** Something that neither switch touches (resident wired memory, thermal state of the session). Probability 20 %. It predicts a fit error in phase B above **12.5 W**, or a fit error that follows the session's idle watts rather than the arm.
- Phase C (fresh prompts, demand misses only): the fit error **9.0 W (6 to 12)**, equal to B's within 3 W (the equal-in-both-phases pattern of E3 and E3b should persist if it is not the drive). Misses a token should not change (prediction had netted about 1 ms in LEDGER DS-PRED-FLOOR); the drive's rate in C should be lower than E3b's 3.56 GB/s.

## 3. Falsifier

H-kernels-server is falsified if the phase B fit error is **below 6 W** (the excess then went with the prediction, which is H-prediction-work) or **above 12.5 W** (more than the drive's removal can explain; H-other). The run is void if phase B's drive rate is 0.3 GB/s or more, if the sampler is not running (frozen counters), or if the idle phases show a fit error beyond +-3 W. An error between 6 and 7 W or between 12.5 and 14 W is reported as unresolved against the fit's 1.3 to 3 W noise, not as a result. A fit error that differs from E3b's by less than 3 W but with the idle watts 4 W or more apart is read as the thermal state, not the arm.

## 4. Intervention

None to the runtime. `benchmarks/energy_arms.sh` gains `E_TAG=name`, which prefixes every output file so this run cannot overwrite the E3 and E3b files; it passes the extra environment (`CACHALOT_PREDICT_TOPK=0`) through to `serve.sh`, as before. Command for Hamed, in two terminals, Cachalot Lab closed, display as it is (note the fan RPM and the idle watts of phase A in the result):

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 800 -o benchmarks/results/energy/e3d-sampler.txt
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && E_TAG=e3d E_FULL=1 CACHALOT_PREDICT_TOPK=0 ./benchmarks/energy_arms.sh > benchmarks/results/energy/e3d-run.log 2>&1
```

About 10 minutes of run (server load, warm-up, 90 + 120 + 150 + 90 s of phases); the sampler's 800 samples cover it with margin. The analysis is `benchmarks/energy_e3.py --log benchmarks/results/energy/e3d-e3b-arms.jsonl --driver benchmarks/results/energy/e3d-arms-driver.jsonl`.

## 5. Result (measured; 2026-10-10, run 22:38:13-22:46:56, Cachalot Lab closed, fans 1,004-1,092 RPM throughout; `benchmarks/results/energy/e3d-e3b-arms.jsonl`, `e3d-arms-driver.jsonl`, sampler `e3d-sampler.txt`; analysis `benchmarks/energy_e3.py`)

`CACHALOT_PREDICT_TOPK=0`, otherwise the E3b arms. Per phase (IOReport watts, `PSTR`, the E1b fit, error = `PSTR` minus fit):

| phase | s | tokens | ms/token | PSTR W (sd) | fit W | error W | drive GB/s | misses/token | system J/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A idle server | 90 | 0 | - | 28.2 (9.3) | 28.6 | -0.4 | 0 | - | - |
| B all-resident | 129 | 1560 | 83 | 71.3 (8.4) | 60.5 | **+10.9** | **0.00** | 0.0 | **5.90** |
| C read-bound | 159 | 1080 | 147 | 73.2 (7.1) | 59.2 | **+14.0** | 2.09 | 33.5 | 10.78 |
| D idle server | 90 | 0 | - | 28.6 (8.6) | 28.7 | -0.1 | 0 | - | - |

Ratio C over B for system energy 1.83. The session was cool (idle 28.2 and 28.6 W, fans about 1,020 RPM, against E3b's idle 29.7 and 38.1 W at 1,000 to 2,500 RPM).

Against the predictions: drive rate in B below 0.1 GB/s: **held** (0.00; the arm removed the reads). Fit error in B 9.8 W (7.0 to 12.5): **held** (+10.9, 1.1 above the point). Seconds a token 76 ms (72 to 82): **missed** (83 ms; against E3b's 87.5 and E3's 86 it is 3 to 4.5 ms faster, less than the 5 ms of LEDGER DS-PRED-FLOOR). Energy a token 5.5 J (4.7 to 6.5): **held** (5.90). Phase C fit error 9.0 W (6 to 12): **missed** (+14.0). Misses a token unchanged (33.5 against 33.7): **held**; drive rate in C below E3b's 3.56 GB/s: **held** (2.09). Equal within 3 W in the two phases: **missed narrowly** (C minus B is +3.1 W). Falsifiers: B error below 6 W (H-prediction-work): **not fired**; above 12.5 W (H-other): **not fired**; the run is not void (idle errors -0.4 and -0.1 W).

## 6. Explanation

With the drive silent (0.00 GB/s) and the prediction off, the all-resident decode still draws **10.9 W** above the component fit, against 12.6 W in E3b: the drive and the prediction together account for about 1.7 W of it, not the 2.8 W predicted from the drive alone (the two sessions differ in thermal state, so the 1.1 W difference is inside what the session can move). In the read-bound phase the drive reads 2.09 GB/s (about 2.3 W by the E2i line) and the error is 14.0 W, so about 11.7 W remain. **About 11 W of decode power is unexplained in both phases, independent of the drive, of the prefetch reads and of the read rate.** H-kernels-server (the model's own kernels and the server's CPU work) survives; H-prediction-work (the prediction's GPU and thread work) is not supported (the error did not fall below 6 W); the 10.9 against 12.6 W is the whole size of what the prediction and prefetch could have been. The read-bound phase's extra 3.1 W over B is what the drive and its misses plausibly add (E2i: 2.3 W at 2.09 GB/s), so the pattern "equal in both phases" of E3 and E3b is now "B's excess is the floor, C adds the drive".

## 7. Generalization

Hardware- and workload-specific (this Mac, DeepSeek exact decode, one session, cool state). The transferable statement: the component fit (a sum of counters calibrated on single-source loads) leaves about 11 W of a real quantized-MoE decode unexplained that no storage activity produces; any joules-per-token claim for decode has to be taken at the system rail, not built from components.

## 8. Bottleneck after

Speed: unchanged (nothing shipped). Energy: about 11 W of decode power that belongs to the model's kernels, the server's CPU work, or the resident memory. Remaining tests, each separating one: (ii) the kernels in-process without the server (floor harness), (iii) a smaller resident budget at the same work.

## 9. Kind

Research (predicted, measured, explained in part): the headline hypothesis survived its falsifier, two of the numeric predictions missed (seconds a token, phase C) and are recorded as missed.
