# E3b record: why the component fit misses real decode by 13-14 W (charter L5b, follows `docs/E3-JOULES-TOKEN-RECORD.md`)

Fields 1-4 written 2026-10-10, before the run, against runtime 0.62.35. Result and explanation are appended after the run.

## 1. Baseline

E3 (HANDOFF 18.122): over idle and both decode phases `PSTR` minus the E1b fit was -0.6 and +1.0 W at idle, **+12.8 W** all-resident and **+14.3 W** read-bound. The fit used only the aggregates `cpu`, `gpu`, `ane`, `dram`, `dcs`, `amcc`, `display_media` of the IOReport "Energy Model" group (565 channels). The logger kept only those aggregates, so the other channels cannot be examined for that run.

Post-hoc look at the E3 log, made before this prediction was written: the error is not a slow ramp. In 30 s windows it is +13.1 W in the first load window, +11.3, +12.4, +12.6, +12.4 W through the all-resident phase and +14.3, +15.0, +15.4, +14.3, +13.1 W through the read-bound one, and falls to +5.5, +0.2, +0.8 W as the load ends. So a slow thermal or fan ramp is not the main cause (it appears within seconds of load); it does not rule out a fast fan or temperature response.

## 2. Hypotheses and predictions (tags: prediction)

- **H-channels.** Power the old aggregates do not capture is carried by other Energy Model channels (GPU sub-blocks such as `GPU0_0`, `GPU CS0_0`, the SRAM channels, media blocks and others). Prediction P1: the sum over all Energy Model channels with energy units that the old aggregates do not use rises by **at least 6 W** from idle to the all-resident decode phase (so about half the gap could be outside the aggregates). Probability I give it: 45 %.
- **H-mixed.** The gap comes from how real decode loads the machine (GPU compute and memory traffic together, many CPU-GPU synchronizations), which no single-source step shows; it would leave the extra channels flat. Probability 35 %.
- **H-other-rails.** The gap is power the IOReport group does not report at all (fans, board rails, the memory subsystem's own supply); the SMC power, current and fan keys would carry it. Probability 20 %.
- P3 (replication): in this run's synthetic steps (the E1b loads) the old fit's error stays within **+-3 W** at every step, as in E1b (rms 1.3 W). If it does not, the fit drifts between sessions and E1b's coefficients are not stable.
- P4: the error in the first 30 s of the all-resident phase and in its last 30 s differ by less than **3 W** (no thermal ramp), as in E3.

## 3. Falsifier

H-channels is falsified if the unmapped channels rise by **less than 2 W** from idle to the all-resident phase. H-mixed is favoured if they stay flat and the SMC keys also stay flat while the error persists at 8 W or more. P3 is falsified by any synthetic step with an error of 8 W or more. A run in which the sampler was not running (frozen counters) is void.

## 4. Intervention

None to the runtime. `benchmarks/power_sources.py` gains `FullSampler` (every Energy Model channel by position, plus the SMC power, voltage, current, fan-speed and temperature keys every 5 s); `benchmarks/dram_calibration.py`'s `Logger`, `benchmarks/energy_logger.py` and `benchmarks/residual_test.py` take `--full`; `benchmarks/energy_arms.sh` takes `E_FULL=1`; `benchmarks/refit_analyze.py` is new. One session: the E1b synthetic steps (two passes, 8 minutes) then the energy arms (about 9 minutes), with Hamed's root `powermetrics` beside the whole time (about 20 minutes), Lab closed.

## 5. Result (measured; 2026-10-10, stage 1 11:24:38-11:32:42, stage 2 11:33:54-11:41:39, Cachalot Lab closed; `benchmarks/results/energy/e1b-refit.jsonl`, `e3b-arms.jsonl` and their `.header.json`, sampler `e3b-sampler.txt`)

**The error replicates.** The E1b fit's error against `PSTR` in the decode phases: all-resident **+12.6 W**, read-bound **+13.9 W** (E3: +12.8 and +14.3); idle A +1.0, D +5.7 W. First and last 30 s of the all-resident phase: +12.2 and +11.1 W. In the synthetic steps of this session the same old fit errs by -6.5 to +4.2 W (rms 3.0 W over the 16 steps).

Per window (PSTR W, old-fit error W, fan RPM): idle 27.2, 0.0, 992; cpu4 59.6, -1.3; cpu16 91.7, +3.1; cpumem8 109.3, -0.7; gpumm 94.0, -6.5; gpustream50 81.5, +1.8; gpustream100 134.4, -3.6; gpu100+cpu16 186.1, 0.0 (1320 RPM); second pass: gpu100+cpu16 191.4, -4.3 (1589); gpustream100 135.6, -1.6 (1956); gpustream50 83.4, +4.2; gpumm 95.8, -4.0; cpumem8 112.5, +2.2; cpu16 90.2, +3.2; cpu4 58.5, -0.4; idle 33.2, +2.5 (2465 RPM). Decode arms: idle A 29.7 W; all-resident 78.1 W (+12.6), read-bound 81.9 W (+13.9), idle D 38.1 W (+5.7).

**System energy a token in this session** (PSTR times seconds a token, from the phase table: 1,440 tokens in 126 s, 1,080 in 157 s): all-resident about **6.8 J** (87.5 ms), read-bound about **11.9 J** (145 ms), ratio 1.74. E3 (a cooler start) read 6.42 and 10.73 J. The decode phases here ran 3.6 and 5.7 W above E3's and the idle phases 4.4 and 7.7 W above (A 29.7 against 25.3 W, D 38.1 against 30.4): the machine had just run eight minutes of up to 190 W of synthetic load, and the fans ran at 1,400-2,500 RPM through the decode arms against 992 at the start of the session. A thermal and fan state adds several watts to the system rail and 6-11 % to a token's system energy between two sessions.

**Which channels.** The channels the old aggregates do not use rise by +24.2 W from idle to the all-resident phase (P1: held, by the number), but that sum includes sub-channels that are parts of `CPU Energy` and parallel views of the GPU, so it double counts and tells nothing about missing power; I should have defined the metric differently. The informative test: `GPU#_#` (a second GPU view) tracks `GPU Energy` (ratio 0.78-0.97 in every GPU window), and adding it, or substituting it, to a refit on the synthetic steps leaves the decode errors at +11.9/+14.4 W and +13.0/+13.0 W. No Energy Model family I tested closes the gap.

**Which rail.** Fitting each SMC power key on the old components over the synthetic steps and reading its residual in decode (B / C): `PSTR` +13.2 / +14.4 W (synthetic rms 3.2); `PVCC` **+11.1 / +11.4** (rms 2.9); `PSVR` +8.1 / +8.9 (2.8); `PMVR` **+4.9 / +4.8** (rms 0.63); the other rails within about +-2 W. So the extra decode power appears on the SoC supply rails and on the memory rail, not in any IOReport channel I checked.

**Fans.** Fan speed does not track the error: the synthetic steps ran at 992-2,504 RPM (the first pass 992-1,320, the second 1,589-2,504) with errors from -6.5 to +4.2 W and no trend between the passes, the all-resident phase ran at 1,590-1,830 RPM with +12.6 W.

Against the predictions: P1 (unmapped channels rise at least 6 W) held by the number but is uninformative (flawed metric); H-channels is **not supported** by the targeted refit; H-mixed and H-other-rails are not separated by this run but the rail residuals favour power outside the channels (PVCC, PMVR); P3 (old-fit error within +-3 W at every synthetic step) **missed** (max 6.5 W; seven of sixteen steps beyond 3 W: gpumm -6.5 and -4.0, gpu100+cpu16 -4.3, gpustream100 -3.6, gpustream50 +4.2, cpu16 +3.1 and +3.2; the 8 W falsifier did not fire); P4 held (-1.1 W).

## 6. Explanation (hypothesis)

The unexplained 12-14 W in real decode sits on the SMC rails `PVCC` and `PSVR` (SoC supplies, +11 and +8 W) and `PMVR` (memory supply, +5 W), and is absent from the IOReport channels the synthetic loads calibrated. The synthetic loads are sequential or cache-resident; decode reads irregular blocks (2-bit expert weights, many small kernels, frequent CPU-GPU synchronization). A plausible reading is that DRAM, the fabric and the system-level cache cost more per byte for irregular access than for a streaming read, which the energy counters, calibrated by streaming, do not show. Not tested. A direct test: a synthetic GPU gather of 9.5 MiB blocks at random offsets, which should reproduce the rail residuals without a model if the access pattern is the cause.

## 7. Generalization

Hardware-specific. The transferable statement: a per-byte energy measured with a sequential stream (49.9 pJ/B for DRAM) is a lower bound for a workload with irregular access, and a system-rail measurement, not a sum of calibrated components, is the total.

## 8. Bottleneck after

Speed: unchanged. Energy: the access-pattern dependence of the memory and SoC rails, and the thermal state (several watts and 6-11 % of a token's energy between sessions).

## 9. Kind

Research (predicted, measured, partly explained): the error replicates and is localized to three rails; H-channels is not supported; P3 missed; the cause is a hypothesis.

## 10. Erratum (2026-10-10, 0.62.37, after the gather test, HANDOFF 18.124)

Section 5 says the decode excess "appears on the SoC supply rails and on the memory rail" (`PVCC` +11 W, `PSVR` +8 W, `PMVR` +5 W) and section 6 reads that as evidence of power outside the channels. The gather test (`docs/E3C-GATHER-RECORD.md`) shows rail residuals of -15 to +10 W in synthetic steps whose `PSTR` is within a few watts of its fit (`PVCC` -14.7 W on small random reads with `PSTR` -4.9 W). A rail's residual therefore moves with the access pattern and the mix of components without extra total power; the rail localization is **suggestive, not established**. What stands from E3b: the error against `PSTR` replicates (+12.6, +13.9 W), it is not a missing Energy Model channel that was tested (a second GPU view), and it is not the fans.
