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

## 5. Result, explanation, generalization, bottleneck after, kind

To be written after the run.
