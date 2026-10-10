# E1b record: what is wrong with the system-power residual (charter L5b, follows `docs/E1-DRAM-CALIBRATION-RECORD.md`)

Fields 1-4 written 2026-10-10, before the run, against runtime 0.62.32. Result and explanation are appended after the run.

## 1. Baseline

E1 (HANDOFF 18.119): the SMC `PSTR` rail minus the sum of the IOReport components (cpu, gpu, ane, dram, dcs, amcc, display_media) was 17.8-20.1 W at 0-176 GB/s of GPU streaming, 13-14 W at 361, 9-11 W at 543 and 2.5-3.5 W at 735 GB/s. All load in E1 came from one source, the GPU streaming, so the components moved together (collinear) and E1 cannot say which one is mis-counted. A constant remainder (SSD, fans, PSU losses) cannot shrink with load, so either PSTR responds to a component by less than 1 W per IOReport W, or two IOReport channels overlap at load.

## 2. Hypotheses

- **H-uniform.** PSTR responds to every IOReport component by about the same factor below 1 (E1's PSTR slope 138.9 pJ/B against 160.3 for the component sum is a ratio of 0.87); nothing overlaps. Prediction: regressing PSTR on the components across steps with independent variation gives every coefficient in 0.75-1.0, and the largest minus the smallest coefficient is below 0.3.
- **H-overlap-memory.** The memory channels (dcs, amcc) are partly counted inside gpu or dram, so adding them over-counts: Prediction: the dcs and amcc coefficients are well below the gpu and cpu ones (below 0.6) while cpu and gpu are near 1.0.
- **H-gpu.** The gpu channel over-reports relative to PSTR (a low gpu coefficient, below 0.7) with the others near 1.

I expect **H-uniform** (the E1 ratio of 0.87 is the evidence for it), at about 55 %.

## 3. Falsifier

H-uniform is falsified by: any coefficient outside 0.5-1.3, or the spread between the gpu, cpu and memory coefficients above 0.3. A CPU-only load that leaves the residual unchanged at about 18 W (within 3 W) while GPU loads shrink it points to a GPU-side cause (H-gpu or H-overlap-memory) and is the first thing to read. A joint fit with R^2 below 0.95 says the structure is not a linear sum at all. A run in which the DRAM counter is frozen is void, not a result.

## 4. Intervention

None to the runtime. New instrument `benchmarks/residual_test.py`: eight 30 s load steps run twice in opposite order (idle; CPU compute at 4 and 16 threads, cache-resident `np.sin`; CPU memory copy of 256 MiB arrays at 8 threads; GPU compute-only, a cache-resident bf16 matmul; GPU stream at 50 and 100 %; GPU stream with 16 CPU threads), the first 5 s of each step excluded. IOReport energy by component and `PSTR` are read once a second by a user-level reader. Needs Hamed's `sudo powermetrics` sampling for the whole run (about 9 minutes of load, 13 for the sampler's -n 800).

## 5. Result (measured, 2026-10-10, 02:04-02:12, `benchmarks/results/energy/e1b-residual.jsonl`, sampler `e1b-sampler.txt`)

Sixteen 30 s steps (two passes, opposite order), means over the last 25 s; paired steps agree within about 3 W of PSTR. Watts (IOReport component, then SMC `PSTR`, and PSTR minus the component sum):

| step | cpu | gpu | dram | dcs+amcc | sum | PSTR | residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| idle (A, B) | 2.0, 3.2 | 0.5, 0.4 | 1.4 | 4.4, 4.5 | 8.8, 10.0 | 26.0, 29.3 | 17.2, 19.3 |
| cpu 4 threads | 20.6, 21.5 | 0.4 | 2.1 | 9.2-9.8 | 32.8, 34.3 | 54.1, 55.2 | 21.3, 21.0 |
| cpu 16 threads | 46.8, 47.3 | 0.4 | 2.4-2.5 | 11.3 | 61.4, 62.0 | 90.3, 92.6 | 29.0, 30.6 |
| cpu memory copy, 8 threads (445 GB/s) | 35.1, 34.9 | 0.5 | 21.1 | 37.4 | 94.6, 94.0 | 109.2, 110.3 | 14.5, 16.2 |
| gpu compute only (matmul, no DRAM traffic) | 3.3 | 49.2-49.3 | 2.9-3.0 | 10.5-10.6 | 66.4, 66.7 | 94.3, 97.2 | 28.0, 30.5 |
| gpu stream 50 % (360 GB/s) | 2.0-2.4 | 23.0-23.5 | 19.0 | 20.9-21.2 | 66.0, 66.2 | 79.4, 81.7 | 13.4, 15.5 |
| gpu stream 100 % (738 GB/s) | 2.8-2.9 | 48.3-49.3 | 38.8-39.0 | 38.3-38.5 | 129.2, 130.0 | 132.9, 135.4 | 3.8, 5.4 |
| gpu stream + 16 cpu threads | 46.8-47.5 | 48.0-48.5 | 36.7-36.8 | 38.3 | 170.3, 171.7 | 189.7, 192.1 | 19.4, 20.5 |

Joint fits of PSTR on the components over the 16 steps (R^2 0.999, rms 1.3-1.4 W): `PSTR = 18.4 W + 1.27 x cpu + 1.24 x gpu + 0.34 x dram + 1.04 x (dcs + amcc)` (dcs and amcc merged); with dcs and amcc separate they are -0.40 and 1.21 (collinear, not separately determined) and dram 0.51. Without a dram term the rms rises to 1.9 W; with dram forced to 1.0 it rises to 2.9 W (R^2 0.993); without dcs and amcc it rises to 4.2 W. A single common coefficient on the sum fits poorly (R^2 0.976, rms 7.3 W).

Against the hypotheses: **H-uniform is falsified** (coefficients 0.34 to 1.27, spread 0.93 against the 0.3 limit; the 0.75-1.0 range missed for cpu and gpu, which are above 1). **H-overlap-memory** predicted dcs and amcc below 0.6 with cpu and gpu near 1.0: not matched (dcs+amcc is 1.04; it is the dram channel that is low). **H-gpu** predicted a low gpu coefficient: not matched (1.24). The falsifier's CPU-only pattern (residual unchanged under CPU load) did not occur: the residual grows with CPU load (+4 W at 4 threads, +12 W at 16). The joint fit R^2 of at least 0.95 held: PSTR is a linear sum of these components.

## 6. Explanation (hypothesis; the fit is measured, the reason is not)

E1's shrinking residual came from its load being mostly DRAM traffic: PSTR rises by only about a third of a watt per IOReport DRAM watt, while it rises by 1.2-1.3 W per CPU or GPU watt. A coefficient above 1 is what a DC-input rail shows when the component counters are die power and about 20 % is lost in the voltage regulators and supply (efficiency about 0.8); the dcs+amcc coefficient near 1.0 and the dram coefficient near 0.3 are then the surprise. Candidates, none tested: (a) the `DRAM0_n` counter reads about three times high against what the supply delivers (its pJ per byte would then be about a third of 49.9 on the PSTR side); (b) part of the DRAM channel's energy is also inside dcs or amcc, so the regression moves it there; (c) the DRAM supply is not fully on the PSTR rail. The constant part of PSTR, 16-21 W depending on the specification (18.4 W in the merged fit), is what the loaded components do not explain: the SSD, fans, USB and the rest of the board at idle, plus the supply loss on the idle components.

## 7. Generalization

Hardware-specific (M3 Ultra Mac Studio, this OS build, one session, display on). The method generalizes: when the sources overlap, load each source alone and regress, as the single-source E1 load could not. For the energy lane: a joule figure built by adding the IOReport components overstates the system's draw for DRAM by up to three times and understates it for CPU and GPU by about 20-27 %, so a total must be a fit to PSTR (or a meter), not a sum.

## 8. Bottleneck after

Speed: unchanged. Energy: the SSD (E2) is the missing component, and its detectability is limited by the fit's 1.3-1.4 W rms against a drive that draws about 3-6 W when busy (a nominal figure for NVMe drives, not measured here); an inline meter on the X10Pro would remove the dependence on this fit.

## 9. Kind

Research (predicted, measured, partly explained): the linear-sum structure prediction held; every hypothesis on the individual coefficients missed, and the miss is recorded.
