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

## 5. Result, explanation, generalization, bottleneck after, kind

To be written after the run.
