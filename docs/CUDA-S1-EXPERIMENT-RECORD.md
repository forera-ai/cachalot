# CUDA lane session S1: experiment record (N0b inventory, N1 smoke and baseline on the RTX 5090)

Written 2026-10-18 against runtime 0.66.15, **before any instance exists and before any GPU measurement**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file afterwards. A prediction edited after its run is void (plan section 12). Labels follow plan section 0. Instrument: `benchmarks/cuda/` (scaffold 0.66.11, plus the CPU-rehearsal and REVISION changes of 0.66.15). Decisions D1 to D5 are in plan section 17.1. Itinerary and commands: `benchmarks/cuda/SESSION-RUNBOOK.md`.

## 1. Baseline

Nothing has been measured on a GPU. The scaffold's cell code has run only in a CPU rehearsal with an event shim (the real cell code, no CUDA). Vendor figures below (memory bandwidth 1,792 GB/s for the RTX 5090, PCIe 5.0 x16 about 63 GB/s, PCIe 4.0 x16 about 31.5 GB/s) are **derived** inputs used to size the intervals, not facts about the rented machine; the doctor reads the real values first, and the intervals for the copy cells are stated as multiples of the **negotiated** link's theoretical rate L, so they do not depend on which generation turns up.

## 2. Session plan and stop rules

Order: (1) `doctor` and read it; (2) the `smoke` profile (45 s wall-clock cap); (3) inspect `run-manifest.json` and `rows.jsonl`; (4) only if the smoke run is clean, the `baseline` profile (300 s cap, 2,048 MiB allocation cap); (5) copy back, validate on the Mac, teardown. Session cap 2 GPU-hours (private cost file).

Stop (end the session, bring the files home, no further cells) if any of: the framework does not see the GPU or the wheel does not target its compute capability; any `failed_cells`; any gate false (`copy_checksum`, `kernel_reference`, `sanity`); the smoke run is `truncated`; the session reaches 80 % of its cap before the baseline starts. A defect found on the instance is fixed on the Mac, not on the billed clock.

## 3. Predictions (registered before the session)

All **predicted**. Let L be the theoretical one-direction rate of the negotiated PCIe link (gen5 x16 about 63 GB/s, gen4 x16 about 31.5, gen3 x16 about 15.75). Smoke cells have n = 12 (smoke-grade); the baseline cells have n = 30.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| S1-P1 | doctor, compute capability and toolchain | capability 12.0; the torch `arch_list` contains `sm_120`; driver CUDA version 12.8 or newer | capability other than 12.0, or no `sm_120` in the arch list (toolchain gate fails: stop) |
| S1-P2 | doctor, VRAM | total 31.5 to 32.0 GiB, free 30.5 GiB or more | free below 30 GiB |
| S1-P3 | doctor, negotiated link | gen5 x16 with probability 0.55, gen4 x16 0.30, anything narrower or older 0.15 | n/a (scored by outcome) |
| S1-P4 (CUDA-H1) | pinned H2D bandwidth at 64 MiB, smoke | 0.70 to 0.92 x L; and at 9.95 MB at least 85 % of the 64 MiB rate | 9.95 MB below 85 % of the 64 MiB rate |
| S1-P5 (CUDA-H2) | pageable H2D at 9.95 MB against pinned | 0.25 to 0.60 of pinned | pageable within 20 % of pinned |
| S1-P6 | pinned D2H at 64 MiB | within 15 % of the pinned H2D rate | differs by more than 15 % |
| S1-P7 | device-to-device peak (bytes read plus written, 128 MiB buffer in smoke, 512 MiB in baseline) | 1,350 to 1,700 GB/s, i.e. 75 to 95 % of the vendor 1,792 | below 1,200 or above 1,792 (the latter is a sanity failure) |
| S1-P8 (CUDA-H3) | fp16 GEMV, one 4096 x 2880 projection, one launch, L2 flushed | share of the measured peak 40 to 75 %; the batched launch of g = 8 shares 70 to 90 % | single launch at or above 80 %, or grouping gains under 10 % |
| S1-P9 | framework launch and sync costs, host monotonic | tiny-kernel launch 5 to 20 us; idle synchronize 3 to 20 us; a (kernel, 4-byte D2H, host wait) layer 15 to 60 us, so a 40-layer chain 0.6 to 2.4 ms | a layer chain above 100 us |
| S1-P10 | event-pair timer overhead | 1 to 5 us | above 10 us (then most small cells are below resolution) |
| S1-P11 | the 1 MiB pinned H2D cell is flagged below instrument resolution | probability 0.5 | n/a (scored by outcome) |
| S1-P12 (CUDA-H7) | compute slowdown while pinned H2D copies cover the compute window | below 5 %, in 0 to 4 %, probability 0.7 | 5 % or more |
| S1-P13 | control cell drift (first and last pinned H2D at 9.95 MB) | under 5 %, probability 0.8 | 5 % or more |
| S1-P14 | smoke run completes without truncation inside 45 s | probability 0.9; estimated upper bound 10.1 s | truncated |
| S1-P15 | baseline run (71 cells) completes inside its 300 s cap | probability 0.75; estimated upper bound 56 s | truncated |
| S1-P16 | first-run defect: at least one cell fails with an exception on the real GPU | probability 0.4 (the cell code has never run on CUDA) | n/a (scored by outcome) |
| S1-P17 | gates on a clean run: copy checksums and the matmul float32 reference pass, no sanity violation | all three hold, probability 0.85 | any false |

## 4. What each outcome means for the plan

* The pinned plateau and the H2D curve give the cost of one expert-sized copy: 9.95 MB over the measured plateau. That replaces the **assumed** 0.40 to 1.0 ms of N0a (record 5.3) and lets H9's threshold be re-stated (plan section 17.1).
* CUDA-H1, H2, H3 and H7 receive a smoke-grade answer (n = 12) at best; the baseline profile (n = 30) is the first answer that counts. Nothing is generalized to the V100, and nothing here is a statement about DeepSeek, MiniMax or GLM performance on NVIDIA hardware.
* Working microbenchmarks do not make a CUDA Cachalot; N2 to N5 are not started by this session.
