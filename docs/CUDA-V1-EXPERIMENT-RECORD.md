# CUDA lane session V1: experiment record (V100 32 GB: the PCIe 3 and no-fp8 contrast, N0b inventory, N1 smoke and baseline, S2 profile)

Written 2026-10-18 against runtime 0.66.18, **before any V100 instance exists and before any V100 measurement**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file afterwards. A prediction edited after its run is void (plan section 12). Labels follow plan section 0. This is the second device of decision D3 (RTX 5090 first, V100 second for contrast). Instrument: `benchmarks/cuda/` (profiles `smoke`, `baseline`, `s2`; the arch gate added in 0.66.19). Nothing here starts an instance: that needs Hamed's yes and his creating it (agent does not provision).

## 1. Baseline (recorded, two RTX 5090 instances)

* Pinned H2D on the 5090: 28.4 GB/s at the 9.95 MB expert size (350 us), 52 GB/s at 64 MiB (83 % of the PCIe 5.0 x16 theoretical 63 GB/s); pageable 24; pinned D2H 55; device-to-device 1,517 GB/s (85 % of the vendor 1,792); launch 2.96 us; router chain 7.5 us a layer; an 8-matmul expert-shaped cell 0.203 ms alone (about 120 TFLOP/s, derived).
* Compute under concurrent pinned H2D: 21 to 26 % slower (S1 instance), 12.3 to 13.3 % (S2 instance); with weights above the L2 18.8 to 22.1 %; a saturating device copy 320 to 364 % (a bandwidth hog, not an L2 test). Batched copy at k = 8: queued separate transfers 0.22 to 0.29 ms an expert, one contiguous transfer 0.20 to 0.24, isolated 0.34 to 0.36.
* The torch 2.7.1+cu128 arch list on the 5090 hosts is `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120`: **no `sm_70`**, so that wheel cannot run kernels on a V100 (compute capability 7.0).
* Vendor figures (**derived inputs**, not facts about the rented machine): V100 32 GB HBM2 at about 900 GB/s; PCIe 3.0 x16 about 15.75 GB/s one direction; fp16 tensor-core peak about 125 TFLOP/s; no bf16 and no fp8 hardware. The doctor reads the real values first, and the intervals below for copies are multiples of the negotiated link's theoretical rate L.

## 2. Session plan and stop rules

Order: (1) check and `setup` with the CUDA 12.6 wheel (the index and the version are env overrides of `session.sh`, listed in the runbook); the setup now stops by itself if the installed torch lists no kernels for the device's compute capability; (2) `doctor`, read it; (3) `smoke` (45 s cap); (4) `baseline` only if the smoke run is clean; (5) `s2` profile, four times if the first is clean; (6) copy back, validate, teardown. Session cap **1 GPU-hour** from Ready (initialization of about seven minutes is not billed on the 5090 class; the console's transactions page is the record). Stop rules as in S1 and S2: stop on a failed cell, a false gate, a truncated run, no kernel image for the device, or 80 % of the cap before the baseline. A defect found on the instance is fixed on the Mac. Any cell with two streams was smoke-tested on a GPU in S2 (the d2d race), so the S2 profile is expected to pass its gates here; it is not new code.

## 3. Predictions (registered before the session)

All **predicted**. L = theoretical one-direction rate of the negotiated PCIe link.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| V1-P1 | the cu128 wheel has no kernels for the V100; the cu126 wheel (2.7.1) lists `sm_70` | cu126 includes `sm_70` with probability 0.85 | `sm_70` absent from the cu126 arch list (then no pinned wheel of this plan works and the session stops after setup) |
| V1-P2 | doctor: compute capability, memory | capability 7.0; VRAM 31.0 to 32.0 GiB total, 30 GiB or more free | capability other than 7.0, or free below 29.5 GiB |
| V1-P3 | negotiated link | PCIe 3.0 x16 with probability 0.65, PCIe 3.0 x8 or narrower 0.15, a newer generation 0.20 | scored by outcome |
| V1-P4 (CUDA-H1) | pinned H2D at 64 MiB | 0.70 to 0.88 x L; and at 9.95 MB at least 85 % of the 64 MiB rate (the 5090 gave 54 %; a slower link makes the fixed cost a smaller fraction) | 9.95 MB below 85 % of the 64 MiB rate |
| V1-P5 (CUDA-H2) | pageable H2D at 9.95 MB against pinned | 0.60 to 0.95 of pinned (the 5090 gave 0.85) | pageable below 0.60 or above 1.00 of pinned |
| V1-P6 | pinned D2H at 64 MiB | within 15 % of the pinned H2D rate | differs by more than 15 % |
| V1-P7 | device-to-device peak (read plus write, 512 MiB buffer) | 650 to 840 GB/s (75 to 93 % of the vendor 900 counted as read plus write of a 900 GB/s memory is a ceiling of about 900) | below 550 or above 900 |
| V1-P8 (CUDA-H3) | single projection launch share; batched g = 8 | single 40 to 80 % of the measured peak; the batched launch of g = 8 is at least 1.5 times faster than the loop | grouping gain under 10 % |
| V1-P9 | fp16 throughput on the expert-shaped cell (m = 128) | 35 to 85 TFLOP/s (no fp8 or bf16 cells run: dtype gates must report them unavailable) | below 25 or above 125 |
| V1-P10 | launch and router chain | launch 3 to 10 us; a 40-layer router chain 0.2 to 1.0 ms | chain above 2 ms |
| V1-P11 | timer overhead | 1 to 6 us | above 10 us |
| V1-P12 (CUDA-H7) | compute slowdown while pinned H2D covers the window, `overlap.resident.h2d` and `overlap.large.h2d` | 2 to 20 % in each (the 5090 gave 12 to 22 %; the slower link makes the copy stream thinner, about 1.7 % of device bandwidth, derived); under 5 % with probability 0.3 | slowdown above 30 %, or the resident cell above the large one by more than 5 points |
| V1-P13 | saturating device copy (`overlap.*.d2d`) | 200 to 450 % (a bandwidth hog on 900 GB/s, same mechanism as on the 5090) | below 100 % |
| V1-P14 | batched copy, per expert at k = 1 / 8 single | k = 1: 0.80 to 1.10 ms (about 9 to 12 GB/s); k = 8: 0.75 to 0.95 ms (the link is the limit, so batching gains under 20 %) | k = 8 more than 25 % below k = 1 per expert (the fixed-cost explanation would then dominate on this link too) |
| V1-P15 | queued loop against one transfer at k = 8 | within 10 % of each other (bandwidth-bound link; probability 0.7) | loop more than 15 % slower or faster than single |
| V1-P16 | control drift (first and last pinned H2D 9.95 MB) | under 8 % (the 5090 gave 2 to 18 %) | 8 % or more |
| V1-P17 | gates and completion | all gates true, no failed cell, smoke 2 to 6 s, baseline under 30 s, the S2 profile under 30 s; probability 0.8 | any gate false, a failed cell or a truncated run |

## 4. What each outcome means for the plan

* A V100 H2D curve gives the cost of an expert-sized copy over PCIe 3 and the size dependence on a slower link: with the 5090 it sets how much of the 5.7-fold cost ratio against the Mac's 2.0 ms miss survives on a link a quarter as fast (plan H9, restated in 17.2).
* If the compute slowdown under copies is similar on both cards, the effect is not tied to the link generation; if it is much smaller on the V100, the generation or the card architecture matters. Two instances of one class already showed a two-fold spread, so a V100 difference of less than that is **not** attributed to the card.
* Nothing here is a statement about DeepSeek, MiniMax or GLM on NVIDIA hardware, and no V100 number transfers to other cards (plan section 11, R10). Working microbenchmarks do not make a CUDA Cachalot.

## 5. Needs Hamed

Create one V100 32 GB instance (the type and region the console offers; read the PCIe form factor), set the private ssh config as for S2 (user and port from the Connect page), and tell the agent. Then the agent runs over SSH for that one instance, no sudo. Terminate when told, log the transactions-page hours in the private cost file.
