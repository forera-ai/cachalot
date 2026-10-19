# CUDA lane N3 stage B: results (what the host issue time is made of; levels L0 to L5; RTX 5090)

Run 2026-10-19 under `docs/CUDA-N3B-EXPERIMENT-RECORD.md` (11 predictions committed in d2d3a15 before any stage B code). Harness: `benchmarks/cuda/overlap_arena.py` at scaffold commit 201ed69 (`REVISION` dirty=0), runtime 0.66.32; trace `routing-trace-deepseek-long` (sha256 prefix `f961e9e7a8971317`). Instance: an RTX 5090 virtual machine of the S2 to N3 class (driver 580.159.03, torch 2.7.1+cu128, 7 vCPUs, 24 GiB RAM, PCIe 5.0 x16, kernel 6.8). Raw rows, manifests and per-token samples: `docs/cuda-results/n3b-rtx5090-2026-10-19/`. Labels follow plan section 0: measured on this one instance unless it says derived or hypothesis. The compute is a stand-in; the host is a Python loop over torch; nothing here says anything about a native runtime, a real model, a V100, or another provider's hosts.

## 1. What ran

Three VMs were needed. Two had a GPU the open NVIDIA driver would not initialize (the second one's kernel log, read by Hamed, said "not supported by open nvidia.ko because it does not include the required GPU System Processor (GSP)", with the GSP firmware present and the packages and module options identical to working VMs; the first showed the same symptom, `nvidia-smi` failing with the module not loaded, a console reboot changed nothing, and it also lost its network after the reboot); the third worked from its first login (RTX 5090, driver 580.159.03, DNS and TCP fine). On it, `check`, `setup`, `doctor`, `smoke` and `n3bsmoke` (14 cells, the first GPU run of the `P0L` arm, the bare level and the CUDA-graph capture) were clean and `n3bfull` ran 43 cells in 679.9 s against the 1,200 s cap I set. **All nine gates true in the smoke and in the full run** (event hash, determinism, counts, `chain2`, readback, calibration, control drift, closure of the accounting at the levels that record it, sanity), no failed cell, status complete. The graph captured and calibrated (-0.8 % against C = 1.0 ms); stand-in calibration errors -0.25 % (C = 1.0) and +3.3 % (C = 0.3, inside the 5 % gate). Control drift of every group's two `N` cells: at most 0.14 %.

This instance's source records (the 64 pinned records, each copied alone): median **c = 0.1937 ms**, 4 of 64 slow (6.25 %): the fast population, as in N2 and N3.

## 2. Results (17 % resident, 400 decode tokens, token 0 a warm-up; p50 per token, ms)

**C = 1.0 ms a layer.** Floor = mean of the group's two `N` cells; `E` = wall minus floor; `H` = `1 - E / E_D` within the group (`L1` to `L3` have no `D` arm; their `H` is against the `L0` group's `E_D` = 9.991).

| level | floor | `E_D` | `E_P0` | `E_P0L` | `E_P40` | `E_O` | `H_P0` | `H_P0L` | `H_P40` | `H_O` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 (stage A as run) | 40.145 | 9.991 | 7.672 | 7.200 | 7.195 | 2.536 | 0.232 | 0.279 | 0.280 | 0.746 |
| L1 no digest | 40.148 | | | 6.805 | | 2.190 | | 0.319 | | 0.781 |
| L2 lean bookkeeping | 40.151 | | | 6.654 | | 2.067 | | 0.334 | | 0.793 |
| L3 pre-built views | 40.152 | | | 6.614 | | 1.962 | | 0.338 | | 0.804 |
| L4 bare events | 39.995 | 9.637 | 6.886 | 6.448 | | 1.777 | 0.285 | **0.331** | | 0.816 |
| L5 one graph replay a layer | 39.949 | 9.460 | | 6.254 | | 1.354 | | **0.339** | | **0.857** |

(Recorded predictor's bound from the serial model on this window: 0.365.)

**Host measures a token (the oracle arm `O`, C = 1.0).** Host issue time (host clock: bookkeeping, copy issue, launch): L0 2.951, L1 2.590, L2 2.455, L3 2.395, L4 2.082, L5 **1.210**; the `N` floor's host issue time: 0.84 (L0 to L3), 0.67 (L4), **0.247** (L5). GPU idle time between layers (the `host_gap`, levels that record events): `N` 0.198; `O` 2.121 (L0), 1.788 (L1), 1.663 (L2), 1.534 (L3). Compute stall of `O`: 0.22 ms at every level that records it (L0 to L3).

**C = 0.3 ms a layer.** L0: floor 12.613, `E_D` 9.949, `E_P0` 9.231 (`H` 0.072), `E_P0L` 9.128 (0.083), `E_P40` 10.862 (**-0.092**), `E_O` 3.462 (0.652). L4: floor 12.466, `E_D` 9.618, `E_P0` 8.631 (0.103), `E_P0L` 8.655 (**0.100**), `E_O` 2.702 (0.719). Compute-lane busy against the floor at L0: `P0` +5.6 %, `P0L` +5.8 %, `O` +5.4 %, `P40` +1.3 %, `D` -1.1 %; at C = 1.0 +1.0 to +1.3 %.

## 3. Scoring the registered predictions

| id | outcome |
|---|---|
| B-P1 (digest off lowers `O`'s host gap by 0.15 to 0.60 ms) | **held**: 2.121 to 1.788 = **-0.333** ms (host issue time -0.36) |
| B-P2 (lean bookkeeping lowers it further by 0.15 to 0.50) | **missed low**: **-0.125** ms (1.788 to 1.663); the falsifier (under 0.05, or a rise) did not fire |
| B-P3 (pre-built views lower it further by 0.10 to 0.45) | **held**: **-0.129** ms (1.663 to 1.534) |
| B-P4 (bare `N` host issue time at most 0.50 ms; falsifier at least 0.70) | **missed high**: 0.674 and 0.669 ms; the falsifier did not fire |
| B-P5 (`E_O` at L4 between 1.0 and 2.0 ms; L0: 2.51) | **held**: **1.777** ms (L0 here 2.536) |
| B-P6 (`E_D` at L4 between 8.8 and 9.9 ms) | **held**: 9.637 ms |
| B-P7 (`H_P0L` at L4 between 0.26 and 0.365) | **held**: **0.331** (and 0.339 at L5) |
| B-P8a (`H_P0L - H_P0` at L0 between +0.02 and +0.09) | **held**: **+0.047** (0.279 against 0.232) |
| B-P8b (`|H_P0L - H_P40|` at L0, C = 1.0, at most 0.04) | **held**: 0.0005 |
| B-P8c (`H_P0L - H_P40` at L0, C = 0.3, at least 0.10) | **held**: **+0.174** (0.083 against -0.092) |
| B-P9 (the `N` wall at L5 lower than L4's by 0.10 to 0.50 ms, or `unavailable`) | **missed low**: **-0.046** ms (39.949 against 39.995); the falsifier (under 0.03) did not fire; the graph built and calibrated, so `unavailable` does not apply |
| B-P10 (`H_P0L` at L4, C = 0.3 at most 0.25) | **held**: 0.100 |
| B-P11 (43 cells inside 20 minutes, gates hold, no failed cell) | **held**: 679.9 s, all gates true, 0 failed cells |

Of the 13 scored items (the 11 ids, B-P8 in three parts): **10 held, 3 missed without the falsifier firing (B-P2, B-P4, B-P9), 0 falsified.**

## 4. What the result is, and is not

- **Measured: stage A's "host issue loop" was about half instrumentation.** For the oracle at C = 1.0, the exposure falls from 2.536 ms (stage A as run) to 1.354 ms (L5), **-1.18 ms (47 %)**: digest -0.35, lean bookkeeping -0.12, pre-built views -0.10, bare events -0.18, one graph replay a layer -0.42 (differences of `E_O` between consecutive levels). The other **1.35 ms is what remains in this harness**. B-P5 holds (L4: 1.78 ms), and the stage A reading ("2.0 of the 2.5 ms is the host's issue loop") was an upper bound with about half of it removable by my own instrumentation and Python choices.
- **Derived: what the remainder is.** At L5 the oracle's host issue time is 1.210 ms against the floor's 0.247: 0.96 ms of host work for 50.4 copies a token, about **19 microseconds a copy** (bookkeeping included), the cost of torch's `copy_` dispatch from Python. That fits the remaining 1.35 ms of exposure (GPU idle while the host issues the layer's copies). It is a property of Python and torch; a native issue path or a batched memcpy API could change it, and nothing here measures one (hypothesis).
- **Measured: the graph removes host time but not wall time.** The `N` floor's host issue time falls from 0.67 to 0.247 ms at L5, and its wall time by only 0.046 ms (B-P9 missed): the GPU is the floor there (the stand-in's compute is 40 ms a token and the host launches ahead of it), so launch overhead was already hidden; the graph pays in the copy arms, where the host is on the critical path between layers (`E_O` -0.42 ms).
- **Measured: the recorded predictor is near its bound once the host cost is low.** `H_P0L` rises from 0.279 (L0) to **0.339 (L5)**, 93 % of the serial model's 0.365; the shortfall that stage A put down to the host (0.242 against 0.365) is mostly host and issue order, and what remains (0.026) is the 32 demand copies that cannot be hidden. B-P7 held. The oracle reaches **0.857**. The ceiling for a recorded predictor on this window is therefore predictor coverage, not the host.
- **Measured: issue order, not window, explains stage A's `P0` against `P40` at 1 ms.** `P0L` (issued after the launch, no wait) equals `P40` within 0.0005 at C = 1.0 and beats `P0` by 0.047 at L0 and at L4 (0.331 against 0.285): issuing the prefetch copies before the layer's launch delays the launch. At **C = 0.3 the window matters**: `P40` is 0.17 below `P0L` (-0.092 against 0.083); and at C = 0.3 no host lever rescues the predictor: `H_P0L` is 0.100 at L4, `H_O` 0.719. B-P8a, B-P8b, B-P8c and B-P10 all held.
- **Interference (CUDA-H7), restated:** the compute lane is +5.4 to +5.8 % slower for `P0`, `P0L` and `O` at C = 0.3 and +1.0 to +1.3 % at C = 1.0 (`P40`: +1.3 % at 0.3, the copies start later in the layer); the plan's 5 % bound is exceeded at 0.3 ms, as in stage A.
- **Not shown.** What a native runtime's issue cost is; whether a batched H2D API, copy-engine queues or fewer, larger copies would remove the 19 microseconds a copy (none was available to test); this instance's population only (c = 0.194 ms; on a slow-population instance every copy-bound term is about 1.85 times larger, derived from c alone and not measured here); the 400-token window; any V100 or AMD behaviour.
- **Consequence for the lane (derived).** In this stack, with the copies hidden by an oracle, an issue floor of about 1.35 ms a token (about 3 % of a 40 ms token at C = 1.0, 11 % of a 12.5 ms token at C = 0.3) remains from Python and torch dispatch; the recorded predictor recovers about a third of the demand exposure at C of 1 ms and above (0.34), almost nothing at 0.3 ms. The levers still open are a better or gated predictor (60 % of prefetches are wasted), fewer and larger copies, and a native issue path.

## 5. Cost and housekeeping

GPU time on the instrument: the smoke 30 s, the full run 680 s. Three VMs were used, two of them only for diagnosis (a GSP-less driver fault each); their billed hours and the working VM's are Hamed's to enter in the private cost file (he said he will); the agent did not guess them. The VM is terminated (his word). Housekeeping stands: delete the S1, S2, V1, N2, B1, B2 and N3 ssh configs, known_hosts files and the key (nothing was deleted here).
