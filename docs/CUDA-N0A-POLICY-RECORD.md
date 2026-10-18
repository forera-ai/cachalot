# CUDA lane N0a: offline resident-fraction policy table (record)

Written 2026-10-18 against runtime 0.66.11. Phase N0a, item 2 of `docs/CUDA-RESEARCH-PLAN.md` section 6. Mac only, no GPU, no credits, no instance. Labels follow plan section 0 (recorded, derived, predicted, hypothesis, unavailable).

**Status of this file at the time of writing: predictions only.** Sections 1 to 4 were written and committed before the script ran. Section 5 (results) is appended after, and nothing above it is edited afterwards; a prediction edited after its run is void (plan section 12).

## 1. Question and baseline

Plan question CUDA-Q5 and the premise of hypotheses CUDA-H5, H6 and H9: on routing traces already recorded on the Mac, how do the hit-rate curve and the policy gaps change when the resident fraction falls from the Mac's 34 % (48 GiB, 5,179 slots of 15,360 experts) to the 8 to 25 % that a 32 GB card could give experts?

**Baseline (recorded, not reproduced here).** The shipped Mac store is LRU at 5,179 slots. Live decode at 34 GiB measured 43.0 misses a token, 28 GiB 51.0, 22 GiB 62.2 (HANDOFF 18.138 to 18.140); at 48 GiB 25.0 (ledger DS-MISS-48). Policies SLRU and LFU were compared at 34 % only, with small gaps (`simulate_policies.py`). Live numbers are not the simulator's numbers: they carry the real prefetch and a different prompt mix.

**Instrument.** A new script, `benchmarks/cuda_offline_policy_table.py`, replays a routing trace through `Store` of `simulate_policies.py` exactly as `cache_sim.simulate` does (tau 0, no prefetch, no LFU decay), but records the misses of every decode token so that a paired block bootstrap can bound each policy gap. Its aggregate counts must equal `cache_sim.simulate` for every cell (the consistency check below).

**Workload and trace limits.** Expert-use pattern of DeepSeek V4.1 Flash, top-6, 40 layers, 384 experts, expert record 9,950,000 B. Only `routing-trace-deepseek-long` holds at least 1,000 decode tokens (1,972, from the Hermes-shaped replay of 2026-10-08). The other DeepSeek traces (114, 163, 230 and 32 decode tokens) are below the simulator's own floor and are run as smoke-grade only, never as evidence. Everything in this record is a statement about those traces, not about a CUDA machine: no copy cost, no VRAM overhead, no prefetch is modelled.

## 2. Cells

Resident fractions 8, 12, 17, 25 and 34 % of 15,360 experts (slots 1,229, 1,843, 2,611, 3,840, 5,222), and the Mac's shipped 5,179 slots (33.7 %) as the anchor. Policies LRU, SLRU, LFU. Prefill order: the runtime's popularity order (as `cache_sim.py`). Bootstrap: paired, over blocks of 20 consecutive decode tokens, 2,000 resamples, seeded.

## 3. Hypotheses and predictions (registered before the run)

All numbers are **predicted**; the intervals come from the recorded Mac figures above and from the shape of the curve, and are wide on purpose.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| N0a-P1 | consistency: the script's aggregates equal `cache_sim.simulate` in all 15 cells, and a second run reproduces every per-token sequence | holds exactly (equal to the last digit) | any cell differs by one miss |
| N0a-P2 | LRU misses a token on `deepseek-long` at 34 % / 25 % / 17 % / 12 % / 8 % | 22 to 36 / 38 to 58 / 62 to 95 / 88 to 130 / 112 to 160 | a value outside its interval (a miss of the curve's shape, not a defect) |
| N0a-P3 (CUDA-H6 premise) | relative miss reduction against LRU of the better of SLRU and LFU, at 17 % and at 34 % | 34 %: 0.5 to 4 %; 17 %: 2 to 9 %; ratio 17 % / 34 % between 1.5 and 5 | ratio below 1.5, or the 17 % reduction inside its bootstrap interval, per plan H6 |
| N0a-P4 | probability that the plan's H6 test (ratio at least 2, outside the noise band) passes on this trace | about 0.55 | n/a (a probability, scored by the outcome) |
| N0a-P5 (CUDA-H9 premise) | marginal misses a token saved per extra GiB of resident experts, LRU, between 8 % and 17 % | 2.0 to 4.0 | outside the interval |
| N0a-P6 (CUDA-H9, derived) | cost per miss ratio, Mac internal SSD 2.0 ms against a pinned host-to-device copy of one 9.95 MB record at 10 to 25 GB/s (0.40 to 1.0 ms, **assumed**, N1 measures it) | 2.0 to 5.0 | n/a (arithmetic on assumptions); consequence: H9's 5x threshold is at the edge or unmet, N1 decides |

## 4. Stop rules (from the plan)

* A policy gap inside the paired bootstrap interval is a null result and is recorded as such.
* A trace with fewer than 1,000 decode tokens is smoke-grade and supports no conclusion.
* If a cell of the consistency check differs, results are void until the script is fixed.
* Estimated cost: Mac CPU only, a single Python process. A timing probe on one cell comes first; a total beyond about ten minutes is announced before it runs (Hamed's standing rule on long tests).
