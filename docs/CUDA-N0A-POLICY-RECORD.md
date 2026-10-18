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

## 5. Results (appended 2026-10-18, after the run; sections 1 to 4 above are unchanged)

Commands (Mac, one Python process each, no other runtime running):

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/cuda_offline_policy_table.py \
    benchmarks/results/routing-trace-deepseek-long/trace.npz --out benchmarks/results/cuda-n0a/deepseek-long.json
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/cuda_offline_policy_table.py \
    benchmarks/results/routing-trace-deepseek-long/trace.npz --fractions 17,34 --policies lru,slru \
    --lfu-decay 16 --determinism --out benchmarks/results/cuda-n0a/deepseek-long-decay.json
```

Trace `routing-trace-deepseek-long`: 1,972 decode tokens, so not smoke-grade. The raw JSON stays local (`benchmarks/results/` is gitignored). Everything below is **measured** by replay on that trace (a trace-and-policy statement; no copy cost, no VRAM overhead, no prefetch), except where marked derived.

### Table (misses a token; gap = misses LRU has and the policy saves, with a paired block-bootstrap 95 % interval)

| resident | experts GiB | LRU | SLRU | SLRU saves | LFU, no decay | LFU, halved every 16 tokens (exploratory) |
|---|---:|---:|---:|---|---:|---|
| 8 % | 11.4 | 90.84 | 87.20 | +3.64 (+1.81 to +5.40), 4.0 % | 130.83 | not run |
| 12 % | 17.1 | 69.64 | 66.02 | +3.62 (+2.02 to +5.20), 5.2 % | 107.55 | not run |
| 17 % | 24.2 | 51.85 | 49.94 | +1.91 (+0.87 to +2.95), 3.7 % | 85.03 | +2.01 (+1.29 to +2.74), 3.9 % |
| 25 % | 35.6 | 35.47 | 34.32 | +1.16 (+0.52 to +1.81), 3.3 % | 58.51 | not run |
| 34 % | 48.4 | 24.78 | 23.97 | +0.82 (+0.48 to +1.19), 3.3 % | 37.82 | +0.30 (+0.12 to +0.49), 1.2 % |
| Mac shipped, 5,179 slots | 48.0 | 25.03 | 24.22 | +0.82 (+0.47 to +1.19), 3.3 % | 38.35 | +0.27 (+0.08 to +0.46), 1.1 % |

The shipped-store anchor, 25.03 misses a token, equals the recorded live figure at 48 GiB (25.0, ledger DS-MISS-48): the replay and the live store agree at that point.

### Scoring the predictions

| id | outcome |
|---|---|
| N0a-P1 consistency | **held.** All 18 cells of the main run and all 9 of the second run equal `cache_sim.simulate` (and, for the decay arm, `simulate_policies.run`) in misses, requests and tokens; a second replay of every LRU and SLRU cell reproduced the per-token series exactly. This is the offline premise of CUDA-H5 only: the CUDA replay itself does not exist yet |
| N0a-P2 LRU curve | **missed on four of five.** Predicted 22 to 36 / 38 to 58 / 62 to 95 / 88 to 130 / 112 to 160; measured 24.78 (held) / 35.47 / 51.85 / 69.64 / 90.84 (all below their intervals). My intervals were anchored on live numbers that include real prefetch and a different prompt mix; the replay curve is flatter. The measured curve still agrees with the live 22 GiB point (62.2 live; the replay at 24.2 GiB reads 51.9) only loosely |
| N0a-P3 policy-gap ratio | **missed, falsifier fired.** Best of SLRU and LFU is SLRU: relative reduction 3.3 % at 34 % and 3.7 % at 17 %, ratio 1.1 (predicted 1.5 to 5; falsifier below 1.5). In absolute misses the ratio is 2.3, but the two intervals overlap, so the plan's CUDA-H6 ("at least twice") is **not established** on the registered definition |
| N0a-P4 probability 0.55 that H6 passes | scored as a **fail** (probability was not far from a coin) |
| N0a-P5 slope, 8 to 17 % | **held.** (90.84 - 51.85) misses over (24.2 - 11.4) GiB = 3.04 misses a token per GiB (predicted 2.0 to 4.0) |
| N0a-P6 cost-ratio arithmetic | **held by construction**: 2.0 ms against 0.40 to 1.0 ms is 2.0 to 5.0, on assumed copy rates that N1 will measure |

### Findings the predictions did not cover (post hoc, labelled)

1. **LFU as `cache_sim` models it (no decay) is far worse than LRU at every fraction** (13 to 40 more misses a token, outside the noise at every size). It is not a candidate policy. With a halving every 16 decode tokens (the `simulate_policies.py` setting; an exploratory arm added after the first result, not registered) LFU matches SLRU at 17 % (+2.01, interval +1.29 to +2.74) and falls to +0.30 at 34 % (+0.12 to +0.49). For that arm the 17 % gap is 6.8 times the 34 % gap in absolute misses and 3.2 times in relative terms, and the intervals do not overlap. So **a frequency-aware policy with decay does show the small-fraction effect CUDA-H6 expects, and SLRU does not (relative)**. This is one trace and one decay constant, chosen from an existing script, after seeing the first result: treat it as a lead for N2, not as a confirmation.
2. **The marginal value of resident memory falls steeply with the fraction** (LRU, misses a token per extra GiB): 3.72 (11 to 17 GiB), 2.50 (17 to 24), 1.44 (24 to 36), 0.83 (36 to 48). A GiB near 17 % resident removes about 3.7 times the misses a GiB removes at the Mac's 34 %.
3. **CUDA-H9, derived, assumption-bound.** H9 expected the value of a GiB of fast memory to be at least 5 times smaller on a PCIe-attached host tier than on the Mac's SSD tier. Multiplying the measured slopes by a cost per miss gives, at the Mac's regime (0.83 misses/GiB at 2.0 ms) about 1.7 ms a GiB (the live fit is 2.2 ms a GiB, HANDOFF 18.140), and at the 17 % regime (about 3.0 misses/GiB at an **assumed** 0.40 to 1.0 ms per miss) 1.2 to 3.0 ms a GiB. The two are within a factor of two of each other, not five apart: the cheaper miss is offset by the steeper curve at smaller fractions. If N1 measures a pinned copy rate in the assumed range, H9 as stated would be falsified. This is a derived expectation, not a measurement.

### What this does not establish

No CUDA claim. Nothing was measured on a GPU; the copy costs are assumptions. One trace (DeepSeek, one Hermes-shaped replay) carries every number; MiniMax and GLM traces, other prompt mixes, and the live store's prefetch were not run. The stop rule on small traces held: the other DeepSeek traces were not used.

### Consequence for the plan

N2's policy comparison should include LFU with decay and SLRU, not LFU without decay. H6 stays open and is registered for a second, longer trace before it counts. H9's threshold is likely too strict and is to be re-stated after N1 with the measured copy rate (a change recorded in the plan, not made silently).

## 6. Second trace: MiniMax-M3 (predictions registered 2026-10-18, before the run)

**Why this trace.** Section 5 left CUDA-H6 open pending a second long trace. No second long DeepSeek trace exists (the other four have 32 to 230 decode tokens); recording one needs the live runtime and Hamed's machine. Of the nine recorded traces only `routing-trace-minimax-long` (1,904 decode tokens, 57 MoE layers, 128 experts a layer = 7,296 experts, top-4, 228 requests a token) reaches the 1,000-token floor besides `deepseek-long`. The GLM traces (229 to 793 decode tokens) are smoke-grade and are not run. A different architecture tests whether the policy gaps are a property of the workload class or of one routing distribution; it does **not** replace the DeepSeek second trace, which stays an open item.

**Setup.** Same script with `--expert-bytes 22124953` (MiniMax slot image, 21.1 MiB) and the resident fractions 8, 12, 17, 25, 34 % of 7,296 experts (584, 876, 1,240, 1,824, 2,481 slots); policies LRU, SLRU and LFU with a halving every 16 tokens (LFU without decay is not rerun: section 5 closed it for DeepSeek and the arm costs most of the runtime); the Mac's shipped 68 GiB cache (3,300 slots, 45 %) as a reference point with no live figure attached. Consistency check against `cache_sim.simulate` and `simulate_policies.run` as before. Estimated cost 12 to 15 minutes of one Mac process (LFU cells about 60 s each, doubled by the check).

**Predictions** (all **predicted**; the memory of the repository is that routing concentration is DeepSeek > MiniMax > GLM, so MiniMax should miss more at the same fraction):

| id | quantity | prediction | falsifier |
|---|---|---|---|
| M-P1 | consistency with `cache_sim` and `simulate_policies`, all cells | holds exactly | any difference |
| M-P2 | LRU miss rate (misses over 228 requests) at 34 % / 17 % / 8 % | 15 to 30 % / 30 to 50 % / 45 to 70 % | a value outside its interval |
| M-P3 | relative SLRU reduction against LRU at 34 %, at 17 % | 1 to 5 % at each; ratio 17 % over 34 % between 0.8 and 3 | ratio outside 0.8 to 3 |
| M-P4 | CUDA-H6 premise (SLRU relative gain at 17 % at least twice that at 34 %, outside the noise) | passes with probability about 0.30 | scored by outcome |
| M-P5 | LFU with decay against SLRU at 17 % (relative reduction) | within 2 percentage points of SLRU | difference above 2 points |
| M-P6 | marginal misses a token per extra GiB (LRU), 8 to 17 % over 17 to 34 % | the 8 to 17 % slope at least twice the 17 to 34 % slope | ratio under 2 |

Stop rules as in section 4. A gap inside its bootstrap interval is a null.

### 6.1 Results (appended 2026-10-18, after the run; sections 1 to 6 above are unchanged)

Command (Mac, one Python process, about 14 minutes, no other runtime running):

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/cuda_offline_policy_table.py \
    benchmarks/results/routing-trace-minimax-long/trace.npz --expert-bytes 22124953 --policies lru,slru \
    --lfu-decay 16 --anchor-slots 3300 --determinism --out benchmarks/results/cuda-n0a/minimax-long.json
```

Trace `routing-trace-minimax-long`: 1,904 decode tokens, 57 MoE layers, 7,296 experts, top-4, 228 requests a token. **Measured** by replay (a trace-and-policy statement; no copy cost, no VRAM, no prefetch). The reference row is the Mac's shipped 68 GiB cache and carries no live figure.

| resident | experts GiB | LRU misses/token (miss rate) | SLRU saves | LFU, halved every 16 tokens, saves |
|---|---:|---|---|---|
| 8 % | 12.0 | 125.98 (55.3 %) | **-79.99** (-63.5 %), see below | +3.75 (+1.14 to +6.28), 3.0 % |
| 12 % | 18.1 | 103.80 (45.5 %) | +1.48 (-1.54 to +4.12), 1.4 %, **inside noise** | +4.13 (+1.63 to +6.48), 4.0 % |
| 17 % | 25.6 | 82.25 (36.1 %) | +3.21 (+1.74 to +4.66), 3.9 % | +3.29 (+1.30 to +5.12), 4.0 % |
| 25 % | 37.6 | 58.51 (25.7 %) | +2.32 (+1.17 to +3.42), 4.0 % | +2.89 (+1.58 to +4.11), 4.9 % |
| 34 % | 51.1 | 40.86 (17.9 %) | +1.79 (+1.18 to +2.43), 4.4 % | +1.71 (+1.13 to +2.33), 4.2 % |
| 3,300 slots (shipped 68 GiB) | 68.0 | 26.23 (11.5 %) | +0.75 (+0.42 to +1.13), 2.9 % | +0.77 (+0.44 to +1.14), 2.9 % |

### Scoring the predictions

| id | outcome |
|---|---|
| M-P1 consistency | **held for LRU and SLRU** (all 12 cells equal `cache_sim.simulate`); a second replay of every LRU and SLRU cell was identical. **LFU with decay is unverified on this trace as run:** the run printed `check False` for all six decay cells because the reference, `simulate_policies.run`, assumes 40 layers and this trace has 57, so it is no reference here (the script now records `None` in that case). What was checked afterwards: with the decay disabled the replay's LFU equals `cache_sim`'s LFU on this trace at 584 slots (310,139 misses both), and the decay hook is the same code path that matched `simulate_policies.run` on DeepSeek. The LFU-decay numbers above are therefore likely right but not independently reproduced |
| M-P2 LRU miss rate | **held in all three intervals**: 17.9 % (15 to 30), 36.1 % (30 to 50), 55.3 % (45 to 70) |
| M-P3 SLRU relative reduction | **held**: 4.4 % at 34 %, 3.9 % at 17 % (each 1 to 5), ratio 0.89 (0.8 to 3) |
| M-P4 CUDA-H6 premise | **fails**, as predicted with probability 0.70: the ratio is 0.89, not 2 |
| M-P5 LFU-decay against SLRU at 17 % | **held**: 4.0 % against 3.9 % |
| M-P6 slope ratio | **missed by 0.01**: 3.23 misses/token/GiB between 8 and 17 % over 1.62 between 17 and 34 % is 1.99, the falsifier was "under 2" |

### Findings (post hoc, labelled)

1. **The small-fraction effect of the DeepSeek LFU-with-decay lead does not replicate.** On DeepSeek, LFU-decay's gain was 3.9 % at 17 % and 1.2 % at 34 % (ratio 3.2). On MiniMax it is 4.0 % at 17 % and 4.2 % at 34 % (ratio 0.95). Over two models, a non-LRU policy saves roughly 3 to 5 % of LRU's misses almost regardless of the resident fraction, with one DeepSeek exception (LFU-decay at 34 %, 1.2 %). **CUDA-H6 is not established on either trace; the lead of section 5 is weakened.** The policy matters by a few percent of misses; the resident fraction matters by factors (LRU misses 126 to 41 over 12 to 51 GiB).
2. **SLRU as modelled collapses at 8 % on MiniMax**: 205.97 misses a token against LRU's 125.98, hit rate 9.7 % against 44.8 %, with the cell consistent with `cache_sim`. At 584 slots the prefill quota is 10 slots a layer; the cause was not investigated, so it is stated as an observed property of this store model at about ten slots a layer, not as an explanation. It matters for N2 only as a warning that a policy's behaviour at very small slot counts per layer must be measured, not extrapolated.
3. **The marginal value of resident memory is steeper at small fractions on MiniMax too**: 3.68, 2.87, 1.97 and 1.30 misses a token per extra GiB over 12 to 18, 18 to 26, 26 to 38 and 38 to 51 GiB. Across the two models the 8-17 % slope is 2.0 to 2.7 times the 17-34 % slope, which is consistent with the H9 reading of section 5.3 (derived, assumed copy rates).
4. MiniMax misses more than DeepSeek at the same fraction, as predicted from the recorded routing concentration: at 17 % resident the miss rate is 36.1 % against 21.6 % (51.85 of 240).

### What this does not establish

Two traces, two models, one decay constant, no GPU. A second long DeepSeek trace still does not exist; H6 is not closed, only unsupported on the data in hand. The GLM traces (229 to 793 decode tokens) are smoke-grade and were not run.
