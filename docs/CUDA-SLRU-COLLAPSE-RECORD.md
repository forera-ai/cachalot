# CUDA lane N0a follow-up: why the SLRU model collapses at 8 % resident on MiniMax (record)

Written 2026-10-18 against runtime 0.66.13. Mac only, offline, no GPU. Labels follow `docs/CUDA-RESEARCH-PLAN.md` section 0. This follows `docs/CUDA-N0A-POLICY-RECORD.md` section 6.1, finding 2.

**Status at the time of writing: predictions only.** Sections 1 to 3 were committed before any run of the probe; section 4 (results) is appended after and nothing above it is edited.

## 1. The observation and what is, and is not, being explained

**Recorded (measured, N0a record 6.1).** On `routing-trace-minimax-long` (1,904 decode tokens, 57 layers, top-4, 228 requests a token) the SLRU store model of `benchmarks/simulate_policies.py` at 584 slots (8 % of 7,296 experts) misses 205.97 times a token against LRU's 125.98 (hit rate 9.7 % against 44.8 %), and at 876 slots (12 %) it is on par (102.32 against 103.80). The cell was consistent with `cache_sim.simulate`, so this is a property of the modelled policy, not a replay defect.

**Scope.** SLRU here is a model in a benchmark file. The shipped store is LRU; nothing in `src/` changes. The aim is to understand a policy failure mode that N2 must not walk into at small slot counts.

**Model under test** (`Store`, policy `slru`): a probation segment and a protected segment; a miss is inserted into probation; a second hit promotes a key to protected; protected is capped at 80 % of the slots and its overflow is demoted to the front (eviction end) of probation; eviction takes the probation front first and the protected front only when probation is empty. At 584 slots the protected cap is 467 and probation is left with at most 117 slots.

## 2. Hypothesis

**H-probation (mechanism).** The collapse is a capacity starvation of the probation segment. A decode token inserts about as many new keys as it misses (LRU: about 126 at 584 slots, 104 at 876). If the probation capacity `(1 - f) x slots` (f = the protected fraction, 0.8 shipped) is smaller than the misses a token inserts, a new key is evicted from probation before its next use, so it can never earn the second hit that promotes it; the protected segment then fills with the early residents, stops changing, and the store behaves as a frozen set plus a probation window too short to be useful. LRU adapts; this SLRU does not. A key corollary is a threshold rule: collapse (SLRU misses more than 1.10 times LRU's) if and only if `(1 - f) x slots` is below the LRU misses a token of that cell.

Alternative explanations the experiments can separate from it: a defect in the demotion or eviction code at small caps (would not depend on `f` the way the rule says); the prefill quota of ten slots a layer (would not depend on `f`); an effect of MiniMax's layer count rather than the slot count (would not reproduce on DeepSeek at a matched probation-to-miss ratio).

## 3. Experiments and predictions (registered before any run)

Instrument: a new `benchmarks/slru_collapse_probe.py` that replays a trace exactly as the N0a table does and additionally records, for an SLRU run, the protected and probation sizes over time, the segment each decode hit is served from, promotions, and for every decode-inserted key its lifetime (in requests) before eviction and whether it was hit again. `Store` gains an optional `protected_fraction` argument whose default (0.8) leaves every existing number unchanged; this is verified first (SLRU at 584 slots must still read 205.97 misses a token).

Definition: **collapse** means SLRU misses a token more than 1.10 times LRU's in the same cell. All predictions below are **predicted**.

| id | experiment | prediction | falsifier |
|---|---|---|---|
| S0 | default-preserving change | SLRU at 584 slots, f = 0.8: exactly 205.97 misses a token (the N0a value), and every other N0a SLRU cell unchanged | any difference |
| S1a | internals, MiniMax, 584 slots, f = 0.8 | the protected segment holds at least 95 % of its cap (at least 444 of 467) from token 200 on and in the last half of the trace | below 95 % |
| S1b | same | the median lifetime of a decode-inserted key before eviction is under one token (under 228 requests) and at least 80 % of decode-inserted keys are evicted with no re-hit | median at or above one token, or fewer than 80 % never re-hit |
| S1c | same | at least 85 % of decode hits are served from the protected segment, and promotions after token 200 average under 10 a token | below 85 %, or 10 or more promotions a token |
| S2 | sweep f in {0.5, 0.65, 0.75, 0.8, 0.9} at 584 slots, MiniMax | f at or below 0.75: no collapse (SLRU within 8 % of 125.98, either side); f = 0.8: collapse, 205.97; f = 0.9: collapse and worse than f = 0.8 | f = 0.5 collapses (mechanism wrong), or f = 0.9 is not worse than f = 0.8 |
| S3 | f in {0.8, 0.9} at 876 slots, MiniMax | f = 0.8: no collapse (102.32 against 103.80); f = 0.9: collapse | f = 0.8 collapses or f = 0.9 does not |
| S4 (out of sample) | DeepSeek `routing-trace-deepseek-long` at 3 % resident (461 slots), f in {0.5, 0.8} | f = 0.8 collapses; f = 0.5 does not (within 8 % of LRU). The rule predicts this because LRU misses about 150 a token or more there while the probation capacity at f = 0.8 is 92 | f = 0.8 does not collapse, or f = 0.5 does |
| S5 | the threshold rule across all collapse/no-collapse cells above | `(1 - f) x slots` below the LRU misses a token separates every collapsing cell from every non-collapsing one, with at most one misclassified cell | two or more misclassified cells |

Predicted outcome if H-probation holds: a one-line statement for N2, "a segmented policy needs a probation segment at least as large as the keys a token inserts, which at small resident fractions means a protected fraction well below 0.8".

## 4. Stop rules and cost

* If S0 fails, results are void until the change is fixed.
* A gap inside a bootstrap interval is not claimed; sweep cells are single deterministic replays, so the 10 % and 8 % bands above are the decision rules, stated in advance.
* Cost: Mac CPU, SLRU and LRU cells take about 2 s each on these traces; the whole sweep is a few minutes. No long test, no runtime process, no GPU.
