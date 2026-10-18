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

## 5. Results (appended 2026-10-18, after the run; sections 1 to 4 above are unchanged)

Command (Mac, one Python process, 12 seconds, no runtime running):

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/slru_collapse_probe.py --out benchmarks/results/cuda-n0a/slru-collapse.json
```

All numbers **measured** by replay (a statement about the SLRU store model on these traces). Collapse means SLRU misses a token more than 1.10 times LRU's in the same cell.

### 5.1 Internals of the collapsed cell (MiniMax, 584 slots, f = 0.8)

| quantity | measured | prediction |
|---|---|---|
| misses a token | 205.97 | S0 |
| protected segment, fraction of its cap (467) | at least 0.951 after token 200, at least 0.964 in the last half | S1a: at least 0.95 |
| median life of a decode-inserted key before eviction | 138 requests = 0.61 token | S1b: under 1 token |
| decode-inserted keys evicted with no second hit | 99.9 % (381,609 evictions) | S1b: at least 80 % |
| decode hits served from protected | 97.7 % | S1c: at least 85 % |
| promotions a token after token 200 | 0.27 | S1c: under 10 |

### 5.2 Sweeps

| trace | slots | f | probation slots | LRU misses/token | SLRU misses/token | ratio | collapse |
|---|---:|---:|---:|---:|---:|---:|---|
| MiniMax | 584 | 0.50 | 292 | 125.98 | 123.64 | 0.98 | no |
| MiniMax | 584 | 0.65 | 204 | 125.98 | 123.78 | 0.98 | no |
| MiniMax | 584 | 0.75 | 146 | 125.98 | 188.71 | 1.50 | **yes** |
| MiniMax | 584 | 0.80 | 117 | 125.98 | 205.97 | 1.63 | yes |
| MiniMax | 584 | 0.90 | 58 | 125.98 | 200.45 | 1.59 | yes |
| MiniMax | 876 | 0.80 | 175 | 103.80 | 102.32 | 0.99 | no |
| MiniMax | 876 | 0.90 | 88 | 103.80 | 184.27 | 1.78 | yes |
| DeepSeek | 461 | 0.50 | 230 | 144.49 | 141.31 | 0.98 | no |
| DeepSeek | 461 | 0.80 | 92 | 144.49 | 223.45 | 1.55 | yes |

Exploratory extension at 584 slots (added after seeing the table above, not registered; same trace): f = 0.66 -> 123.74 (ratio 0.98, no); 0.68 -> 125.69 (1.00, no); **0.70 -> 160.22 (1.27, yes)**; 0.72 -> 188.17 (1.49); 0.74 -> 187.24 (1.49). In terms of probation capacity over the LRU misses a token: no collapse at 1.58 and 1.48 times, collapse at 1.39, 1.30, 1.21 times and below.

### 5.3 Scoring the predictions

| id | outcome |
|---|---|
| S0 | **held.** The default-preserving change reproduces 205.97 exactly; the MiniMax and DeepSeek 17 % SLRU cells also reproduce N0a (79.04, 49.94) |
| S1a | **held, narrowly** (minimum 0.951 against the 0.95 line after token 200) |
| S1b | **held** (0.61 token; 99.9 % never re-hit) |
| S1c | **held** (97.7 %; 0.27 promotions a token) |
| S2 | **partly missed.** f = 0.5 and 0.65 do not collapse (held) and f = 0.8 reproduces 205.97 (held), but **f = 0.75 collapses** (ratio 1.50) where I predicted no collapse for f at or below 0.75, and **f = 0.9 (200.45) is not worse than f = 0.8 (205.97)**: the registered second falsifier fired. The collapse saturates near 1.6 times LRU rather than growing with f |
| S3 | **held** (876 slots: f = 0.8 on par, f = 0.9 collapses) |
| S4 | **held** (DeepSeek at 3 % resident: f = 0.8 collapses at 1.55, f = 0.5 does not) |
| S5 | **held, with a correction.** The rule "probation below the LRU misses a token" misclassifies exactly one cell of nine (584, f = 0.75, probation 146 against 126 misses; the allowance was one). The data put the cliff higher: collapse from about 1.4 times the misses a token downward, none from 1.48 times upward |

### 5.4 Conclusion

**H-probation is supported.** Collapse depends on the protected fraction exactly where the probation arithmetic says it should, on two models with different layer counts and expert counts, and it appears on DeepSeek at a matched ratio, which excludes a MiniMax-specific cause. The prefill quota (ten slots a layer) does not depend on f, yet f = 0.5 at the same slot count does not collapse, which excludes it. The mechanism in the instrumented cell is the one hypothesised: a key lives 0.6 token in probation, 99.9 % of decode-inserted keys die unseen, promotions have stopped (0.27 a token), the protected segment is full of the early residents and serves 97.7 % of the few hits that remain. The store behaves as a frozen set plus a probation window too short to be hit twice.

**Where the hypothesis was too crisp.** The cliff is at about 1.4 to 1.5 times the misses a token inserted, not at 1.0 times; the registered threshold would have called f = 0.75 safe. The reason was not tested (a plausible one: the misses a token is an average, a key is re-hit only after more than one token, and a probation window of one token's inserts is not enough). The saturation (f = 0.9 no worse than f = 0.8) fits a frozen set but was not predicted. This is a property of the SLRU model in `simulate_policies.py` with its fixed promotion rule (second hit) on these two traces; other promotion rules and a protected fraction tuned per cell were not run.

**For N2 and the plan.** A segmented policy at small resident fractions must size probation at roughly 1.5 times the experts a token inserts or more, which at about 56 % misses means a protected fraction of 0.65 or lower on MiniMax at 8 %, and the safe fraction depends on the cell's miss rate, so it must be set from the measured miss rate, not fixed at 0.8. Because the non-LRU gains seen in N0a are only 3 to 5 % of misses, a policy that needs this tuning has little to offer over LRU at these fractions. Nothing in the shipped LRU store is affected.
