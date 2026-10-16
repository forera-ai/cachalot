# L3 quantization sweep: DeepSeek expert bank bit width (plan, written 2026-10-16 before any run)

Status: plan only. Nothing has been run and nothing in `src/` or `benchmarks/` changes. Written against runtime 0.66.5 for Hamed's request "Plan the L3 quantization sweep (banks). Write the prediction and stop rule first and give me the time estimate before running anything." Charter item: `docs/RESEARCH-DIRECTION.md` section 5, L3 item 5 (bank bit width against bytes per token and quality). Workload classes: W1 (short interactive decode, default class) for speed; W4 (code) is where quality shows first.

Tags: every number below is **recorded** (from the cited file), **derived** (arithmetic on recorded constants) or **prediction**.

## 1. What already exists (read before building anything)

- Banks on disk today: the FP4 checkpoint (17.93 MiB per expert, X10Pro, the only copy) and the 2-bit g128 bank (9.49 MiB per expert, 142 GiB, internal since 2026-10-03, X10Pro copy as fallback). No 3-bit or 4-bit bank exists (recorded, `PROJECT-MAP.md`, HANDOFF section 3 of the September records).
- The axis was already walked in September 2026: 2-bit g64, 3-bit g64 and g128, and a searched activation-weighted 3-bit bank were built, gated and deleted (HANDOFF around lines 1836-1890 and 3520). Recorded outcome: the quantized banks are "not meaningfully different from each other" on code; the genuine +5.3 points of top-1 from 3-bit did not translate into usable code improvement; the 3-bit bank tied the oQ3e download. So this sweep is a re-measurement as a curve on today's constants (drive, budget, sysctl, prefetch), not a discovery. Those September constants are stale (charter rule 4): the old bank sizes and hit rates predate the 2.6 GB/s knee model and the 48 GiB cap.
- Instruments that exist: `benchmarks/build_affine_bank.py` (writes 2/3-bit g64/g128 banks from FP4; `CACHALOT_EXPERT_BANK=<out>` is the whole switch), `expert_requant_error.py` (weight and output error per layer without running the model), `quant_fit_screen.py`, `nll_expert_precision.py` (cheap quality gate), `code_validity.py` (gate above 2 bits), `cache_sim.py` and `whatif.py` (misses and token time), `pareto.py`.

## 2. Design: three steps, cheapest first

**Step A, offline (no model, about 30 minutes of work).** Compute bytes per token and predicted token time for formats 2-bit g128 (9.95 MB), 3-bit g128 (14.38 MB, 205.6 GiB bank) and 4-bit (size from `build_affine_bank.py`; to be confirmed, about 18.8 MB for g128), at the shipped 48 GiB budget (fewer experts fit as they grow) and at equal-capacity budgets, using `cache_sim.py` on the saved 0.55.0 weighted trace and the `whatif.py` DeepSeek line (`WI-DS`). Write predictions to this file first (section 3).

**Step B, quality screen (small machine use, estimate 10-20 minutes, not measured).** `expert_requant_error.py --experts-per-layer 2 --probes 8` at 2, 3 and 4 bits g128 against the FP4 weights. It reads about 80 experts from the X10Pro FP4 checkpoint and quantizes in memory. No server, no bank written. This gives the quality axis of the curve without the 205 GiB bank.

**Step C, live (needs Hamed, expensive, only if A and B say it is worth it).** Build a 3-bit g128 bank and run a W1 server arm at the shipped budget. Problem found while planning: the bank is 205.6 GiB; the internal disk has 134 GiB free and already holds the 142 GiB 2-bit bank, so it cannot live there. On the X10Pro the expert drive is 1 GB/s, so the arm would measure the drive and not the quantization (section 3, last prediction). X10Pro free space was not verified (`df` printed nothing for it). A fair live arm needs one of: deleting the internal 2-bit bank (his explicit yes, and it reverses the 2026-10-03 move), a partial-layer bank (changes the workload), or the read throttle held at the internal rate (cannot exceed the drive's real 1 GB/s). So Step C is a decision for Hamed, not part of this plan's default.

## 3. Predictions (hypotheses, written before the run)

Baseline (recorded): 2-bit g128 at 48 GiB, W1, exact, prediction on: 118.6 ms a token, 23.5 misses a token (`ST-BW-CURVE`, `WI-DS`); 1 GB/s drive 445.7 ms (read-bound, 453 MB a token). Run-to-run drift 1.5 % sd, bound about 6 % (§18.147, §18.148).

1. **Capacity.** At a fixed 48 GiB budget a 3-bit g128 expert is 1.445 times larger, so the cache holds 0.69 times the experts, equal to a 33 GiB budget of the 2-bit bank. Predicted misses a token on W1: 33 to 38 (centre 35). Falsifier: below 30 or above 42 in `cache_sim.py`.
2. **Bytes.** Bytes read a token: 35 x 14.38 MB = 503 MB against 234 MB (23.5 x 9.95), a ratio of 2.1 (interval 1.8 to 2.4).
3. **Token time, internal drive (6.8 GB/s).** `WI-DS` gives 78 + 35 x 1.73 = 139 ms against 118.6: +17 % (interval +10 to +25 %). Falsifier: under +6 % (inside the drift bound) or over +30 %.
4. **Token time, X10Pro (1 GB/s).** The byte-bound term scales by (35 / 23.5) x 1.445 = 2.15: about 950 ms against 445.7 (interval 800 to 1,050). This is the number that makes a live 3-bit arm on the X10Pro meaningless as a quantization measurement.
5. **Quality screen.** Output error relative to FP4 (`y-err`) falls about as 2^-bits (uniform quantizer, hypothesis): 3-bit / 2-bit in 0.40 to 0.60, 4-bit / 2-bit in 0.18 to 0.32. Falsifier: 3-bit / 2-bit above 0.70.
6. **Value.** Prediction of the verdict: no format above 2-bit g128 pays on this machine, because the token cost rises 17 % on the internal drive and the September record found no usable code-quality gain; the curve's worth is its shape (cost per bit) for the charter, not a new default.

## 4. Stop rules

- Step A: stop and report if the predicted token change for 3-bit is inside the 6 % drift bound (then a live arm could not resolve it).
- Step B: stop if the 3-bit / 2-bit `y-err` ratio is above 0.70 (the 3-bit format buys too little to pursue) or after one rerun disagrees with the first by more than 10 %.
- Step C: do not start without Hamed's decision on where the bank lives. If started, stop at the first void arm; a run where memory pressure, the wired-limit governor or the drive rate differs between arms is void (use `guarded_run.sh`, `settle.sh --budget-gib`, one process).
- Anywhere: no default changes. A format above 2-bit becoming a default would be an output-changing decision and is Hamed's call.

## 5. Time estimate (planning figures, not measured)

| step | machine | wall time |
|---|---|---|
| A offline | none | about 30 minutes |
| B quality screen | brief, small memory | about 10 to 20 minutes |
| C bank build for 3-bit g128 | X10Pro read of about 275 GiB of FP4 experts at about 1 GB/s | at least 75 minutes, probably 1.5 to 2 hours, plus 205 GiB written (the old half-hour figure was from the internal drive) |
| C live arms | one server, about 25 minutes an arm, ABBA = 4 arms | about 2 hours, X10Pro-bound |

Recommended scope for now: A and B, about 45 to 60 minutes total, no sampler, no sudo, nothing on the drives written. Ask before C.

## 6. Open items before running

- X10Pro free space (unverified).
- Whether B may start while no server runs (it needs about a few GiB of memory; the one-runtime rule still applies).
- Release shape: planning doc and later results go in one patch release per `RELEASE.md`; nothing is committed yet.

## 7. Results of steps A and B (2026-10-16, same session; nothing written to the drives)

Kind: research by the charter's definition (predictions above were written before the runs), offline and derived for speed, measured for the quality screen. Step C was not run.

**Step A** (`cache_sim.py` on `benchmarks/results/routing-trace-deepseek-long/trace.npz`, 2,028 decode tokens of the 22k Hermes conversation, LRU, floor 80 ms, 48 GiB; expert sizes 9.95, 14.38 and 18.80 MB; the 4-bit size follows from the builder's 4.42 MB per bit and equals the FP4 expert, 17.93 MiB):

| bank | slots at 48 GiB | decode hit | misses a token | MB read a token | internal ms (constant 2.0 ms a miss) |
|---|---:|---:|---:|---:|---:|
| 2-bit g128 | 5,178 | 89.6 % | 25.0 | 249 | 130 |
| 3-bit g128 | 3,584 | 84.1 % | 38.1 | 548 | 156 |
| 4-bit g128 | 2,741 | 79.4 % | 49.5 | 931 | 179 |

The simulator's 2-bit row reproduces the live 25.3 misses (recorded, §18.52). The simulator charges the same 2.0 ms a miss to every bank, which understates larger experts: a miss is mostly a read of the expert (about 1.5 of the 2.0 ms at 9.95 MB over 6.5 GB/s, by arithmetic on recorded constants, derived). Scaling that read share with size gives (derived) 3-bit about 80 + 38.1 x 2.67 = 182 ms (+40 %) and 4-bit about 80 + 49.5 x 3.34 = 245 ms (+88 %); the constant-cost table above is a lower bound (+20 % and +38 %).

**Step B** (`expert_requant_error.py --experts-per-layer 2 --probes 8 --candidates 2/128,3/128,4/128`, 40 layers, 22.1 s wall, FP4 checkpoint as reference; raw rows in `benchmarks/results/expert_requant_error.json`, a local ignored folder): mean output error y-err 2-bit 0.7104, 3-bit 0.3988, 4-bit 0.1846; weight error 0.4457, 0.2401, 0.1074. Worst layer is 27 (2-bit and 3-bit) and 29 (4-bit), within 5 % of the mean in every format. The searched 2-bit fit scores 0.5955 (2/128) and 0.5447 (2/64) at no extra bytes (and 2/64 min-max 0.9121), so a better fit at the shipped width buys about as much error as half a bit.

**Predictions against results.**

| # | prediction | result | verdict |
|---|---|---|---|
| 1 | 33-38 misses a token at 48 GiB for 3-bit | 38.1 (22k Hermes trace, not the short-prompt W1 set the baseline used) | missed narrowly high; falsifier (above 42) did not fire; workload differs from the prediction's basis |
| 2 | bytes ratio 1.8-2.4 | 2.20 | held |
| 3 | internal token +10 to +25 % | +20 % constant-cost bound, +40 % with size-scaled misses (derived) | missed high; the registered falsifier "over +30 %" fires on the size-scaled estimate. The cause is that I took the miss cost as size-independent |
| 4 | X10Pro token about 800-1,050 ms | not measured; derived 640 (cost scaled) to 1,060 ms (byte-bound term), against 335 ms for the 2-bit bank in the simulator | untested; a live X10Pro arm would measure the drive |
| 5 | y-err ratio 3-bit/2-bit 0.40-0.60; 4-bit/2-bit 0.18-0.32 | 0.561 and 0.260 | held |
| 6 | no format above 2-bit g128 pays | supported on cost (+40 % to +88 % token for 44 % and 74 % less output error); the code-quality side is not re-measured | hypothesis stands |

**Stop rules.** Step A: the predicted change (+20 to +46 %) is far above the 6 % drift bound, so a live arm could resolve it; no stop. Step B: the 3-bit/2-bit ratio 0.561 is below 0.70, so the stop rule that would close the idea does not fire; the 3-bit bank genuinely reduces output error by 44 %.

**Explanation and generalization (hypotheses).** On a read-bound token each bit costs about 4.4 MB an expert, which at a fixed byte budget both enlarges every miss and multiplies the misses: 3-bit pays 1.52 times the misses at 1.45 times the size (2.2 times the bytes). This is a property of experts with this size and a cache this size, not of DeepSeek: a model with a working set well inside the budget would pay only the larger miss. It is a systems statement on this Mac (internal NVMe 6.8 GB/s, 48 GiB cap), not a datacenter claim.

**Bottleneck after.** None moved (no change shipped). The curve says the expert drive and the cache capacity are what price a bit. Open: step C needs a decision on where a 205 GiB bank can live (section 2) and the September finding that 3-bit did not improve code; a better 2-bit fit (searched, activation-weighted) is the cheaper quality lever that costs no bytes and is already screened above, but it is an output-changing bank swap and Hamed's call.
