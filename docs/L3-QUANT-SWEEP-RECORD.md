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

## 8. Step C: the live arms (record opened 2026-10-16, before any server arm ran)

Hamed's decision of 2026-10-16: delete the internal 2-bit bank, build the 3-bit bank there, run the follow-up arms. Done so far, with no server arm run: the internal 2-bit bank was deleted after its names and sizes matched the X10Pro copy and two shards compared byte for byte (`cmp`); a 3-bit g128 bank was built with the same fit as the shipped 2-bit bank (`--fit mlx`, so bits are the only variable) at `~/DeepSeek-V4.1-Flash-q3g128`: 205.7 GiB in 7 minutes, 8 sampled experts byte-identical to the source (the 1.5 to 2 hour estimate of section 5 was wrong: the FP4 source reads sequentially far faster than the 1 GB/s expert-read plateau). Disk free after: about 70 GiB. Consequence for the design: the internal 2-bit baseline no longer exists; the recorded 118.6 ms (0.61.0, floor 78 ms) predates the 0.66.0 FP8 default (floor about 74 ms).

**Arms (not yet run).** Driver and gates as in HANDOFF 18.76/18.77 (`floor-server-0.60.10/driver.py`: warm-up, floor prompt x3, 12 mixed prompts, floor x2; greedy, 120 tokens; a fresh `./serve.sh` under `guarded_run.sh --budget-gib 48 --force`, `CACHALOT_DECODE_MISS_BUDGET=off`, `CACHALOT_SNAPSHOT_DIR=`, manifest first, `settle.sh` between arms). Workload class W1.
- **Arm A:** 3-bit g128 bank, internal drive.
- **Arm B:** 2-bit bank, X10Pro copy (same-day floor and the misses a token of the 2-bit bank; its drive cost is the recorded 390 ms regime).
- **Arm C (optional, needs a yes to write 206 GiB to the X10Pro):** 3-bit on the X10Pro, the paired test of prediction 4.

**Predictions for arm A (internal 3-bit, mixed W1, exact, prediction on).** (a) Misses a token 32 to 40 (the simulator's 38.1 is on the 22k trace, where it also gives 25.0 for the 2-bit bank against 23.5 live on this driver, so 38.1 x 23.5 / 25.0 = 35.8). (b) All-resident token 72 to 80 ms (FP8 default; independent of the bank). (c) Mixed token 150 to 185 ms, centre 170 (floor 76 + 36 misses x 2.6 ms, the size-scaled miss cost of section 7), that is +30 to +55 % against the recorded 118.6 and about +40 % against today's expected 2-bit figure of about 115. Falsifier: under 135 ms (the cost is the constant-cost bound, +15 %) or over 200 ms. (d) Bytes read a token 2.0 to 2.4 times the 2-bit arm's.
**Predictions for arm B.** Misses a token 22 to 25; all-resident 72 to 80 ms; mixed token 330 to 410 ms (the recorded 390.1 at 1.0 GB/s).
**Stop rule.** A void arm (pressure above 1, swap growth, drive or sysctl differing between arms, the quiet gate failing, or a guardian kill) is rerun once and then reported void; no default changes whatever the result; the 3-bit bank is not a candidate default (output-changing, and September's code gate found no usable gain).
**Cleanup after the arms.** Delete the 3-bit bank, copy the X10Pro 2-bit bank back to `~/`, compare it, and `touch .cachalot-verified` so `serve.sh` stops falling back to the X10Pro (until then it serves at about 390 ms a token).
**Time estimate (not measured).** Arm A about 15 to 20 minutes, arm B about 20 to 25, settle gaps and manifests about 10: about 50 to 60 minutes for A and B; arm C adds a 206 GiB write (about 10 to 15 minutes) and one 25-minute arm; restoring the 2-bit bank about 10 to 15 minutes.

## 9. Step C results (2026-10-16; arms A, B and C ran 10:51 to 12:34)

Kind: research (predictions in section 8 were written before the first server arm). Workload W1, exact (`CACHALOT_DECODE_MISS_BUDGET=off`), prediction on, 48 GiB, FP8 decode default on, one fresh `./serve.sh` an arm under `guarded_run.sh --force`, manifests `9aca0cbcbd765390` (A), `1f7ec1ad9bb8e700` (B), `def2b31e7e0f7add` (C), driver and raw rows in `benchmarks/results/quant-sweep-0.66.7/`, analysis `analyze.py`. The quiet gate (load under 2.5) held for about 50 minutes at the start because Hamed's Hermes Desktop was open; no arm ran with it open. Each arm: pressure 1, swap 0, 2,040 generated tokens (1,440 in the mixed set).

| arm | bank, drive | mixed ms a token | all-resident repeats | misses a token | MB read a token | predicted loads used |
|---|---|---|---|---|---|---|
| A | 3-bit g128, internal | **164.8** | 82.2, 79.5 | 34.4 | 907 | 32 % |
| B | 2-bit g128, X10Pro | **401.6** | 94.5, 89.0 | 25.7 | 440 | 35 % |
| C | 3-bit g128, X10Pro (pair of B) | **747.7** | 88.1, 88.0 | 34.3 | 883 | 33 % |

Misses and bytes are whole-run counts after the warm-up request (prefill included) over generated tokens.

**Against the predictions.** Arm A: misses 32-40 held (34.4); mixed token 150-185 ms held (164.8; falsifier under 135 or over 200 did not fire; centre 170 missed by 5 ms); bytes ratio 2.0-2.4 held at its low edge (907 / 440 = 2.06; C / B = 2.00); all-resident 72-80 ms missed narrowly high (79.5-82.2). Arm B: misses 22-25 missed narrowly high (25.7); mixed 330-410 held at its top (401.6, against the recorded 390.1 for the same drive, +3 %); all-resident 72-80 missed high (89-95: the X10Pro arms carry speculative reads, as in HANDOFF 18.77). Earlier prediction 4 (X10Pro token for 3-bit 800-1,050 ms, 2.15 times the 2-bit): **missed low**, 747.7 ms (1.86 times); the byte ratio is 2.0, not 2.15, because the 3-bit bank missed 1.34 times as often as the 2-bit bank, not the simulator's 1.52. Prediction 6 (nothing above 2-bit pays): the cost side is now measured; the code-quality side is still not re-measured.

**Derived.** (1) Cost of a miss scales with the bytes of the expert: (164.8 - 80) / 34.4 = 2.47 ms a miss against 1.73 for the 2-bit bank, 1.43 times for 1.445 times the bytes; the size-scaled estimate of section 7 (2.6-2.67 ms) was right in kind and 5-8 % high. (2) Internal-drive 3-bit penalty: against the 2-bit bank's expected 80 + 25.7 x 1.73 = 124 ms the 3-bit bank is +33 %, against the recorded 118.6 ms it is +39 % (not paired: the internal 2-bit bank was deleted for the arm, so there is no same-day internal 2-bit run). (3) The X10Pro arms are pipe-bound: 440 MB / 401.6 ms = 1.10 GB/s and 883 MB / 747.7 ms = 1.18 GB/s effective, so the token follows the bytes (ratio 1.86 for 2.0 times the bytes, the 80 ms of compute not scaling). (4) Output error per cost: 3-bit buys 44 % less y-err (0.399 against 0.710) for +33 % token time internal and +86 % on the 1 GB/s drive.

**Explanation (hypothesis).** A bit adds 4.4 MB to every expert. At a fixed byte budget the cache holds 0.69 times as many experts, which raises misses 1.34 times (less than the simulator's 1.52: routing locality absorbs part of the shrink), and each miss reads 1.445 times the bytes, so a token reads 2.0 times the bytes. Where reads are on the critical path the token time follows the bytes, which is why the penalty is +33 % at 6.8 GB/s (reads partly hidden) and +86 % at 1 GB/s (reads the whole token). Generalization: an architecture-level statement for experts of this size and a cache of this size; a model whose working set fits the budget would pay only the larger miss.

**Limits.** One run an arm, no replicate; the internal 2-bit baseline is derived, not measured the same day (its bank was deleted for the arm and restored afterwards); the first all-resident repeat after the warm-up (floor0, 120-129 ms in several arms) reads cold and is not used; quality was not re-measured at the code level; the page cache held parts of the 3-bit bank copy during arm C (the copy ended 15 minutes earlier), which would flatter C slightly.

**Stop rules.** None fired. Step C is done; a code-validity gate on the 3-bit bank (`code_validity.py`) is the only open item on this axis and September's finding stands unless it is rerun.

**Cleanup done.** The 3-bit bank was deleted from both drives; the 2-bit bank was copied back to `~/DeepSeek-V4.1-Flash-q2g128`, names and sizes matched, three shards compared byte for byte, `.cachalot-verified` touched, so `serve.sh` is back on the internal bank.
