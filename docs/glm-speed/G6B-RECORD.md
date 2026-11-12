# G6b: a predicted-set read ahead for short GLM prefill chunks

Written 2026-11-12 against runtime 0.84.0, before any run. HANDOFF section 18.276 when scored. `docs/GLM-SPEED-LANE.md` lists G6b as the follow-up of G6 (`G6-RECORD.md`, HANDOFF 18.274): keep the read ahead's overlap, drop its waste.

## 1. Baseline (measured, 18.274, internal SSD, 46 GiB)

On the G6 ladder after Hermes's 15.6k-token block, the whole-layer read ahead (`CACHALOT_GLM_SPECULATE_MIN_TOKENS`, chunks at or above the floor read every non-resident expert of the next layer) makes every prefill from 172 to 795 tokens take about 20 s. Reading only the routed set on demand (the floor above the chunk) took 15.7 s at 172 tokens, 16.9 at 239, 18.2 at 342, 19.3 at 499 and 22.0-22.5 at 795 (+7 % against the read ahead). Since 0.84.0 `serve-glm.sh` sets the floor to 512, so chunks under 512 tokens read on demand and chunks from 512 read the whole next layer. Prefills under 128 tokens never read ahead: 7.6 s at 35 tokens, 10.6 at 62, 13.8 at 116.

## 2. Pricing (derived, `docs/glm-speed/g6b_price.py`, before any build)

**Lever.** A prefill chunk below the floor scores the next MoE layer's router on this layer's MoE input for every row (the decode predictor of HANDOFF 18.35, `GlmModel._install_predictors`, already installed on every GLM MoE layer), takes the union of each row's top k, and hands it to the store as `speculate`, most-predicted experts first. The store reads them into transient slots while this layer computes and stops at its transient budget; the next layer's routed experts the union missed are read on demand. Outputs are bit-identical: only which reads start early changes.

**Instrument.** The weighted GLM trace of 0.62.10 (`benchmarks/results/routing-trace-glm-weights/trace.npz`) records predictions (top 5) for decode tokens only. Prefill rows have none, so a group of N consecutive decode tokens of one reply stands in for a prefill chunk of N rows (Hermes text through the same routers). Calibration: a stand-in group of 172 rows routes to 213 experts a layer, a real 217-token prefill to 224 (18.274); lever A's share of the whole-layer reads is 71-76 % for stand-in groups of 172-256 rows and 72-74 % for the real 217- and 304-token prefills. Residency is g6_price.py's: an LRU of 3,489 slots (46 GiB), warmed and run once over decode under budget 2, held fixed.

**Reads a chunk (summed over the 41 predicted layers), k = 5:**

| rows | routed a layer | whole layer | routed misses (A) | G6b ahead | G6b late | G6b waste | recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 88 | (none ahead) | 2,281 | 1,529 | 993 | 241 | 56.5 % |
| 64 | 151 | (none ahead) | 4,029 | 2,878 | 1,460 | 308 | 63.8 % |
| 172 | 213 | 8,409 | 5,946 | 4,690 | 1,524 | 268 | 74.4 % |
| 256 | 226 | 8,409 | 6,405 | 5,283 | 1,332 | 210 | 79.2 % |
| 384 | 260 | 8,409 | 7,292 | 6,255 | 1,229 | 192 | 83.1 % |

"Ahead" is the union's non-resident part (read while the previous layer computes), "late" the routed misses it missed (read on demand), "waste" predicted reads no row routes to. At k = 3 recall is 54-63 % on 172-384 rows; k above 5 cannot be priced from this trace. Groups longer than 400 rows do not exist in it (the longest reply is 400 tokens).

**Time (derived, a two-rate model fitted to G6).** The whole-layer arm's 8,409 reads in 19.9 s at 172 tokens give 2.36 ms a read when read ahead; the demand arm's 5,946 in 15.7 s give 2.65 ms a read on demand. G6b at 172 rows: 4,690 x 2.36 + 1,524 x 2.65 = 15.1 s against 15.7 (-4 %); 256 rows 16.0 against 17.0 (-6 %); 384 rows 18.0 against 19.3 (-7 %). Against the whole layer from 512 tokens the read count falls by an extrapolated 3-8 % with most reads still ahead: -2 to -7 %. Under 128 tokens the hidden latency (ahead x 0.29 ms) and the waste (x 2.36 ms) about cancel: flat. The model ignores compute, which grows with the chunk and is what made demand reads lose at 795 tokens; it is a planning number.

**Size, said plainly.** About 0.6-1.4 s off a 150-500-token prefill of 16-19 s, and perhaps 0.5-1.5 s off a 512-1,200-token one: 3-7 % of the prefill, 2-5 % of a 25-30 s turn. Below the single-digit ceiling the lane already named (`GLM-SPEED-LANE.md` section 4). It is cheap (no new reads path; the store's `speculate` argument and the predictor both exist) and bit-identical.

## 3. Hypothesis

A short GLM prefill on the internal drive pays a per-layer latency for every expert it reads on demand. Reading ahead the next layer's predicted union hides that latency for about three quarters of the routed misses at a waste of 3-4 % more reads, so it beats demand reads below 512 tokens and beats the whole-layer read ahead from 512 to about 1,200 tokens.

Alternative: the extra work (one router matmul a layer on every row, a larger host array per sync) or the waste costs as much as the hidden latency, and G6b is flat or slower than demand reads; or, above 512 tokens, the whole layer's deeper queue matters more than its 3-8 % extra reads, and G6b loses there.

## 4. Design

Knob: `CACHALOT_GLM_PREFILL_PREDICT_TOPK=k` (default 0, the shipped path), built in `glm/experts.py` with `tests/test_glm_prefill_predict.py` (7). It acts only on prefill chunks of more than one row and fewer than the floor, when the caller passed no `speculate` list (MiniMax passes its own and has no `predict`). The prediction is evaluated in the same sync as the layer's routing.

Arms (`docs/glm-speed/g6b_run_all.sh`): S = 0.84.0 shipped (floor 512, k 0); P = G6b everywhere below 1,200 (floor 1,200, k 5). ABBA: S1, P1, P2, S2, `benchmarks/settle.sh` between arms. Each arm a fresh `./serve-glm.sh --expert-budget-gib 46`, no disk snapshots, upscale knob on as live.

Workload (`g6_driver.py` with `G6_RUNGS=25,60,120,160,250,400,528,640,800,950`, W5): the live session's first request (15.6k-token block plus "Hi") prefilled cold, then ten one-token greedy requests that keep the block and prefill consecutive slices of `RESEARCH-DIRECTION.md`. The first seven slices are G6's (35-499 tokens); 640, 800 and 950 are new slices after them, the last kept under 1,200 (G6's 1,100 target prefilled 1,223 and tested nothing). The block's 2,048-token chunks and its 1,233-token last chunk read the whole layer in both arms.

Runtime estimate: four arms of about 7 minutes (40 s load, 180 s block, about 3 minutes of rungs) plus settling, about 32-36 minutes. Needs the sysctl at 88064 and no other runtime.

## 5. Predictions (registered before the run)

Per rung, P against S, the mean of the two pairs; the clean pair is read first if one arm's controls drift (v320 rule).

- **G6b-1.** 35, 62 and 116 tokens (P reads its union ahead, S reads nothing ahead): P between -15 % and +10 %.
- **G6b-2.** 172 and 239 tokens (S on demand): P faster by 0-12 %.
- **G6b-3.** 342 and 499 tokens (S on demand): P faster by 0-14 %.
- **G6b-4.** The 640 and 800 rungs (S whole layer, if they prefill 512-1,199 tokens): P between -12 % and +3 %.
- **G6b-5.** The 950 rung (S whole layer): P between -8 % and +5 %; a control if it prefills 1,200 or more.
- **G6b-6.** Identity: prefilled counts, reply hashes and decode misses a token equal in all four arms on every rung; the cold block within the S spread plus 3 %.
- **G6b-7.** Reproduction: S on rungs 1-7 (same text and same path as G6's D arms) within +-6 % of G6's D2.
- **Falsifier.** P slower than S on any of 172-499 tokens by more than the S1-S2 spread plus 3 %: the predicted read ahead costs more than it hides, and G6b closes below 512.

**Decision rule.** If G6b-2 and G6b-3 hold, offer Hamed `CACHALOT_GLM_PREFILL_PREDICT_TOPK=5` in `serve-glm.sh` (bit-identical; his call); if G6b-4 also holds, with the floor raised to 1,200. If only G6b-4 fails, the floor stays 512. If the falsifier fires, G6b closes and the lane's next lever is G4.

## 6. Results (measured 2026-11-12/13, commit 8535b23)

**Run.** Hamed: "Yes, run it". Sysctl 88064, no other runtime, swap 0 throughout, pressure level 1, apps open (Toggl, Claude, Chrome; a "Memory Cleaner" utility running; WindowServer at about 54 % CPU when sampled). `benchmarks/settle.sh` wanted 65 GiB available and saw 62 before the first arm, so `g6b_run_all.sh` passes `--timeout 120` (it timed out once, before S1, and passed in 20-25 s before the other arms). Arms 23:31-23:58, then one more arm D3 (floor 1,200, k 0: G6's D setting) at 00:00-00:06, added after scoring to test the control below. Raw: `benchmarks/results/glm-speed/g6b/`, guard output and samples in `benchmarks/results/guarded/glm-g6b-*`. The ten rungs prefilled 35, 62, 116, 172, 239, 342, 499, 624, 886 and 1,004 tokens.

| rung | tokens | S1 | P1 | P2 | S2 | D3 | G6 D2 (0.83.1) | P mean / S mean | P mean / D3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| block | 15,569 | 196.93 | 179.94 | 179.70 | 197.17 | 195.88 | 179.70 | -8.7 % | -8.2 % |
| 1 | 35 | 7.80 | 7.57 | 7.67 | 10.52 | 7.93 | 7.76 | -15.0 % | -3.9 % |
| 2 | 62 | 15.53 | 10.59 | 10.67 | 13.88 | 11.06 | 10.65 | -27.5 % | -3.9 % |
| 3 | 116 | 16.89 | 13.89 | 13.80 | 14.77 | 14.39 | 13.78 | -12.2 % | -3.8 % |
| 4 | 172 | 16.15 | 15.82 | 15.67 | 22.53 | 16.57 | 15.73 | -16.2 % | -5.0 % |
| 5 | 239 | 21.97 | 16.56 | 16.66 | 18.52 | 17.69 | 16.88 | -17.3 % | -6.1 % |
| 6 | 342 | 18.19 | 17.31 | 18.03 | 22.65 | 19.05 | 18.16 | -12.6 % | -7.2 % |
| 7 | 499 | 21.55 | 18.25 | 18.54 | 20.68 | 20.25 | 19.27 | -12.8 % | -9.2 % |
| 8 | 624 | 25.25 | 19.41 | 19.40 | 22.10 | 21.28 | | -17.7 % | -8.8 % |
| 9 | 886 | 24.82 | 23.34 | 20.05 | 22.23 | 22.46 | | -7.9 % | -3.4 % |
| 10 | 1,004 | 21.82 | 19.72 | 22.53 | 22.60 | 22.94 | | -5.0 % | -7.9 % |

Prefill seconds; "P mean / S mean" is the mean of the two pairs' P/S - 1, as registered.

**Identity (measured).** Prefilled counts, reply hashes and decode misses a token are equal in all five arms on every rung, and the block's decode line (60 misses, 62 % hits) too.

**The arms disagree with themselves.** P1 and P2 agree within 0.6-4.2 % on the block and rungs 1-8 (14-16 % on 886 and 1,004). S1 and S2 disagree by 4-40 % on rungs 1-8 (S2's 172-token rung read decode experts at 8.4 ms against 4.3). S1 and S2 do not reproduce G6's D2 on the same text and path (rungs 1-7: S1 +0 to +46 %, S2 +7 to +43 %).

**The control that failed.** Every block chunk is 1,228 or 2,048 rows, at or above both arms' floors, so the knob can act there only on the request's last chunk of 5 tokens, worth at most about 4 s even if every read were hidden (derived: 42 layers x at most 40 routed experts x 2.65 ms). Yet the block took 196-197 s in all three arms without the knob (S1, S2, D3, spread 0.6 %) and 179.7-179.9 s in both arms with it, while G6 (0.83.1) took 178.9-179.8 s in three of its four arms, P's setting not among them. The guard's samples show the same free memory, compressed memory (4.6-5.7 GiB) and page cache (12.8-13.6 GiB) during every block. Not explained. D3, the same setting and text as G6's D2, ran 2-5 % slower than D2 on rungs 1-7 and 9 % slower on the block.

**Scored.**
- G6b-1 (35-116 tokens, -15 to +10 %): missed, -15.0, -27.5 and -12.2 % (the 62-token rung outside).
- G6b-2 (172, 239 tokens, P faster by 0-12 %): missed high, -16.2 and -17.3 %.
- G6b-3 (342, 499 tokens, 0-14 %): held, -12.6 and -12.8 %.
- G6b-4 (624, 886 tokens, -12 to +3 %): split, -17.7 % outside, -7.9 % inside.
- G6b-5 (1,004 tokens, -8 to +5 %): held, -5.0 %.
- G6b-6 (identity; block within the S spread plus 3 %): identity held; the block failed, P -8.7 % against an S spread of 0.1 %.
- G6b-7 (S reproduces G6's D2 within +-6 %): failed.
- Falsifier (P slower than S at 172-499 tokens): not fired.

**Reading (hypothesis).** The registered comparison is void: its two controls (G6b-6's block, G6b-7) failed, and P beat S by about as much on a block the knob cannot touch as on the rungs it can. Two readings remain, and this run cannot choose between them:

- In-session, against D3: P is faster on every rung, -3.8 to -9.2 %. If the block's 16 s belongs to the knob through a path not understood, this is G6b's effect; if it is a state of the machine that happened to hold in the three non-P arms, part or all of it is not.
- Across sessions, against G6's D2 (same setting, same text, 2026-11-12 afternoon, clean controls): P is -2.4 to +0.5 % at 35-239 tokens, -2.6 % at 342 and -4.5 % at 499. That fits the price (section 2: about -4 to -7 % at 172-384 rows) at its low end, or nothing.

Either way the effect is a few percent of a short prefill, about 0.5-1.5 s, inside this machine's state-to-state variation on the day.

**Decision.** The knob stays off; nothing is offered as a default. The open question is the block: whether `CACHALOT_GLM_PREFILL_PREDICT_TOPK=5` changes a cold block's prefill when the code says it touches only a 5-token chunk. Next test: block-only arms alternating k 5 and k 0 at floor 1,200 (four arms of about 4 minutes), with the knob's reads counted. If the block gap follows the knob, the knob acts where the code says it does not, and the read path is the thing to understand before any speed claim. If it does not, the machine's state alternated with the arms, and G6b is worth at most the cross-session few percent.

**Limits.** One text, one residency path (block, then one-token replies), five arms on one night with applications open and the settle gate cut short, a stand-in price for prefill predictions, and an unexplained 16 s offset on the control.

**Kind.** Research: predicted, measured; the registered controls failed, so the comparison is void and the knob's effect is unresolved (a few percent at most on the clean cross-session baseline). Knob built, default off.
