# G6: what a GLM short prefill reads, and the two levers that would read less

Written 2026-11-12 against runtime 0.83.0, before any run. HANDOFF section 18.274 when scored. `docs/GLM-SPEED-LANE.md` lists G6 as "a prefill miss budget for short chunks, MiniMax's 0.42.0"; v319's first job asked for its price from a trace, per affected row (18.260), before any port.

## 1. Baseline (measured, 18.272, internal SSD, 46 GiB)

G3's replay: a 160-token prefill 19.39 s of a 26 s turn; 528 tokens 18.78 s; 1,252 and 1,322 tokens (images) 26.2 and 22.5 s. A short prefill costs about the same whatever its length between 160 and 528 tokens.

## 2. Pricing (derived, `docs/glm-speed/g6_price.py`, before any build)

Instrument: the weighted GLM routing trace of 0.62.10 (`benchmarks/results/routing-trace-glm-weights/trace.npz`, six requests of the 0.61.12 Hermes session: prefills of 5, 24, 19, 2,738, 217 and 304 tokens) plus the two unweighted replays of the same requests. Residency: one LRU of 3,489 slots (46 GiB at 13.5 MiB), decode under budget 2, a warm-set start, prefill not changing residency (the store's scan path).

**What the shipped path reads.** From 128 tokens (`CACHALOT_GLM_SPECULATE_MIN_TOKENS`, chosen in 17.1 on the comment "at 128 tokens top-8 of 288 reaches ~97 % of them"), each layer queues every non-resident expert of the next layer. A chunk of 128 to 2,048 tokens therefore reads about 8,600 experts, 42 x (288 - 83 resident), whatever its length. That matches G3's flat 19 s: 8,600 reads x about 2.25 ms is 19.4 s (derived; 2.25 ms is the fit of this one point, against 2.9 ms an expert measured single-file in G3).

**The 97 % premise does not hold on Hermes text.** A 217-token prefill routes to 224 of 288 experts a layer (78 %), 304 tokens to 227 (79 %), 24 tokens to 77. GLM's routing is concentrated, not uniform.

**Lever A, bit-identical: no whole-layer read ahead below a threshold.** Reads = the routed non-resident experts only. Reads saved against shipped: 217 tokens 25.6-26.0 %, 304 tokens 28.3-28.6 % (three traces, budget 2 or exact, 46 GiB); 31.8 % at 60 GiB. A 2,738-token prompt: 4.1 % on its 2,048-token chunk, 8.9 % on its 690-token chunk. Below 128 tokens nothing changes (no read ahead today). The cost: reads no longer overlap the layer's compute, and the drive idles while each layer routes.

**Lever B, MiniMax's rule on top of A (output-changing).** Leave a missing expert unread when it is routed to at most 32 rows and holds under 0.20 of every one of them. Upper bound (a top-8 trace cannot show the runner-up the rule also needs). Reads saved 79-88 % on short prefills, but it changes 78-99 % of the (row, layer) pairs, and a changed row loses a median 15-28 % of its routing weight (p90 35-54 %). At a GLM-scaled threshold of 0.10 (MiniMax routes 4 experts, GLM 8): 44-63 % of reads, 33-87 % of pairs changed, median loss 8-12 %. DeepSeek's rare-expert skip failed the quality gate with a median of 12.5 % per affected row (18.260), and GLM is the model whose C# already breaks at 4-bit (18.44). **B is closed on price**: no port, no gate run.

## 3. Hypothesis

A short GLM prefill on the internal drive is bound by reads, and a third of those reads are read-ahead of experts no row routes to. Reading only the routed set cuts a 160-300-token prefill by about the reads it saves, less the overlap it loses; the loss grows with the prompt's compute, so a threshold of about 500-800 tokens is where the two cross.

Alternative: the read-ahead's overlap is worth more than its wasted reads even at 160 tokens (the drive needs depth to reach its rate, and demand reads issue one layer at a time), so lever A loses or is flat everywhere.

## 4. Design

The knob exists: `CACHALOT_GLM_SPECULATE_MIN_TOKENS` (default 128). No code change. Arms: S = 128 (shipped), D = 1,200 (no read ahead for any chunk under 1,200 tokens). The cold block's 2,048-token chunks and its last 1,233-token chunk read ahead in both arms.

Workload (`g6_driver.py`, W5): a fresh `serve-glm.sh --expert-budget-gib 46`, no disk snapshots, upscale knob on as live. Request 0 is the live session's first request (15.6k-token system block plus "Hi"), prefilled cold. Then nine requests keep that block and replace the user message with consecutive slices of `docs/RESEARCH-DIRECTION.md` of about 25, 60, 120, 160, 250, 400, 528, 800 and 1,100 tokens. Every request is greedy with `max_tokens` 1, so no decode changes residency between rungs, and both arms see identical tokens. ABBA: S1, D1, D2, S2, `benchmarks/settle.sh` between arms (`g6_run_all.sh`). Recorded: per rung the prefilled count, prefill seconds and a hash of the one-token reply.

Runtime estimate: four arms of about 6.5 minutes (40 s load, 205 s block, about 2.5 minutes of rungs) plus settling, about 32 minutes.

## 5. Predictions (registered before the run)

Per rung, D against S, the mean of the two pairs:

- **G6-1.** 25 and 60 tokens (no read ahead in either arm): within +-10 %.
- **G6-2.** 120 tokens (under the shipped 128): within +-10 %.
- **G6-3.** 160 tokens: D faster by 20-45 %; S takes 15-23 s.
- **G6-4.** 250 tokens: D faster by 10-35 %.
- **G6-5.** 400 and 528 tokens: D between -25 % and +5 %.
- **G6-6.** 800 and 1,100 tokens: D between -10 % and +20 % (the lost overlap may make it slower).
- **G6-7.** Bit-identity: the same prefilled counts and reply hashes in all four arms on every rung.
- **G6-8.** S's rungs from 160 to 1,100 tokens lie within 15-25 s, nearly flat (the whole-layer read ahead dominates).
- **Falsifier.** D is not faster at 160 tokens by more than the S1-S2 spread: the overlap is worth the wasted reads, the hypothesis is wrong, and lever A closes.

**Decision rule.** If G6-3 and G6-4 hold, the next step is a threshold between the fastest rung D wins and the first it loses, offered to Hamed as the `serve-glm.sh` default (bit-identical; the default is still his call). A predicted-set read ahead (read only what the next router is likely to pick, MiniMax's 0.27.0) is the follow-up that keeps the overlap and drops the waste; it is not built here.

## 6. Results

Not run.
