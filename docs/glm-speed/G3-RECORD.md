# G3: GLM's expert bank on the internal SSD

Written 2026-11-06 against runtime 0.81.1 plus the `serve-glm.sh` default, before the run. Hamed chose the full bank, cleared caches and removed Docker's data. HANDOFF section 18.272 when scored.

## 1. Baseline (measured, 18.271, X10Pro)

Cold 15,569-token block 1,316.8 s. Turns: address form prefill 159.2 s and decode 0.97 tok/s; logo 168.6 s, 1.06; tool call 25 tokens 37.3 s, 0.83; tool result 528 tokens 166.9 s, 1.05. Misses a token 56-75.

## 2. Hypothesis

Both halves are bound by the X10Pro's 1 GB/s. On the internal drive, about 2.9 ms an expert against 14.7, decode approaches 63 ms + 2.9 ms x misses, and prefill approaches 2,048-token chunks of about 30 s.
Alternative: the internal drive's sustained rate beside the GPU falls short, as the 0.48.1 baseline (2.0-2.9 tok/s exact at 52 GiB) hints.

## 3. Design

One fresh `serve-glm.sh --expert-budget-gib 46`, same knobs as the live session, no disk snapshots, the same five requests from the private dump, non-streaming, bank at `~/GLM-5.3-Flash-bank` (byte-identical to the X10Pro copy, 44 files). One arm: the live session is the control, on a different run and with Hermes's window open.

## 4. Predictions (registered before the run)

- **B-1.** Five requests, no error; startup shows the internal bank.
- **B-2.** Cold block 200-420 s (8 chunks of 25-50 s).
- **B-3.** Decode 2.5-4.5 tok/s on the replies of over 50 tokens.
- **B-4.** Prefill of the 1,170-1,240-token image turns 30-70 s; the 25-token turn 5-15 s.
- **Falsifier.** Decode under 2 tok/s: the internal drive is not the whole story, and the lane's G3 price is wrong.

Runtime estimate: about 12 minutes.

**Amendment before any reply (2026-11-06).** The budget is 40 GiB, not 46: after `sudo purge`, with Chrome and ChatGPT open, the guard counted 61 GiB, under the about 68 GiB that 46 needs and about the 62 that 40 needs. The internal disk has 12 GiB free, so a swap run is worth avoiding. Fewer resident experts lean against the arm: decode at 40 GiB ran 0.87-0.89 tok/s on the X10Pro (18.269) against the live session's 0.83-1.06. Predictions unchanged.

**Second amendment, before any reply.** The 40 GiB attempt returned no data: `driver.py` crashed on the dump's response rows while the server loaded, and the server idled 40 minutes to the guard's limit (fixed in the driver and in `arm.sh`). The machine then showed 75 GiB, so the run goes back to the registered 46 GiB, the live session's budget.

## 5. Results (measured 2026-11-06, commit of the second amendment)

**Run.** `arm.sh R2 46`, about 7 minutes, sysctl 88064, no other runtime, ChatGPT open; 75.4 GiB by the guard's count, pressure level 1, swap flat at 1,573 MB, peak wired 74.3 GiB. Replies were not stored; the replay's text differs from the live run's, so prefilled counts after turn 1 differ.

| turn | X10Pro, live (18.271) | internal, replay | decode tok/s, then and now | misses a token |
|---|---|---|---|---|
| cold block | 1,316.8 s prefill, 1,328 s turn | **205.2 s** prefill, 208 s turn | | 57 |
| address form | 159.2 s for 1,240, 317 s turn | 26.2 s for 1,252, **63 s** turn | 0.97, **3.60** | 67 |
| logo | 168.6 s for 1,170, 298 s turn | 22.5 s for 1,322, **59 s** turn | 1.06, **3.90** | 58 |
| tool call | 37.3 s for 25, 65 s turn | 19.4 s for 160, **26 s** turn | 0.83, **3.39** | 75 |
| tool result | 166.9 s for 528, 289 s turn | 18.8 s for 528, **52 s** turn | 1.05, **3.94** | 58 |

**Scored.** B-1 held. B-2 held (205 s, low edge). B-3 held (3.60-3.94). B-4: image prefills 22-26 s, faster than the predicted 30-70; the 25-token case did not recur (160 tokens prefilled in 19 s, not comparable). Falsifier not fired.

**Reading (derived).** The token moved from about 1.05 s to 0.25-0.29 s, as G3's price said (0.27 s). The bottleneck stays reads, now on the internal drive, and short prefill: a 160-token prefill still costs 19 s. Same bank bytes; the X10Pro copy stays as a fallback.

**Limit.** One replay against one live run, not a pair; Hermes was not driving this arm, and its window state here was not recorded. Internal disk has 13 GiB free.

## 6. Kind

Research: predicted, measured, explained (B-1 to B-3 held, B-4 split). One default changed (the bank's location, Hamed's call).
