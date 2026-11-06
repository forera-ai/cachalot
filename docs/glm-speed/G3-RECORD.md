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

## 5. Results

Not yet run.

## 6. Kind

Not yet scored.
