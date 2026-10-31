# DI-1 server A/B: results (the knob takes 0.668 of the prefill time on the live session's three large prefills, same build, ABBA)

Run 2026-10-31, 14:31 to 14:53 CET, against runtime 0.71.1 plus the driver (arms 1 to 3) and 0.71.2 (arm 4, see section 4). Record: `docs/DI1-SERVER-AB-RECORD.md` (S-1 to S-8, committed first in 6e16a14, before the driver in 47d84c2). Raw outputs: `docs/di1-server-ab/` (one JSON and one server log an arm, `score.json`). Every number is **measured** unless marked derived. Machine: `iogpu.wired_limit_mb` 88064, the internal 2-bit bank, `serve.sh` defaults (48 GiB budget, decode miss budget 0, wired 80, the 863-expert hotlist, fast synch), no other runtime, the display as it was.

## 1. Prefill (the server's `prefill_seconds`)

| request | prefilled (reused) | off: arm 1, arm 4 | on: arm 2, arm 3 | mean off | mean on | on over off |
|---|---|---|---|---|---|---|
| 1, the cold system block | 16,040 (0) | 163.03, 163.18 | 106.65, 109.96 | 163.10 | 108.31 | **0.664** |
| 2, a short turn | 27 (16,041) | 2.44, 2.44 | 2.45, 2.42 | 2.44 | 2.43 | 0.997 |
| 3, a large tool result | 11,570 (16,069) | 103.15, 103.66 | 66.25, 66.34 | 103.40 | 66.29 | **0.641** |
| 4, a second tool result | 6,544 (27,640) | 65.36, 64.58 | 47.18, 46.50 | 64.97 | 46.84 | **0.721** |
| 1 + 3 + 4 | 34,154 | 331.5, 331.4 | 220.1, 222.8 | 331.5 | 221.4 | **0.668** |

Per prefilled token: off 10.17, 8.94 and 9.93 ms; on 6.75, 5.73 and 7.16 ms. The client's wall time is the server's `prefill_seconds` plus 0.19 to 0.29 s in every request (the one-token decode and HTTP).

**S-1 held** (0.664 in 0.55 to 0.70). **S-2 held** (0.641 in 0.50 to 0.75). **S-3 held** (0.721 in 0.55 to 0.85). **S-4 held** (0.668 in 0.55 to 0.72). **S-5 held**, both parts: every request prefilled the same count in all four arms (16,040, 27, 11,570, 6,544), so the knob's snapshot policy changed no reuse on these requests. **S-7 held:** the off arms' request 1 within 0.1 %, the on arms' within 3.1 %. **S-8 held:** the on arms' request 1 mean is 108.31 s against the live session's 102.45 (+5.7 %).

Against the bench (derived): the server's 0.664 on a 16,040-token cold block sits between the bench's 0.622 at 14,796 tokens and 0.608 at 22,000, with the knob-off rate (10.17 ms a token) equal to the bench's 10.21. The knob saves 54.8 s on a cold system block, 37.1 s on the 11,570-token tool result and 18.1 s on the 6,544-token one, 110.1 s over the three.

## 2. The one greedy token

Requests 1 to 3 gave the same token in all four arms ("Hey", "\n\n", "That"). Request 4 gave "Here" in arms 1 and 2 and "Top" in arms 3 and 4: the split follows the run order (off, on, on, off), not the knob. **S-6 missed** as worded (equal on all four requests); its falsifier (a difference on request 1) did not occur.

**Explanation (derived, not tested).** Request 4's first token comes from its prefill's logits, and DeepSeek's prefill drops no expert, but its 27,640 reused tokens end with the one token request 3 decoded ("That"), and that decode step ran under `serve.sh`'s decode miss budget 0, which drops every non-resident expert. Which experts are resident at that step depends on the cache's history. The miss counts do not separate the two groups (request 3's decode step missed 64 and 62 experts in arms 1 and 2, 65 and 61 in arms 3 and 4); which experts were dropped is not logged. A different set of dropped experts gives that token's KV slightly different values, and request 4 reads it. The two arms run first agree with each other and the two run last agree with each other, which fits a cache state that drifts over the run. A test would replay with `CACHALOT_DECODE_MISS_BUDGET=off`; not run.

## 3. Memory

The watchdog never tripped. Pressure reached level 2 (warning) in arms 2 to 4, never 4. Swap grew from 1,204 MB before arm 1 to 1,733 MB after arm 4; inside one arm at most +288 MB (arm 2), under the 512 MB stop rule.

## 4. Deviations from the record, stated

* **The settle gate ran at 40 GiB, not 48.** The first launch (14:21) waited 603 s and timed out before arm 1: at 48 GiB `settle.sh` asks for 77 GiB available and the machine reached 73.9 at most with Hamed's applications open. Nothing ran. The gate only checks that the previous arm gave its memory back; at 40 it asks for 69. The server itself held `serve.sh`'s 48 GiB in every arm, as recorded.
* **Arm 4 ran under 0.71.2's `serve.sh`.** Commit 6181f7a (14:44, during arm 3) changed `serve.sh` to derive the bank and hotlist from `CACHALOT_MODEL_PATH`, with the old defaults unchanged, and `src/` only in the version string. Arm 4's log resolves the same checkpoint, the same internal bank (9.49 MiB an expert, 15,360 experts) and the same 863-expert hotlist at 48 GiB as arms 1 to 3, and its request 1 is within 0.1 % of arm 1's. The record's "no `src/` edit while it runs" was not breached in substance; it is stated here.

**Erratum (0.72.0):** Hamed's download of DeepSeek-V4.1-Flash-1017 to the X10Pro started at 14:35, during arm 1's third request, and ran through arms 2 to 4; it was not noticed when this was written. Its effect looks small: arm 1 (mostly before it) and arm 4 (inside it) agree within 0.1 % on request 1, 0.5 % on request 3 and 1.2 % on request 4. `docs/DI1-DEFAULT-ON-RESULTS.md` section 3.

## 5. What it means (derived; decisions are Hamed's)

The knob's saving holds in the server on Hermes's real prompts, on a same-build pair run both ways: a third of the prefill time on a cold system block and on large tool results, with no change in what a follow-up reuses. With the long-context gate (`docs/V270-MAC-JOBS-RESULTS.md` section 2: closer to the full prefill than re-chunking noise) and the live session's answers (`docs/DI1-LIVE-SESSION-RESULTS.md`), every check registered for DI-1 has now run. What remains is policy: the default in `serve.sh`, and whether snapshots taken with the knob on should be marked (§18.221), since they differ from a full prefill's in rounding.

## 6. Limits

Four requests of one session, two replays each per knob value, one machine state. One greedy token a request, so the quality evidence here is only that the first token matched on three of four requests and that the fourth's difference followed run order. The compression re-prefill was not exercised.
