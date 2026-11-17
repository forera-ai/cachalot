# GLM-5.3-Flash speed lane

Written 2026-11-06 at runtime 0.81.1, revised at 0.83.0 after G3 and G2, at 0.83.1 after G6 (18.274), at 0.84.0 when its threshold shipped (18.275), at 0.85.0 after G6b (18.276), at 0.85.1 when G6b closed (18.277), at 0.85.2 when G4 was parked (18.278), at 0.85.3 after G5 (18.279), at 0.85.4 when the internal bank was deleted again (18.280), at 0.85.5 when it was copied back (18.281), and at 0.85.6 after a live session (18.282). Hamed, after the live session of HANDOFF 18.271: "update the next lanes to focus on GLM speed. it is unusable now!" Every price below is derived from
recorded numbers unless it says measured; none is a result until its experiment record runs. Rank order: expected gain, confidence, cost, and whether output changes.

## 1. Where the time goes (measured, 2026-11-06, `serve-glm.sh --expert-budget-gib 46`, knob on, Hermes, 21 tools)

| turn | prompt | prefilled | prefill | reply tokens | decode | turn wall |
|---|---|---:|---:|---:|---:|---:|
| "Hi", cold block | 15,569 | 15,569 | **1,316.8 s** (11.8 tok/s) | 13 | 10.7 s | 1,328 s |
| address form, 906 x 943 | 16,821 | 1,240 | 159.2 s | 153 | 157.6 s (0.97 tok/s) | 317 s |
| logo, 438 x 398 | 18,143 | 1,170 | 168.6 s | 136 | 128.9 s (1.06 tok/s) | 298 s |
| tool call | 18,303 | 25 | 37.3 s | 23 | 27.6 s (0.83 tok/s) | 65 s |
| tool result answer | 18,853 | 528 | 166.9 s | 128 | 121.9 s (1.05 tok/s) | 289 s |

Decode ran 56-75 misses a token at 70-82 % hits. Both halves were read-bound on the X10Pro's 1 GB/s: a 13.5 MiB expert costs 14.7 ms, and a token is 63 ms of compute (resident, 18.161) plus
about 1 s of reads. Short prefills cost as much as decode because prefill has no miss budget: 25 tokens took 37 s. A chunk of 2,048 tokens routes to nearly all 12,096 experts (17.1), so each
chunk of a long prefill scans most of the 163 GB bank: the 15.6k block made about 8 passes at about 165 s each.

## 2. The levers, in order

| id | lever | price (derived) | output | needs | closes on |
|---|---|---|---|---|---|
| G1 | `CACHALOT_GLM_PREFILL_CHUNK=8192`. The knob exists; 2,048 was chosen long ago | only a fully cold block now: 196 s to perhaps 100 s (derived); G2 removed most of its use | changes KV bits and every snapshot's identity | a run | not worth a re-prefill of everything unless cold blocks recur |
| G2 | In-block snapshot pins and a same-block restart check | **DONE, measured 0.83.0 (18.273):** same block after a restart 5.7 s, changed block 29 s, bit-identical | bit-identical | shipped, `CACHALOT_GLM_BLOCK_PINS=0` is off | |
| G3 | Expert bank on the internal SSD | **DONE, measured 0.82.0 (18.272):** decode 3.39-3.94 tok/s, cold block 205 s, turns 26-63 s. **Undone 0.85.4, restored 0.85.5 (18.280-18.281):** deleted for space and copied back the same day, byte-compared | bit-identical | shipped; internal again, verified | |
| G4 | Expert budget above 46 GiB | `cache_sim` on four traces (0.85.2, `docs/glm-speed/G4-RECORD.md`): misses -9 to -12 % at 52, -14 to -17 % at 56, -18 to -23 % at 60; token about -10 to -13 % at 56 (derived). **64-68 do not fit:** GLM wires about budget + 27 GiB of 96 | bit-identical (except budget 2's residency-dependent drops) | **parked 0.85.2 (18.278):** the guard counted 63-65 GiB on a running desktop, 85 needed at 56; needs a restarted Terminal-only session; runner `g4_run_all.sh` | a run in that state; a 56 default would not start in Hamed's usual state |
| G5 | Decode budget 1 | **Run 0.85.3 (18.279, `docs/glm-speed/G5-RECORD.md`), blind, 48 replies at 24 GiB:** 205 against 463 ms a token (0.44); flawed 14/24 against 13/24, but corrupted tokens in 6 budget-1 replies and no exact one | changes outputs: b1 KL 0.04-0.14 (18.107); corrupted words and identifiers | **not a candidate** (registered rule: a new failure class); `serve-glm.sh` stays at budget 2 | a 46 GiB rerun needs about 73 GiB available |
| G6 | Short prefill reads. **Measured 0.83.1 (18.274):** the whole-layer read-ahead from 128 tokens makes 172-795 tokens all about 20 s; routed set only (`CACHALOT_GLM_SPECULATE_MIN_TOKENS` above the prompt) -21 % at 172 (clean pair), -15 % at 239, -10 % at 342, -5 % at 499, +7 % at 795. MiniMax's miss rule (the original G6) closed on price: 78-99 % of rows changed at a median 15-28 % of their weight | bit-identical | **shipped 0.84.0:** 512 in `serve-glm.sh`; live check 0.85.6 (18.282) did not reach the 128-511 band (667 tokens 17.8 s, under 128 tokens 6.8-8.2 s). **G6b** (predicted-set read-ahead, `CACHALOT_GLM_PREFILL_PREDICT_TOPK`, 0.85.0, off): priced 3-7 %; A/B void (18.276); the block check found the gap was the machine (18.277): **G6b closed**, at most -2.6/-4.5 % at 342/499 tokens | the miss rule and G6b: closed |
| G7 | A 2-bit GLM bank | half the bytes a miss, twice the slots; fits the internal SSD (about 82 GiB) | changes outputs; GLM's C# already garbles at 4-bit (18.44) | a requantization and the full gate | the last resort |

## 3. What is already closed for GLM

Bank layout (shipped, 18.86), pair index, S1c, MTP, decode prefetch (off, 18.105), an fp32 stream, a higher-precision hybrid. Do not reopen them without a new mechanism.

## 4. Honest ceiling

GLM's resident token is 63 ms; on the internal drive a Hermes token is about 0.27 s, on the X10Pro about 1 s (the fallback copy). The cold block is solved for ordinary use (2-30 s). What is left is misses (58-75 a token) and short prefill (19-26 s a turn), so G4, G5 and G6 are what a turn feels now. G4 needs a restarted Terminal-only session to run at all (18.278). G6's threshold (0.84.0) takes 3-4 s off a 172-token prefill; prefills under 128 tokens still cost 8-14 s. M19 stays halted.

## 5. First jobs (0.85.6)

0. The internal bank is back and verified (0.85.5, 18.281); the block's pins are on disk since 0.85.6. The 0.85.6 session ran with swap at 5.1 of 6 GiB (cold block 558 s, not about 200; 18.282): `sudo purge` (Hamed) and applications closed before any live check or guarded run.
1. A live Hermes check of 0.84.0's 512 threshold (Hamed drives), with at least one 150-500-token follow-up (a pasted paragraph): it should prefill in about 15-19 s, not about 20. 0.85.6's session had none.
2. G5 ran (0.85.3, 18.279): budget 1 corrupts tokens at 24 GiB, so budget 2 stays. A rerun at 46 GiB only with applications closed (about 73 GiB available).
3. G4 only in a restarted session with Terminal alone: `bash docs/glm-speed/g4_run_all.sh 56` (52 if the gate fails). Parked, not closed.
