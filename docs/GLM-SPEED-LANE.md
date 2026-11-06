# GLM-5.3-Flash speed lane

Written 2026-11-06 at runtime 0.81.1, revised at 0.83.0 after G3 and G2. Hamed, after the live session of HANDOFF 18.271: "update the next lanes to focus on GLM speed. it is unusable now!" Every price below is derived from
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
| G3 | Expert bank on the internal SSD | **DONE, measured 0.82.0 (18.272):** decode 3.39-3.94 tok/s, cold block 205 s, turns 26-63 s | bit-identical | shipped | |
| G4 | Expert budget 60-68 GiB | `cache_sim`: misses -18 to -27 % against 46 GiB on two traces, token about -15 to -25 % | bit-identical | a quiet machine: 68 needs about 97 GiB available | the guardian; GLM at 52 GiB was killed with apps open |
| G5 | Decode budget 1 | 1,100-1,133 to 642-693 ms a token, about -40 % (18.107); budget 0 is 68 ms | changes outputs: b1 KL 0.04-0.14, b0 dNLL +0.185 on prose | a blind replay and the compile panel; default Hamed's | visible loss |
| G6 | A prefill miss budget for short chunks, MiniMax's 0.42.0 | turn prefills of 25-1,240 tokens are 40-50 % of a turn; MiniMax's follow-ups -20 % | changes outputs; the gate of 18.260 applies, price per affected row | a port, a gate | DeepSeek's rare skip landed at the noise arm's KL |
| G7 | A 2-bit GLM bank | half the bytes a miss, twice the slots; fits the internal SSD (about 82 GiB) | changes outputs; GLM's C# already garbles at 4-bit (18.44) | a requantization and the full gate | the last resort |

## 3. What is already closed for GLM

Bank layout (shipped, 18.86), pair index, S1c, MTP, decode prefetch (off, 18.105), an fp32 stream, a higher-precision hybrid. Do not reopen them without a new mechanism.

## 4. Honest ceiling

GLM's resident token is 63 ms; on the internal drive a Hermes token is about 0.27 s. The cold block is solved for ordinary use (2-30 s). What is left is misses (58-75 a token) and short prefill (19-26 s a turn), so G4, G5 and G6 are what a turn feels now. M19 stays halted.

## 5. First jobs

1. G6's price: count a short prefill's reads per turn from a trace, then decide on a port. A 160-token prefill costs 19 s of a 26 s turn. Price per affected row (18.260).
2. G4 on a quiet machine: GLM at 60-68 GiB. Internal disk has 15 GiB free, so a swap-prone run is not free.
3. G5, a blind replay at budget 1, if a token at 0.27 s is still too slow: a default is Hamed's call.
