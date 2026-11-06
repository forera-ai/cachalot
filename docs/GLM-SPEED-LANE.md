# GLM-5.3-Flash speed lane

Written 2026-11-06 at runtime 0.81.1. Hamed, after the live session of HANDOFF 18.271: "update the next lanes to focus on GLM speed. it is unusable now!" Every price below is derived from
recorded numbers unless it says measured; none is a result until its experiment record runs. Rank order: expected gain, confidence, cost, and whether output changes.

## 1. Where the time goes (measured, 2026-11-06, `serve-glm.sh --expert-budget-gib 46`, knob on, Hermes, 21 tools)

| turn | prompt | prefilled | prefill | reply tokens | decode | turn wall |
|---|---|---:|---:|---:|---:|---:|
| "Hi", cold block | 15,569 | 15,569 | **1,316.8 s** (11.8 tok/s) | 13 | 10.7 s | 1,328 s |
| address form, 906 x 943 | 16,821 | 1,240 | 159.2 s | 153 | 157.6 s (0.97 tok/s) | 317 s |
| logo, 438 x 398 | 18,143 | 1,170 | 168.6 s | 136 | 128.9 s (1.06 tok/s) | 298 s |
| tool call | 18,303 | 25 | 37.3 s | 23 | 27.6 s (0.83 tok/s) | 65 s |
| tool result answer | 18,853 | 528 | 166.9 s | 128 | 121.9 s (1.05 tok/s) | 289 s |

Decode ran 56-75 misses a token at 70-82 % hits. Both halves are read-bound on the X10Pro's 1 GB/s: a 13.5 MiB expert costs 14.7 ms, and a token is 63 ms of compute (resident, 18.161) plus
about 1 s of reads. Short prefills cost as much as decode because prefill has no miss budget: 25 tokens took 37 s. A chunk of 2,048 tokens routes to nearly all 12,096 experts (17.1), so each
chunk of a long prefill scans most of the 163 GB bank: the 15.6k block made about 8 passes at about 165 s each.

## 2. The levers, in order

| id | lever | price (derived) | output | needs | closes on |
|---|---|---|---|---|---|
| G1 | `CACHALOT_GLM_PREFILL_CHUNK=8192` on the X10Pro. The knob exists; chunk 2,048 was chosen on the internal drive | cold block 1,317 s to about 350 s (2 passes, not 8) | changes KV bits; 17.1: chunkings differ by KL ~0.024, flat NLL. Gate on a long context | a run, about 1 hour | no gain, or memory: MiniMax needed a prefill capacity plan at large chunks |
| G2 | In-block snapshot pins, DeepSeek's 0.59.0, and a same-day restart check | 22,618 of 23,104 system-block characters are shared across days, so a new day re-prefills about the tail chunk, not the block: minutes, not 22 | bit-identical at chunk boundaries | a build | the shared prefix shrinks |
| G3 | Expert bank on the internal SSD | token 1.14 s to about 0.27 s (63 ms + 69 x 2.9 ms), 3.7 tok/s; short prefills about 3.5x shorter; an image turn 317 s to about 80 s | bit-identical | **Hamed's decision: 152 GiB needed, 103 GiB free** | the internal drive cannot sustain it beside prefill |
| G3b | A hot tier on the internal SSD, 70 to 90 GiB of the most-used experts | top 7,000 experts (92 GiB) cover 84 % of decode references (one trace pair sharing a prompt, not independent); misses are colder, so gain under G3: about -35 % a token | bit-identical | a build, no deletion | misses fall outside the tier |
| G4 | Expert budget 60-68 GiB | `cache_sim`: misses -18 to -27 % against 46 GiB on two traces, token about -15 to -25 % | bit-identical | a quiet machine: 68 needs about 97 GiB available | the guardian; GLM at 52 GiB was killed with apps open |
| G5 | Decode budget 1 | 1,100-1,133 to 642-693 ms a token, about -40 % (18.107); budget 0 is 68 ms | changes outputs: b1 KL 0.04-0.14, b0 dNLL +0.185 on prose | a blind replay and the compile panel; default Hamed's | visible loss |
| G6 | A prefill miss budget for short chunks, MiniMax's 0.42.0 | turn prefills of 25-1,240 tokens are 40-50 % of a turn; MiniMax's follow-ups -20 % | changes outputs; the gate of 18.260 applies, price per affected row | a port, a gate | DeepSeek's rare skip landed at the noise arm's KL |
| G7 | A 2-bit GLM bank | half the bytes a miss, twice the slots; fits the internal SSD (about 82 GiB) | changes outputs; GLM's C# already garbles at 4-bit (18.44) | a requantization and the full gate | the last resort |

## 3. What is already closed for GLM

Bank layout (shipped, 18.86), pair index, S1c, MTP, decode prefetch (off, 18.105), an fp32 stream, a higher-precision hybrid. Do not reopen them without a new mechanism.

## 4. Honest ceiling

GLM's resident token is 63 ms. On the X10Pro no software lever short of dropping experts reaches 2 tok/s. G3 is the only bit-identical lever that makes a Hermes turn feel usable: G1 and G2 fix the
cold block only, and G4 is a fifth off a token. M19, a faster external drive, stays halted by Hamed's 2026-10-01 decision and is not proposed.

## 5. First jobs

1. G1, after an experiment record with predictions committed first. About 1 hour: one cold 15.6k prefill at 8,192 (about 6-8 minutes), a repeat at 4,096, then the long-context quality check.
2. Hamed: G3's decision. Three ways to find 152 GiB: move MiniMax's coded bank to the X10Pro copy that exists (`/Volumes/X10Pro/models/MiniMax-M3-coded-bank`, compare bytes first, then delete the internal 158 GiB with his explicit yes; MiniMax, rank 2, then reads over USB);
   delete DeepSeek's copy (rank 1: no); or G3b within the free space.
3. G2, then G4 on a quiet machine.
