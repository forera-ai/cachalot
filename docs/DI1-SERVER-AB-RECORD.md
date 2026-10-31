# DI-1 server A/B on the live session's first four requests (experiment record; predictions before any build or run)

Written 2026-10-31 against runtime 0.71.1, **before the driver is written and before any run**. Charter section 7.2 fields 1 to 3 are fixed here; this file is committed first and results go in `docs/DI1-SERVER-AB-RESULTS.md`. A prediction edited after its run is void. Hamed asked on 2026-10-31 ("Run the A/B first") for the same-build server pair that the live session (`docs/DI1-LIVE-SESSION-RESULTS.md`) lacked, offered at about 20 minutes.

## 1. Baselines (recorded)

* **Bench** (`docs/V270-MAC-JOBS-RESULTS.md` section 1, in-process, 0.70.1 plus the knob): a cold 22,000-token prefill 224.6 s full against 136.5 s with `CACHALOT_PREFILL_ENCODER_ONLY=1` (ratio 0.608; 10.21 and 6.20 ms a token); at 14,796 tokens 0.622.
* **Live session, knob on** (0.71.0, `docs/v270-results/di1-live-server.log`): request 1, 16,040 tokens cold, prefill 102.45 s (6.39 ms a token); request 2 reused 16,049 and prefilled 19 in 1.93 s; request 3 reused 16,161 and prefilled 11,478 in 63.38 s (5.52 ms); request 4 reused 27,704 and prefilled 6,480 in 44.62 s (6.89 ms).
* **Knob off on this build:** not measured in the server. The live write-up derived about 164 s for request 1 at the bench's rate.

## 2. Method

`benchmarks/di1_server_ab.sh` (built after this record) runs four arms in the order **off, on, on, off** (ABBA). Each arm:

1. `benchmarks/settle.sh --budget-gib 48`, then `./serve.sh` with `CACHALOT_PREFILL_ENCODER_ONLY` set to `0` or `1` and `CACHALOT_SNAPSHOT_DIR` set to a new empty directory, so request 1 prefills the system block cold and no arm reads another's snapshots. Every other setting is `serve.sh`'s default (48 GiB budget, decode miss budget 0, internal 2-bit bank, wired 80, hotlist, fast synch). The server log goes to the arm's own file.
2. The driver (`benchmarks/di1_server_ab_driver.py`, run with the system Python after the server answers `/v1/models`) sends the dump's first four request bodies (`~/cachalot-di1-live.jsonl` lines 0, 2, 4, 6) in order, each with `max_tokens` 1, `temperature` 0 and `stream` false, and nothing else changed. It records the server's `prompt_tokens`, `reused_prefix_tokens`, `prefill_seconds`, the client wall time and the one completion token.
3. The server is stopped with SIGINT.

Because `max_tokens` is 1, the replies the live session spliced back (93 and 65 tokens) are not generated, so requests 3 and 4 may reuse a different prefix than live. Both arms see the same bodies, so any reuse difference between arms comes from the knob's snapshot policy (snapshots only at the requested boundaries and the prompt's end) and is part of what is measured. The primary quantity is the server's `prefill_seconds`; the prefilled token count is reported per request and arm.

**Stop rules:** a server that dies or does not answer within 15 minutes; `kern.memorystatus_vm_pressure_level` at 4 (critical) on two samples, or swap growing more than 512 MB during an arm (the driver samples both every 2 s and SIGINTs the server); any request error. One runtime process at a time. No `src/` edit while it runs. The driver writes `docs/di1-server-ab/arm-<n>-<off|on>.json` after every request.

**Estimate (derived):** knob off about 164 + 117 + 66 s of prefill, knob on about 102 + 63 + 45 s; model load and the 20 s quiet wait about 2 minutes an arm; **about 26 minutes** in all.

## 3. Predictions

All **predicted**; probabilities mine.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| S-1 | request 1 (16,040 tokens, cold): mean on `prefill_seconds` over mean off | 0.55 to 0.70 (0.70) | above 0.80 |
| S-2 | request 3 (the large tool result): on over off | 0.50 to 0.75 (0.60) | above 0.90 |
| S-3 | request 4 (the second tool result): on over off | 0.55 to 0.85 (0.55) | above 0.95 |
| S-4 | requests 1, 3 and 4 summed: on over off | 0.55 to 0.72 (0.65) | above 0.85 |
| S-5 | prefilled tokens | request 1 equal in all arms (16,040) (0.95); requests 2 to 4 equal between arms (0.60) | request 1 differs |
| S-6 | the one greedy completion token | equal between arms for all four requests (0.80) | it differs on request 1 |
| S-7 | order noise | the two off arms' request 1 `prefill_seconds` within 5 % of each other, and the two on arms' too (0.70) | either pair more than 15 % apart (the pair cannot be read) |
| S-8 | the live session reproduces | on arms' request 1 mean within ±10 % of the live 102.45 s (0.70) | more than 20 % off |

## 4. What each outcome means

* **S-1 to S-4 held:** the bench's saving holds in the server path on Hermes's real prompts, on a same-build pair. Together with the long gate (`docs/V270-MAC-JOBS-RESULTS.md` section 2) and the live session's answers, the case for a default-on is Hamed's to weigh against the snapshot policy (§18.221). **S-1 falsified:** something in the server path (the snapshot points, the chunking around the system block's boundary) eats the saving, and the default stays off until that is found.
* **S-5 second part falsified:** the knob's snapshot policy changes what a follow-up can reuse; reported with the per-request counts, since the sum (S-4) is what a session pays.
* **S-6 falsified on request 1:** the rounding difference reaches the first greedy token on this prompt; reported, and weighed against the long gate, which measured it closer to the full prefill than re-chunking noise.

## 5. Limits fixed in advance

One machine state, four requests of one session, one replay each per arm (two per knob value). The `prefill_seconds` is the server's own; the client wall time is reported beside it.
