# Experiment record: does the 1017 startup hotlist help a Hermes first turn? (ABBA through `serve.sh`)

Written 2026-11-02 before the first arm, runtime 0.73.0 (0503b3e). Hamed: "Go, run A/B" (v294's open item: a first-turn A/B of the
1017 hotlist on Hermes-shaped traffic).

## 1. Question

`serve.sh` on 1017 preloads 863 experts (8.0 GiB, `~/cachalot-hotlist-DeepSeek-V4.1-Flash-1017.json`) in the background at startup.
On 1017's own sessions that set covers 32-33 % of a session's expert requests (§18.246), but coverage is an upper bound, and on 0.60.10's
short-prompt driver the miss count did not fall (§18.248). Does the preload make a real Hermes first turn (cold system block plus the
first reply) faster, or change how many experts budget 0 drops?

## 2. Baseline (recorded)

- First release, 2026-09-17 (HANDOFF-2026-09-17 section 28): a 512-token cold prompt at 36 GiB, four runs a side: cold prefill -5.7 %
  with the hotlist (16.6 against 15.7 s, arms did not overlap), first-turn hit rate +4.0 points, decode +2.5 % (inside the 7 % spread),
  ready +0.2 s.
- 1017 live (0.72.1, §18.243) and the DI-1 server A/B (0.71.3): the cold ~16,040-token Hermes system block prefills in 102-108 s with
  DI-1 on.
- The preload is LRU, not pinned (`ResidentExpertStore` preload docstring): it fills the cache before the first prompt and leaves a
  reserve; whatever the turn does not use is evicted first. A 16k-token prefill touches nearly every expert in every layer, far more
  than the 48 GiB pool (about 5,180 experts) holds.

## 3. Predictions (fixed before the run)

- H-1. Cold first-turn prefill (system block + first user message, DI-1 on): hotlist on minus off between -3 % and +1 % of the off
  time (mean of two arms each). Falsified if on is more than 5 % faster or more than 3 % slower. Reason: the preload saves at most
  one read of 8 GiB (about 1.3 s on the internal SSD at 6.8 GB/s, less when hidden under prefill compute) out of ~105 s, and the prefill
  evicts most of the set before the last chunk.
- H-2. First-reply decode, ms a token: |on - off| under 4 %. Falsified above 8 %.
- H-3. Experts dropped by budget 0 in the first reply, a token: on minus off between -3 and +1. Falsified if on drops more than 1.5 a
  token more than off.
- H-4. Repeatability: the two arms of one condition agree within 3 % on prefill time.
- H-5. Hit rate over the turn (from `/v1/stats`): on minus off within +-3 points.

Stop rules: the guard's own; one attempt per arm; if the guard kills an arm, report and do not retry without Hamed.

## 4. Method

`docs/v1017-hotlist-ab/run-all.sh`: ABBA, arms A1 (on), B1 (off), B2 (off), A2 (on); quiet gate (no other runtime, 5-minute load under
3.5) before each, `settle.sh --budget-gib 40 --timeout 300` between. Each arm (`arm.sh`): a manifest (`run_manifest.py`, workload
W3:hermes-first-turn), a fresh `./serve.sh` with its shipped 1017 defaults (internal bank, DI-1 on, budget 0, wired 80) under
`guarded_run.sh --budget-gib 48 --max-seconds 1500 --force`, `CACHALOT_SNAPSHOT_DIR=` so the block prefills cold, and for B arms
`CACHALOT_HOTLIST=`. Driver (`driver.py`): request 0 of `~/cachalot-dump-1017.jsonl` (system block of 22,640 characters, 21 tools,
the first user message), non-streaming, temperature 0, at most 160 new tokens, `/v1/stats` before and after. Scored from the server's
`[request]` line (prefill s, decode tok/s, miss/tok, hit %) and the stats difference. The OS page cache is not flushed (no sudo); ABBA
balances a linear drift, not a first-arm effect. Raw output in `benchmarks/results/hotlist-ab-1017/` (gitignored); results in HANDOFF.
Estimate: about 2.5-3 minutes an arm plus settle, 15-25 minutes in all.
