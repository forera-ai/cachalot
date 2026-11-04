# Does MiniMax's decode miss substitution break tool-call arguments? Replay record

Written 2026-11-03 against runtime 0.80.0, before the run. Hamed: "MiniMax or GLM", then "MiniMax: tool-call replay". HANDOFF section
18.267 when scored.

## 1. Baseline (recorded)

- DeepSeek: under the decode miss budget 0 the Hermes Desktop turn wrote a non-JSON integer `limit` in 2 of 24 replays (4 of 34 pooled)
  and 0 of 24 with exact decode inside tool calls (18.255); `CACHALOT_DECODE_TOOL_EXACT` is DeepSeek-only (`serve.sh` default since
  0.77.0).
- MiniMax-M3 decodes under a different approximation, on by default since 0.43.0: a missing expert holding under 0.20 of its layer's
  routing weight is not read and the best resident of the next four takes its place (decode), and the same rule in prefill. Its 0.42.0
  battery (36 thinking-on and tool-call tasks) passed 33 of 36 with the substitution against 31 of 36 exact (18.22-18.23); no Hermes
  integer-argument check was ever run on MiniMax. A bare `<tool_call>` without its namespace token was fixed in 0.39.2.

## 2. Hypothesis and alternative

H: the substitution damages tool-call arguments as DeepSeek's budget did, so a tool-exact port to MiniMax would pay. Alternative: a
substituted expert keeps the routing mass (no weight is dropped) and only experts under 0.20 of a layer's weight are affected, so
arguments stay well formed; no port needed.

## 3. Method

`run-all.sh`: two fresh `./serve-minimax.sh` servers (68 GiB under the governor, mirror on, no disk snapshots), arm A shipped, arm B
exact (`CACHALOT_MINIMAX_MISS_DROP=0 CACHALOT_MINIMAX_PREFILL_MISS_DROP=0`), settle between. `driver.py 12`: the three recorded Desktop
requests of 18.255 (P1-P3, pinned by dump line and messages sha256) rotated, four each, as Hermes sent them (temperature 0.7, 21 tools,
reasoning effort none), non-streaming, `max_tokens` 1,500 (MiniMax may reason first). `analyse.py` uses 18.255's classifier: valid
(a `search_files` call with an integer or absent `limit`), nonjson, other-call (another tool), no-call (text only, a lost call),
error. Each arm's manifest records its environment; the driver records `/v1/stats` at the end.

**Amended before any reply was produced (2026-11-03, Hamed: "Rerun at 48 GiB").** The first attempt at the shipped 68 GiB was stopped by
the guardian 11 s after start (swap 57 to 1,467 MB under pressure 2; 74.8 GiB available against 97 GiB needed, with desktop apps and the
screensaver running); no reply was produced. Both arms now run at `--expert-budget-gib 48`. A smaller budget means more missing
experts, so the shipped arm substitutes more often than at 68 GiB: a harsher test of the same question. A clean shipped arm here covers
68 GiB too; a broken one would need a 68 GiB check before any port is decided.

## 4. Predictions (registered before the run)

- **M-1.** 24 replies, no error; every reply makes a tool call or (at most 2 an arm) answers in text.
- **M-2.** Shipped: non-JSON arguments plus lost calls 0-2 of 12.
- **M-3.** Exact: 0-1 of 12.
- **M-4.** Decode 2.5-4.5 tok/s shipped, exact 10-30 % slower on the cold server (at 48 GiB both nearer the low end).
- **Decision rule.** Port tool-exact to MiniMax only if shipped has 3 or more bad replies and exact at most 1; otherwise record that
  MiniMax's tool arguments do not need it (with this sample's bound).

Runtime estimate: about 70 minutes (two cold MiniMax servers, three prompt prefills of about 3 minutes each at 80-100 tok/s, 12 replies
of 30-90 s, a settle). Needs `iogpu.wired_limit_mb` at 88064, the X10Pro mounted, no other runtime.

## 5. Results (measured 2026-11-04, commit f92c80c, 48 GiB)

**Run.** `run-all.sh` 11:26-12:04, sysctl 88064, mirror on, no other runtime; arms 991 s and 952 s, peak wired 78.5 / 77.8 GiB, peak
swap 125 / 794 MB (1,467 MB of stale swap from the first attempt; the guardian measures growth). Arm B's manifest records both drops at
0, arm A's leaves them at the 0.20 default; MiniMax's `/v1/stats` has no substitution counter (its `skipped_experts` is the DeepSeek
store's and read 0 in both), so the manifests are the arm-state proof. `report.txt` holds `analyse.py`'s output.

| arm | replies | non-JSON / lost call / error | calls |
|---|---|---|---|
| A, shipped (drop 0.20, substitution 4) | 12 | **0** | `terminal` 11 (`ls -la ~/Desktop`, 9 with `2>&1 | head -50`), `tool_call` 1 (Hermes's dispatcher, a deferred tool) |
| B, exact | 12 | **0** | `terminal` 11, `search_files` 1 (no `limit`) |

Every argument string parsed as JSON. Unlike DeepSeek, MiniMax chose `terminal` for the Desktop listing in 23 of 24 replies and never
wrote an integer argument, so the integer-as-word failure of 18.243 was not exercised. Each request re-prefilled its 15.6-15.8k-token
prompt (about 70 s; three rotating prompts did not stay in the prefix cache). Decode: shipped 5.25-11.69 tok/s, exact 4.32-5.97 tok/s
on 30-69-token replies.

**Scored.**
- **M-1 held:** 24 replies, no error, every reply a tool call.
- **M-2 held:** shipped 0 of 12 bad.
- **M-3 held:** exact 0 of 12 bad.
- **M-4 missed (high):** shipped decode 5.25-11.69 tok/s (predicted 2.5-4.5) on very short replies with the experts warm after each
  prefill; exact 4.32-5.97, slower as predicted.
- **Decision rule:** shipped had fewer than 3 bad replies: no tool-exact port to MiniMax. Bound: with 0 of 12 at a harsher budget than
  shipped, the bad rate on this turn is under about 22 % (one-sided 95 %), and integer arguments are untested.

## 6. Kind

Research: predicted and measured (M-1 to M-3 held, M-4 missed high); the decision rule applied. No default changed.
