# The v270 Mac jobs: DI-1 at 22k, DI-1's long-context quality gate, the `act_quant(o)` port and the `encoding.py` revision (experiment record; predictions before any build or run)

Written 2026-10-28 against runtime 0.68.5, **before any code for these jobs is written and before any run**. Charter section 7.2 fields 1 to 3 are fixed here; this file is committed first and results go in `docs/V270-MAC-JOBS-RESULTS.md`. A prediction edited after its run is void. Hamed asked on 2026-10-28 ("ok, complete the mac related open tasks from 270"), chose all four jobs with their estimates (DI-1 22k timing about 20 minutes; the long-context quality gate about 30 minutes to build and 30 to run; the `act_quant(o)` port about 40 to build and 30 to run; the `encoding.py` diff, minutes) and ran `sudo sysctl iogpu.wired_limit_mb=88064` (read back 88064). The live Hermes session with the knob on and the Hermes compression choices are his and are not part of this record.

## 1. Baselines (recorded)

* **DI-1** (§18.218, §18.221; `docs/DI1-SPLIT-RESULTS.md`, `docs/DI1-BUILD-RESULTS.md`): the decoder half (layers 20 to 39) is 46.5 % of a cold 4,096-token prefill chunk's layer time; derived saving of an exact-tail encoder-only prefill on a 22.4k block 39 to 42 % of layer time. On 6,100 tokens the deferred prefill's logits differ from the full prefill's by KL 0.036 (a re-chunking control: 0.020), same top-1 and 16 greedy tokens; the whole process was 75 against 89 s. The 22k timing and any quality gate have not run.
* **`pareto.py` cannot see the knob as built.** Its teacher-forced texts prefill 384 tokens in one `prefill_tokens` call and its battery prompts are single-call chats; `CACHALOT_PREFILL_ENCODER_ONLY` changes only prompts of more than one call. A long-context mode is needed (section 2.2).
* **`act_quant(o)`** (§18.223): the DeepSeek reference added, on 2026-10-12, an FP8 activation quantisation of the attention output `o` before the grouped `wo_a` einsum, in the main and the DSpark attention; this runtime runs that einsum in bf16 without it. The reference code is not in the local checkpoint (revision `dba1be0`, 2026-09-10).
* **`encoding.py`** (§18.223): the 2026-10-16 revision keeps tool-result images in prompt order; the server loads `encoding/encoding.py` from the local checkpoint, the old revision.
* **Quality bands** (§18.58, §18.223): the short `pareto.py` noise arm (prefill in 96-token chunks), measured on MLX 0.32.2: dNLL interval half-width 0.0243 nats, KL mean 0.0144. On MLX 0.32.4 the reference changed by less than that band.

## 2. Methods

### 2.1 DI-1 at 22,000 tokens (timing and state)

`benchmarks/di1_tail_equality.py deferred` then `full`, `--tokens 22000 --decode 16`, one process each under `guarded_run.sh` (budget as the guard's preflight allows, 34 GiB as in the earlier DI-1 runs unless it refuses), the server's environment (internal 2-bit bank, page cache, wired 80, fast synch), `settle.sh` between. **The deferred arm runs first** (cold), so any page-cache warmth from the first process helps the full arm and the measured saving is, if anything, understated. Read `prefill_s` from each `meta.json`; `compare` the two states.

### 2.2 DI-1's long-context quality gate (`pareto.py --long`)

A base option `"encoder_only_supported"` is not needed; instead `pareto.py` gains, in `teacher_forced`, an arm option `"encoder_only": true` (each prefill call made with `encoder_only=True`, then `finish_decoder()`), and a base option `"battery": false` (skip the task battery, whose single-call prompts the knob does not touch). Arms file `benchmarks/pareto_arms.long.json`: three texts of the repository (prose from `docs/HANDOFF.md`, code from `src/cachalot/model/text_decode_runtime.py`, numeric markdown from `CHANGELOG.md`), **8,192 tokens prefilled in calls of 4,096**, then 192 teacher-forced decode steps each (576 positions). Arms: `ref` (full prefill, 4,096-token calls), `noise-chunk3000` (full prefill, 3,000-token calls: the band), `di1` (the knob's path). One process an arm, reference first, `settle.sh` between.

### 2.3 `act_quant(o)` behind a knob

Fetch the reference `inference/model.py` (and its kernel helpers) of `deepseek-ai/DeepSeek-V4.1-Flash` at the current revision into the session scratchpad, read the change, and port it behind `CACHALOT_ATTN_ACT_QUANT_O` (default off; off is bit-identical to today, tested). With the knob on, the attention output is quantised to FP8 (e4m3, per-block scale as the reference does) and dequantised before the `wo_a` einsum, wherever the runtime runs that einsum (`attention_sliding_window.py`, `attention_compressed.py`, `attention_layer0.py`, `attention_prefill_batched.py`, `dspark_draft.py`). Quality: the standard short `pareto.py` (384-token prefill, 192 teacher-forced positions on three texts, the 21-task battery) with arms `ref`, `noise-chunk96` and `actq` (knob on), reference first.

### 2.4 `encoding.py` revision

Fetch the 2026-10-16 `encoding/encoding.py` into the scratchpad, `diff` it against the checkpoint's, and render every request body of the recorded 142-request Hermes session (`benchmarks/results/hermes-live-di4/requests.jsonl`) to a prompt string with each revision; count identical renderings and characterise the differences. Nothing in the checkpoint is changed (Hamed's call).

## 3. Predictions

All **predicted**; probabilities mine.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| T-1 | DI-1 at 22k: `prefill_s` deferred over full | 0.55 to 0.66 (the derived 39 to 42 % of layer time, diluted by the non-layer part of a prefill) (0.60) | above 0.80 |
| T-2 | DI-1 at 22k: final logits | same top-1 and the 16 greedy tokens equal (0.70); KL(full to deferred) at most 0.05 (0.70) | a different first greedy token, or KL above 0.2 |
| L-1 | long gate: the band | `noise-chunk3000` not identical to `ref`; its KL mean between 0.003 and 0.05 (0.70) | identical to `ref` (no band: the gate cannot be read) |
| L-2 | long gate: DI-1's loss | `di1`'s dNLL interval contains 0 (0.75) | its interval lies wholly above 0 (a systematic loss) |
| L-3 | long gate: DI-1's divergence | `di1`'s KL mean at most 1.5 times the noise arm's (0.60); top-1 agreement with `ref` at least 0.90 (0.80) | KL mean above 3 times the noise arm's |
| A-1 | `act_quant(o)` off | bit-identical to the current code at every changed call site (a test on model shapes) (0.95) | any difference with the knob off |
| A-2 | `act_quant(o)` on, quality | `actq` dNLL between -0.02 and +0.02 nats with its interval containing 0, KL mean at most 2 times the noise arm's (0.65) | dNLL interval wholly above 0 |
| A-3 | `act_quant(o)` on, speed | the teacher-forced decode ms within ±5 % of `ref` (0.60) | more than 10 % slower |
| C-1 | `encoding.py` | the two revisions render every recorded request identically when it carries no image in a tool result (0.85); the session's requests with a tool-result image, if any, render differently (reported) | a request without a tool-result image renders differently |

## 4. What each outcome means

* **T-1, T-2 held:** the knob's saving at the size that matters is measured, not derived; with L-2 and L-3 held, the case for a default-on is Hamed's to weigh against the snapshot policy (§18.221). **L-2 falsified:** the knob loses quality systematically and stays off.
* **A-2 held:** porting the reference's FP8 step is quality-neutral on this gate; whether to turn it on (parity with the reference against a speed cost, A-3) is Hamed's call. **A-2 falsified:** the runtime's bf16 path is the better one on this checkpoint's weights here, and the port stays off.
* **C-1 held:** updating `encoding.py` changes only image tool results; Hamed decides whether to replace the checkpoint's file.

## 5. Stop rules and cost

One runtime process at a time, every run through `guarded_run.sh` from a script file (the guardian's process check matches a benchmark name on the launcher's command line), `settle.sh` between arms; stop on any guard kill, an arm crash or swap growth. No `src/` edit while a benchmark runs. Estimated machine time: DI-1 22k two processes about 20 minutes; long gate three arms about 25 minutes; short gate three arms about 30 minutes; the rest offline.
