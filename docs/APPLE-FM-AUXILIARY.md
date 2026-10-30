# Apple Foundation Models as Cachalot's auxiliary engine

Written 2026-10-29, runtime 0.69.0. Measurements are in `docs/APPLE-FM-RESULTS.md`, predictions in
`docs/APPLE-FM-RECORD.md`, the concurrency question in `docs/APPLE-FM-HETEROGENEOUS.md`, speculative decoding in
`docs/APPLE-FM-SPECULATIVE-FEASIBILITY.md`.

## 1. What this is, in one paragraph

`src/cachalot/auxiliary/` lets Cachalot ask a small model for short structured jobs (classify a request, extract
fields, condense a document) while the large target model (DeepSeek V4.1 Flash, GLM-5.3, MiniMax-M3) keeps the GPU.
The first backend is Apple's on-device system model (macOS 27.2 reports "AFM 3 Core Advanced", context 8,192 tokens),
which Hamed's Instruments run places on the Neural Engine. Everything is **off by default**, nothing in the target
model's load, prefill, decode or sampling path imports the package, and every task falls back to a deterministic
heuristic. It does not run Cachalot's models on the Neural Engine: the Foundation Models framework can host other
models, but that is not a way to execute DeepSeek or GLM on the ANE, and nothing here tries.

## 2. Backend choice

| option | verdict | why |
|---|---|---|
| A. Apple's Python SDK `apple-fm-sdk` (0.2.1, Apache-2.0) | **chosen** | Cachalot is Python; the SDK is ctypes over Swift C bindings of FoundationModels.framework and covers availability (with reasons), instructions, greedy sampling, streaming, JSON-schema guided generation, `token_count`, `context_size`, cancellation (FMTaskCancel) and typed errors. It constructs only `SystemLanguageModel`, the on-device model: there is no Private Cloud Compute path to fall into. Runtime imports are the standard library only. |
| B. A Swift bridge of our own | not needed | the SDK already is that bridge; a second one would duplicate it |
| C. `/usr/bin/fm` | diagnostic only | a process per call, text parsing, and `--model pcc` one flag away; `fm serve` is a persistent HTTP server we would have to supervise. Useful to check the machine (`fm models`), not as a backend. |

Install (an optional extra; built from source with Xcode's Swift toolchain, needs macOS 26+ and full Xcode):

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ~/venvs/deepseek-v41/bin/pip install --no-deps 'apple-fm-sdk==0.2.1'

`--no-deps` keeps the SDK's declared `build` and `setuptools` requirements (needed only to build it) out of the
runtime venv; it imports nothing outside the standard library at run time.

## 3. Where it sits in Cachalot

What the repository has (read 2026-10-29): a single-flight `server.engine.Engine` (one request at a time on a lock,
streamed through a worker thread and cancelled with a `threading.Event`), an OpenAI-compatible FastAPI server, a
frozen-dataclass `RuntimeConfig` overridden by `CACHALOT_*` environment variables, three model families each with its
own runtime, and benchmark instruments under `benchmarks/` writing to `benchmarks/results/`. There was no auxiliary or
multi-backend abstraction to extend, and the server's request path is performance-sensitive and measured.

So the integration boundary is a **separate package with no caller in the serving path**:

```
cachalot.auxiliary
  config.AuxiliaryConfig        CACHALOT_AUX_* (same convention as RuntimeConfig); all off
  provider.AuxiliaryProvider    blocking API + cancel Event, like Engine.stream_chat; Capabilities
  apple_fm.AppleFoundationModelsProvider   the SDK on a private asyncio loop thread
  isolated.IsolatedProvider     the same API, the provider in ONE child process (default)
  worker                        the child: JSON lines on stdin/stdout, exits when stdin closes
  tasks                         classify / extract / compress: typed I/O, schemas, validators, heuristics
  schema.validate               deterministic JSON-schema subset validator (every result passes it)
  engine.AuxiliaryEngine        config + provider + per-feature gate + bounded counters
  levers                        LeverSpec registry A-G, CompressionLever, RoutingLever
  cli                           `cachalot aux ...`
```

The target model never waits on the auxiliary: no server or engine code calls it, so its absence, failure or
slowness cannot affect serving. Wiring a lever into the request path is a separate, measured change (section 7).

**Why a child process by default.** The SDK loads Swift bindings into the calling process; a fault there would take a
server with a loaded 100+ GiB model down with it. In a child it costs one restart. The parent enforces its own deadline
(request timeout + 3 s) and kills a child that misses it; restarts are capped (3 in 60 s, then `crash_loop`); the child
exits when its stdin closes, so it cannot outlive the parent; there is at most one child. Measured cost: none
(0.315 s through the child against 0.333 s inline, median of 7). Model inference itself runs in a system service
(`TGOnDeviceInferenceProviderService`), not in either process.

## 4. Provider contract

* `capabilities()` reports only what the SDK states or the provider verified: `local_only` is always true;
  `logits`, `tokenizer_access`, `kv_access` are false; `context_tokens` is the model's `context_size`.
* Unavailability carries a reason: `sdk_missing`, `unsupported_os`, `unsupported_hardware`, `init_failed`,
  `apple_intelligence_not_enabled`, `device_not_eligible`, `model_not_ready`, `assets_unavailable`, `closed`,
  `crash_loop`, `worker_start_failed`.
* Every request is checked against the context window first (prompt + instructions + schema + response cap + 64):
  the SDK reports an oversized prompt only as a generic status 255 after seconds of work.
* Timeouts are asyncio deadlines that cancel the native task; a cancel `Event` does the same; both measured to return
  within about 0.5 s and leave the session usable.
* Admission is bounded: `max_concurrent_requests` (1; the system model serializes anyway: two parallel sessions took
  3.56 s against 1.57 s for one) plus `queue_limit` (4); beyond that `AuxBusy`, never an unbounded queue.
* Errors map to `cachalot.auxiliary.errors`: `AuxUnavailable`, `AuxDisabled`, `AuxTimeout`, `AuxCancelled`, `AuxBusy`,
  `AuxContextOverflow`, `AuxRefused` (guardrails and refusals; SDK 0.2.1 reports a refusal as status 255 with
  "The model refused to answer"), `AuxInvalidOutput`, `AuxProviderError`.
* Each request opens a fresh `LanguageModelSession`, so no transcript grows across requests.

## 5. Configuration

| variable | default | meaning |
|---|---|---|
| `CACHALOT_AUX_ENABLED` | 0 | master switch; every feature also needs its own flag |
| `CACHALOT_AUX_PROVIDER` | `apple_foundation_models` | or `deterministic` (heuristics only) |
| `CACHALOT_AUX_MODEL` | `system` | anything else is rejected; there is no cloud model setting |
| `CACHALOT_AUX_ISOLATION` | `process` | or `inline` |
| `CACHALOT_AUX_TIMEOUT_MS` | 10000 | per request |
| `CACHALOT_AUX_MAX_CONCURRENT_REQUESTS` / `CACHALOT_AUX_QUEUE_LIMIT` | 1 / 4 | admission bound |
| `CACHALOT_AUX_USE_CASE` | `general` | or `content_tagging` (the system model's tagging adapter) |
| `CACHALOT_AUX_CLASSIFICATION`, `_STRUCTURED_EXTRACTION`, `_CONTEXT_COMPRESSION`, `_SEMANTIC_VALIDATION`, `_ADAPTIVE_ROUTING` | 0 | feature flags |
| `CACHALOT_AUX_COMPRESSION_MIN_TOKENS` / `_MAX_RATIO` / `_MAX_UNGROUNDED` | 2000 / 0.6 / 0.1 | compression acceptance |

Disable completely: leave `CACHALOT_AUX_ENABLED` unset (the default), or uninstall the SDK; either way every task
method raises `AuxDisabled` or falls back, and the server never imports the package.

## 6. Tasks and levers

Every task: typed request and result, a JSON schema, a deterministic validator on every result, a heuristic that is
both fallback and baseline, `source` ("model" or "heuristic") and `fallback_reason` on the result. No task asks the
model for a confidence value, and none trusts the model's self-reports: in this session the system model's `found`
flags were false for values it had copied correctly, and it filled absent fields with inventions.

<a id="lever-a"></a>**Lever A, classification** (`tasks.classify`). Labels `task_type`, `complexity`, `needs_tools`,
`needs_large_model`; the route is decided by code, not the model: arithmetic is `deterministic` (no model call);
`auxiliary` only when the model calls it simple, safe-typed, tool-free *and* the heuristic agrees; any disagreement is
`uncertain` (answered by the target); any model failure routes to the target. Measured: 0.82 s a call, route accuracy
0.90 (registered run; 0.95 after three fixes found by that run, on the same set), false routing 3 then 1 of 20.

<a id="lever-b"></a>**Lever B, source-grounded compression** (`tasks.compress`). Paragraph segments `s1..sN`; segments
with constraint words (must, never, do not, only, at least, ...) and fenced code are kept **verbatim** and never sent
for rewriting; the rest go in chunks to the model, which returns facts citing a segment id; a fact is kept only if its
segment exists and contains every number and identifier the fact states; a failed chunk stays verbatim; the result is
rejected (original returned) above `max_ratio` or `max_ungrounded`. **Measured: it loses.** 10.9-15.3 s per 1,000 input
tokens against DeepSeek's 99 tok/s cold prefill (net -4.9 to -16 s per 1,000), and on the real results document it kept
2 of 11 critical numbers while passing every gate: grounding detects invented facts, not omitted ones. Stays off; see
the results document for what would reopen it.

<a id="lever-c"></a>**Lever C, routing** (`levers.RoutingLever`). Returns a route decision; it never answers a request
and nothing calls it. It needs both `adaptive_routing` and `classification`; anything else returns "target".

**Command-review offload** (`approval.py`, 0.70.0): recognizes Hermes's command-safety review and would answer only an
exact APPROVE from the system model. Not wired into the server: the system model approved 2 of 36 dangerous commands on
the gate (`benchmarks/apple_fm_approval_gate.py`, `docs/APPLE-FM-APPROVAL-RESULTS.md`).

**Structured extraction** (`tasks.extract`). Flat typed schema; a value is accepted only if it occurs in the text
(`grounded_in`: case- and whitespace-insensitive for strings, numeric for numbers); "" and 0 are the instructed
"not stated" placeholders, so a real zero reads as missing. Measured 0.958 field accuracy against the heuristic's
0.375; the one remaining invention copied a value present in the text into the wrong field, which grounding cannot see.

**Lever D (semantic cache):** no-go on the measured Hermes session (one recoverable eviction, which a 3 GiB budget fixes;
`docs/APPLE-FM-E2-E3-RESULTS.md`). **Lever E (ANE/GPU scheduling):** measured, no scheduler needed (`docs/APPLE-FM-HETEROGENEOUS.md`). **Lever F
(speculative decoding):** no-go, `docs/APPLE-FM-SPECULATIVE-FEASIBILITY.md`.

<a id="lever-g"></a>**Lever G, semantic validation:** research only; the flag exists and nothing reads it. Deterministic
validators (JSON, schema, types, exact constraints) stay deterministic; an auxiliary opinion may only ever be advisory.
Measuring it needs a labelled set of target answers with known defects, which does not exist yet.

`levers.LEVERS` declares each lever's requirements, workloads, overhead, failure behaviour, quality constraint, metrics
and `changes_output`, so a later wiring cannot pretend a lever is semantics-neutral.

## 7. Developer commands

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli aux check
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli aux respond --stream "Name two planets."
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli aux classify "Refactor resident_store.py to use LFU"
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli aux extract --field host --field port:integer "Serve on host 127.0.0.1, port 8080."
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli aux compress --file docs/CUDA-PIN-IDLE-RESULTS.md --min-tokens 300
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m pytest -q tests/test_auxiliary_*.py
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m pytest -q --apple-fm tests/test_auxiliary_apple_fm_live.py
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/apple_fm_bench.py overhead|classify|extract|compress|memory|contention

`cachalot aux <task>` is the opt-in for that one command: it enables the layer and that task's feature for the
command's own engine. In code: `AuxiliaryEngine(AuxiliaryConfig(enabled=True, classification=True))`.

Before wiring any lever into the server: decide the call site (before `Engine.stream_chat` takes its lock, so the
target never waits behind the auxiliary), add a server stat for it, A/B the request path, and get Hamed's go for
anything that changes what the target sees (`changes_output`).

## 8. Safety properties, and how each is held

| property | held by |
|---|---|
| no cloud fallback | the SDK has no PCC path; `model` must be `system`; `local_only` is not configurable |
| no private APIs, no asset access | only the public SDK; no model files are read |
| no subprocess accumulation | one child per provider, killed on deadline, exits on stdin EOF, restart cap |
| bounded queues and memory | admission bound; fresh session per request; fixed-size counters |
| no deadlocks with the target | no shared lock with `Engine`; nothing in the serving path calls the package |
| no silent context loss | compression keeps constraints and code verbatim, keeps failed chunks verbatim, rejects on ratio and grounding, and is off (and measured unsafe for omissions) |
| no change to target sampling | the package never touches `SamplingParams` or the runtime |
| model output untrusted | schema validation, grounding, routing by code, no self-reported confidence |

## 9. Limits

One Mac, one macOS (27.2), one SDK version (0.2.1, "Development Status :: 3 - Alpha"). The labelled sets are small and
written by the agent before the runs; the fixes after the first classify and extract runs were tuned on those same
sets. The service's memory is visible only as RSS without root `footprint`. Licensing: the SDK is Apache-2.0; the
system model's terms are Apple's (`fm license`); nothing model-related is redistributed by this repository.
