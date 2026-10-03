# Speed research, 2026-10-03: what the literature and our own record say is left

Written for the next session. Hamed's request: check the online literature and projects for the same problem (streaming a large MoE from SSD on one Mac), find changes we missed,
review our own process scientifically, and find untested ways to gain speed "by much", with speed above quality in priority. Nothing in `src/` changed; this is analysis plus a measured
regression from this session's logs. Every number marked *measured* comes from a log or script of ours; every number marked *estimate* is arithmetic from measured inputs and says what must
be measured to confirm it; claims about papers are what their abstracts, model cards and fetched summaries say (arXiv identifiers and Hugging Face repositories are given so they can be
re-read; none of the papers' code was run).

## 1. Where the time goes now

**Model sizes against memory (measured from configs and banks).**

| | layers x experts | top-k | expert size | all experts | expert budget | slots | resident share | decode hit rate |
|---|---|---|---|---|---|---|---|---|
| MiniMax-M3 (3-bit) | 60 x 128 = 7,680 | 4 | 21.1 MiB | 158 GiB | 68 GiB | 3,277 | 43 % | 94-95 % chat, ~90 % agent |
| GLM-5.3-Flash (4-bit) | 42 x 288 = 12,096 | 8 | 13.5 MiB | 159 GiB | 52 GiB | 3,944 | 33 % | 66-70 % |

**MiniMax, this session (measured).** Fourteen replies of one Swift prompt through `./serve-minimax.sh` (default settings, temperature 0.7 and 0): 114 ms a token on average (107-127), 10.9
misses a token (9.5-12.8). A least-squares fit of ms a token against misses a token gives `ms = 77 + 3.42 x misses` (r = 0.65, n = 14, narrow range). The slope, 3.4 ms, equals the raw read
time of one expert (21.1 MiB at 6.5 GiB/s = 3.2 ms): **in the observed range a miss costs its full read time, so the prefetch and the GPU-side select do not hide reads from the token**, and
`decode_wait_ms_per_miss` (1.2 ms in the user's chat `/stats`) understates the real cost by about three times because reads also slow the GPU part of the token (§18.4: ~25 ms). The
intercept, 77 ms, is the part of the token that does not scale with misses. §18.19 measured the all-hit floor at 35-43 ms. **Either the floor moved since 0.38.0, or the 77 is an
extrapolation artefact of a narrow range, or the screensaver (it ran during the replies; ~13 % per §18.16) and other apps account for part of it. This is the single most valuable open
measurement: 77 against 43 is up to 34 ms of a 114 ms token, ~30 %.**

**GLM (measured, earlier sessions).** 375-400 ms a token at ~100-110 misses a token (8 x 42 layer calls at ~30 % misses), 70-75 % of it store wait, the drive at 5.2 GiB/s (84 % busy);
`2.6 ms x misses + ~105 ms` reproduces the figure. GLM is still read-bound; MiniMax, at chat contexts, is about one third misses.

**Speed against hit rate (estimate; the model is `token = floor + cost_per_miss x misses`, misses = layer calls x top-k x (1 - hit rate)).**

GLM (336 expert uses a token, floor ~105 ms, 2.6 ms a miss, which reproduces the measured ~365 ms at 70 %):

| decode hit rate | misses/token | ms/token | tok/s | speed-up |
|---|---|---|---|---|
| 70 % (now) | 100 | 367 | 2.7 | 1.00 |
| 80 % | 67 | 279 | 3.6 | 1.32 |
| 85 % | 50 | 235 | 4.3 | 1.56 |
| 90 % | 34 | 193 | 5.2 | 1.9 |
| 95 % | 17 | 149 | 6.7 | 2.5 |
| 99 % | 3 | 114 | 8.8 | 3.2 |

MiniMax (240 expert uses a token, 3.4 ms a miss; the floor is the open question, so both values are shown):

| decode hit rate | misses/token | ms/token if floor 77 | ms/token if floor 43 | speed-up (floor 77 / floor 43) |
|---|---|---|---|---|
| 94.5 % (now) | 13.2 | 122 | 88 | 1.00 / 1.00 |
| 96 % | 9.6 | 110 | 76 | 1.11 / 1.16 |
| 97 % | 7.2 | 101 | 68 | 1.21 / 1.29 |
| 98 % | 4.8 | 93 | 59 | 1.31 / 1.49 |
| 99 % | 2.4 | 85 | 51 | 1.44 / 1.73 |
| 99.5 % | 1.2 | 81 | 47 | 1.5 / 1.9 |

The conclusion the tables force: **for GLM every route to a higher hit rate is worth up to 2-3x; for MiniMax a higher hit rate is worth 1.1-1.5x if the floor is 77 ms and up to 1.9x if it is 43 ms, so MiniMax's first job is to find out which floor is real.**

## 2. Review of our own process

**What produced the gains and what produced closures.** Of the ~80 levers in the closed list, the ones that moved a token by more than 10 % all changed the *workload* the drive and the GPU
saw, not the code that ran it: the contiguous bank (bytes and pieces per expert), prefetch from the layer's own routing (when reads start), miss substitution (which experts are read at all),
slot images and bias codes (bytes per expert, so more experts in memory), the memory governor and working-set sysctl (how many experts stay). The levers that kept the workload and
optimised the machine (kernel variants, fusions, queue tricks, scheduling) each landed at 0-5 % and several below the measurement noise. The closed list is the evidence: kernel work is
exhausted on this hardware, so a new gain of "much" has to come from changing the workload again. Each such lever changes outputs, so each needs a quality gate.

**The measurement floor.** Process-to-process drift is +-10 % (§18.3, §18.35), the screensaver costs ~13 % (§18.16), the visible Hermes window up to 30 % (§18.18), memory pressure doubles the
non-read part (§18.1). Any lever under ~15 % is invisible without per-token alternation inside one process. So the remaining plan favours few, large, structural levers and per-token
alternation for the rest. This session saw it again: the hybrid GLM server fell to 0.8 tok/s at memory pressure level 9 (screensaver on, swap near full) and recovered to 2.3 tok/s at a smaller budget.

**A metric that misled us.** `decode_wait_ms_per_miss` (the time the host waits per miss) is 1.2 ms for MiniMax, which reads as "a miss is cheap"; the fitted total cost is 3.4 ms. A
regression of ms a token on misses a token over many requests, as in section 1, is the honest exchange rate and should be printed in the stats.

**No common currency for approximations.** Miss substitution (MiniMax), top-k drops, REAP-style pruning, bit-sliced reads and a lower-bit tier all trade quality for misses, and each was
judged in its own session with its own texts. A single *Pareto harness* (misses a token against NLL/KL and a checkable battery on one fixed set of Hermes-shaped traces) lets every
approximation compete on one curve and be combined, and is the prerequisite for the levers below.

**Selection effects in what was tried.** The documented history contains no cache-aware *saliency* (only router weight), no per-layer thresholds, no pruning of any kind, no substitution for GLM
or DeepSeek, no mixed-precision reads, and nothing that changes the model's expert set. They are the gaps in the table of section 4.

## 3. What the literature and public projects say (and our status)

| Idea (source) | What it claims | Status in Cachalot |
|---|---|---|
| **REAP**, router-weighted expert activation pruning ([arXiv 2510.13999](https://arxiv.org/abs/2510.13999), ICLR 2026) | Saliency = mean over active tokens of gate x ||expert output||; one-shot, layer-wise; 25 % pruning loses ~0.1-0.2 % on math and non-agentic coding, 50 % ~1-2 % on Qwen3-Coder-480B/Kimi-K2 (160-384 experts), but **GLM-4.5-Air (128 experts) -11 % coding, -10 % creative writing at 50 %**, agentic tool use -5 to -6 %, multiple choice -12 %; calibration domain matters (C4 calibration collapsed code to 0 %) | **not tried**; we only have a dynamic cousin (MiniMax substitution) |
| Public REAP checkpoints | GLM-5.3-Flash: `pipenetwork/...-REAP25/37/50-MLX` (216/181/144 of 288 experts per layer, 96-118 GB, wikitext-2 + code calibration, **perplexity 3.46 -> 4.88 (37 %) and 6.08 (50 %) on wikitext-2, +0.34 and +0.56 nats a token**); MiniMax-M3: `JANGQ-AI/MiniMax-M3-REAP22/32-Coder` (87 of 128 experts, calibrated on an agentic-coder set, JANG format), `bullerwins/...REAP25-MXFP8`, `sparkarena/...REAP25/50-NVFP4` | no mask files are published; **the keep-lists can be recovered** because a pruned expert's scales and biases equal ours byte for byte (§18.49): range-read one small tensor per layer (~1 GB) and match fingerprints |
| **HOBBIT** / mixed-precision expert offloading ([summary](https://www.themoonlight.io/en/review/hobbit-a-mixed-precision-expert-offloading-system-for-fast-moe-inference)) | on a miss, load a low-precision (int4/int2) copy for the less important experts; up to 4x lower load latency, small accuracy loss | **not tried** (we substitute or read full precision, nothing between) |
| **SliceMoE**, bit-sliced expert caching ([arXiv 2512.12990](https://arxiv.org/abs/2512.12990)) | split each expert into MSB and LSB slices (nested "Matryoshka" quantisation); cache MSB for all, LSB only for critical ones; MSB-only compute for non-critical experts; 1.6-1.8x decode on DeepSeek-V2-Lite and Qwen1.5-MoE, perplexity ~unchanged; predictive cache warm-up from prefill hotness alone 1.96x | **not tried**; our bank is already a custom slot image, so a plane layout is possible |
| **MoE-CORE** ([arXiv 2610.01950](https://arxiv.org/abs/2610.01950)) on DeepSeek-V4-Flash and GLM-5.2 | per-layer cache allocation from hit-rate curves, domain-informed cache initialisation, a routing-history-aware replacement policy, cross-layer prefetch, optional score-based substitution; TPOT 38-45 ms against 62-71 ms (1.39-1.75x); exact hit rate 47 % -> 96 % across 12 -> 120 slots a layer; substitution lifts availability 92.43 % -> 99.97 % | prefetch and substitution done; **per-layer allocation, domain-informed init and RHAC-style replacement not done** (SLRU/LFU/TinyLFU were measured worse, §18.1/18.5, but not RHAC or per-layer quotas) |
| **Self-speculative decoding with residency-aware drafts**: DraftExpert ([2607.24434](https://arxiv.org/abs/2607.24434)), AcceptMoE ([2608.02989](https://arxiv.org/abs/2608.02989)), MoE-Spec ([2602.16052](https://arxiv.org/abs/2602.16052)), EcoSpec ([2607.12696](https://arxiv.org/abs/2607.12696)), "The Limits of Speculation" ([2609.22156](https://arxiv.org/abs/2609.22156)) | draft with only resident experts (no reads), then verify with a budget on the verifier's non-resident experts | closed on arithmetic for us (§18.7, §18.40: a width-2 verify reads ~1.75x the experts); the papers add *verifier-side expert budgets*; re-price only after the Pareto harness exists (section 4, item N8) |
| **flash-moe** (danveloper, Qwen3.5-397B-A17B, MacBook M3 Max 48 GB, 4.4 tok/s; [repo](https://github.com/danveloper/flash-moe)) | pread() of 6.75 MiB experts (4 of 512), trusts the OS page cache (71 % hit), hand-written Metal, deferred GPU expert compute | we are ahead (own cache, governor, prefetch, bank); the page-cache-only approach is what we replaced |
| **Runtime expert quantisation and dropping** ([MoE-Infinity issue 234](https://github.com/EfficientMoE/MoE-Infinity/issues/234)), dynamic expert quantisation ([2511.15015](https://arxiv.org/abs/2511.15015)) | same family as HOBBIT/SliceMoE | not tried |
| **M5 neural accelerators in MLX** ([BaseRT 2607.19438](https://arxiv.org/abs/2607.19438), MLX PRs on NAX gather_qmm) | 3-4x faster prefill on M5, small-batch quantised matmul on NAX; MLX 0.32.2 does not route small-row calls to NAX | not applicable to the M3 Ultra; relevant only to a hardware change |

I could not retrieve full text for AcceptMoE, EcoSpec and "The Limits of Speculation", only abstracts and summaries, and several results are on GPUs with PCIe offloading, not unified-memory
SSD streaming; treat their speed-ups as direction, not as predictions.

## 4. Candidate levers, ranked

"Gain" is an estimate from section 1's model; "evidence needed" says what turns it into a measurement. Output-changing levers need Hamed's decision to become a default.

| # | Lever | Why it might work | Estimated gain | Cost | Quality risk | Evidence needed first |
|---|---|---|---|---|---|---|
| **N1** | **Re-anatomise the current MiniMax token**: where do 114 ms go at ~11 misses? Per-layer GPU timeline at 0 misses (warm set loaded, repeated prompt), then with misses; check the screensaver and other-app tax | the fit says a 77 ms floor against §18.19's 35-43 ms; if the floor is 43, ~30 % of the token is unexplained | up to 1.3x MiniMax (zero if the 77 is an artefact) | 0.5 session | none | the measurement itself |
| **N2** | **Miss substitution for GLM** (and DeepSeek): the rule `drop a missing expert whose router share is under tau, use the best resident of the next ranks`, with tau relative to top-8's flatter weights (about 0.06-0.10) | GLM has no substitution; MiniMax's removed ~40 % of misses and gained 9-13 % at a quality inside the noise | 1.2-1.45x GLM (30-45 % fewer reads at a read-bound 70 % of the token) | 1 session; code exists in `minimax/` | GLM already fragile; needs the harness | KL/NLL on 3 texts + the C# replay + a tool battery |
| **N3** | **A saliency-aware substitution rule** (REAP score: gate x mean ||expert output|| from a calibration pass, instead of gate alone), and per-layer tau from per-layer sensitivity | skips experts whose outputs are small even at a high gate and keeps small-gate experts with large outputs; the same misses removed at a lower KL, or more misses at the same KL | +10-25 % more misses removed than N2 at equal quality | 1 session after the harness | lower than N2 at equal speed | the harness; a calibration pass on Hermes traces |
| **N4** | **REAP as a router mask over our existing banks** (no new weights): recover keep-lists from the public checkpoints by fingerprinting expert scales (~1 GB of range reads), or compute our own saliency from Hermes dumps (agentic calibration; the public ones used wikitext or a coder set); mask pruned experts in the router and renormalise | the cache never sees the pruned experts; GLM resident share 33 % -> 44 / 52 / 65 % at 25 / 37 / 50 %; MiniMax 43 % -> 61 % at 32 % pruned | GLM 1.3-2x (hit 80-90 %, section 1 table); MiniMax 1.1-1.3x unless N1 finds a lower floor | 2 sessions (mask loader, saliency or fingerprints, gate) | **large**: published REAP37/50 for GLM cost +0.34 / +0.56 nats on wikitext (perplexity +41 % / +76 %); GLM-4.5-Air (also 128 experts) -11 % coding at 50 %; MiniMax has only 128 experts and top-4 | the harness on Hermes text; start at 25 % |
| **N5** | **S1c for GLM after N2/N4**: its design rule was "reopen above ~90 % hits" | at a 90 % hit rate about 66 % of layers are all-hit instead of 16 % | +8-15 % on top of N2/N4 | 2 sessions | none (bit-identical) | the hit rate from N2/N4 |
| **N6** | **Bit-sliced experts** (MSB/LSB planes, SliceMoE) for non-critical misses: read 2/3 of an expert's bytes (3-bit -> 2 MSB bits, doubled scale) when its share is mid-range, full for critical ones | between "substitute" and "read in full" there is nothing today | 10-20 % of miss bytes | 3 sessions (bank planes, Metal kernel, quality gate) | medium; MSB-only is 2-bit-quality | a plane-layout feasibility check on the 3-bit packing |
| N7 | Per-layer cache quotas and a routing-history replacement policy (MoE-CORE) | deficits at specific layers; our LRU is global | 3-8 % | 1 session | none (cache policy) | per-layer miss curves from a recorded trace |
| N8 | Re-price speculative decoding with a verifier-side expert budget | AcceptMoE/MoE-Spec/DraftExpert claim gains when verification is capped | probably still closed (a width-2 verify reads 1.75x; capping reads is substitution again) | 1 session of arithmetic first | medium | per-position union sizes on the recorded traces |
| N9 | Domain-informed warm sets (code, prose, tools) selected from the system prompt | MoE-CORE: domain rankings differ strongly | few % on first turns | 0.5 session | none | resident-set snapshots per domain |
| N10 | Tiering by model: a fully resident small MoE (for example a 100B-class MoE at ~4 bits) for subagent and tool turns, MiniMax/GLM for hard turns | no misses at all, 30+ tok/s class | 3-10x on those turns | a routing policy and memory discipline (one runtime at a time) | a different model | Hamed's call |

Hardware routes, listed so they are not rediscovered: a Thunderbolt 5 NVMe mirror (M19, halted by Hamed 2026-10-01, do not raise); an M5-class Mac (NAX, 3-4x prefill; MLX needs routing of small-row calls).

**A combined path for GLM** (estimates): N2 (-35 % misses, 100 -> 65: ~280 ms) -> N4 at 25-37 % pruning (hit 70 -> ~85 % before substitution; together ~90 %: ~195 ms) -> N5 (-15-25 ms): **2.7 -> ~5.5 tok/s, 2x**,
if quality survives each gate. **For MiniMax:** N1 first; N3 (+10-20 % fewer misses) and N4 at 22-32 % with agentic calibration on top: 1.15-1.5x depending on N1.

## 5. Experiments, in order, with stop rules

1. **N1 (MiniMax floor).** Alternate zero-miss and normal tokens (resident warm set, a repeated 300-token prompt, `TF_ALTERNATE`-style per-token alternation) with the display's state recorded. Stop
   when the 114 ms split into floor and miss parts is known to +-5 ms. If the floor is ~43 ms, find where the 34 ms went (per-layer timeline, `sample()` host work, pressure watcher, substitution bookkeeping, stats).
2. **The Pareto harness.** One fixed set of traces (Hermes dumps already exist: `/tmp/cachalot-requests-*.jsonl`, GLM `glm-0.51`), teacher-forced NLL and KL per text, the checkable battery (the C# replay
   with the library project of §18.49, tool-call tasks), and a routing-trace simulator that replays recorded routing through a cache model (slots, policy, substitution, mask) to predict hit
   rates *before* any weights move. Output: misses a token against quality for each approximation. This also resolves the section 1 table's hit-rate guesses.
3. **N2 on GLM** at three thresholds, per-token alternation inside one process, then the harness. Stop when KL leaves the rounding-noise band of §18.21 or the gain is under 15 %.
4. **N3**, then **N4** (fingerprint the public keep-lists first, since it is a 1 GB range read; compare against our own saliency from Hermes dumps), 25 % before 37 %.
5. **N5** only if the measured hit rate crosses ~90 %.

Rules that still apply: one runtime at a time, no server beside Hamed's, per-token alternation for levers under 15 %, cold pages for cache benchmarks, never edit `src/` during a sweep, the display is never put to sleep.

## 6. What needs Hamed

- Whether output-changing speed levers (N2-N4, N6) may be tried against the harness at all. They are the only route to "much"; each becomes a default only on his word.
- The screensaver set to Never while benchmarking (13 % tax); the Hermes window hidden during runs (up to 30 %).
- Whether a different model for subagent turns (N10) is on the table.

## 7. Sources

[REAP, arXiv 2510.13999](https://arxiv.org/abs/2510.13999) ·
[SliceMoE, 2512.12990](https://arxiv.org/abs/2512.12990) ·
[MoE-CORE, 2610.01950](https://arxiv.org/abs/2610.01950) ·
[DraftExpert, 2607.24434](https://arxiv.org/abs/2607.24434) ·
[AcceptMoE, 2608.02989](https://arxiv.org/abs/2608.02989) ·
[MoE-Spec, 2602.16052](https://arxiv.org/abs/2602.16052) ·
[EcoSpec, 2607.12696](https://arxiv.org/abs/2607.12696) ·
[The Limits of Speculation, 2609.22156](https://arxiv.org/abs/2609.22156) ·
[Dynamic Expert Quantization, 2511.15015](https://arxiv.org/abs/2511.15015) ·
[HOBBIT review](https://www.themoonlight.io/en/review/hobbit-a-mixed-precision-expert-offloading-system-for-fast-moe-inference) ·
[MoE-Infinity issue 234](https://github.com/EfficientMoE/MoE-Infinity/issues/234) ·
[flash-moe](https://github.com/danveloper/flash-moe) ·
[BaseRT on M5 neural accelerators, 2607.19438](https://arxiv.org/abs/2607.19438) ·
Hugging Face: [pipenetwork/GLM-5.3-Flash-REAP37-MLX-mixed-4_8bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-REAP37-MLX-mixed-4_8bit),
[pipenetwork/GLM-5.3-Flash-REAP50-MLX-mixed-4_8bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-REAP50-MLX-mixed-4_8bit),
[JANGQ-AI/MiniMax-M3-REAP32-Coder](https://huggingface.co/JANGQ-AI/MiniMax-M3-REAP32-Coder),
[bullerwins/MiniMax-M3-REAP25-MXFP8](https://huggingface.co/bullerwins/MiniMax-M3-REAP25-MXFP8).
