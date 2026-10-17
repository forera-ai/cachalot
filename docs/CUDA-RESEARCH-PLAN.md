# NVIDIA CUDA research plan

Written 2026-10-17 against runtime 0.66.10. Status: **planned**. No CUDA code exists in this repository, no CUDA measurement has been made, and no instance has been provisioned, so nothing in this document is a result. It is a programme: hypotheses, experiments, gates and rules, written before any GPU time is spent so that the predictions can be compared with what is later measured (charter `docs/RESEARCH-DIRECTION.md`, rule 8.1.1 and section 7.2).

This plan adds a research lane. It does not change the shipped runtime (Apple Silicon, MLX, Metal), any default, the HTTP server, the test suite, or Hamed's lane order (Hermes, vision, speed). It does not claim CUDA support of any kind.

## 0. How to read this document

Evidence labels. Every statement about a number carries one of five labels, and a table cell without a label is a bug.

| label | meaning | example in this document |
|---|---|---|
| **recorded** | measured by an earlier Mac session, cited by ledger row or HANDOFF section, not reproduced here | DS-MISS-48, 2.0 ms a decode miss |
| **derived** | arithmetic from recorded values, vendor specifications or this plan's own definitions, with the arithmetic shown | 240 expert uses a token times 9.95 MB |
| **predicted** | a number or direction written before a run, with an interval and a falsifier | none yet at a number level; section 5 gives the form |
| **hypothesis** | an explanation or expectation that no measurement here tests yet | CUDA-H3 |
| **unavailable** | a quantity the platform or the rented instance cannot report; recorded as absent, never as zero | GPU performance counters inside a restricted container |

Vendor specifications (memory bandwidth, PCIe generation, supported data types) are used only to size experiments and to state what a measurement should look like. They are **derived** inputs, not facts about the rented machine; the machine manifest (section 7) records what the instance actually reports.

The five labels extend the three tags of `benchmarks/run_manifest.py` (`measured`, `derived`, `estimated`). Mapping, until the manifest helper is extended: `measured` = measured, `derived` = derived, `estimated` = hypothesis when it has no measurement behind it and derived when it does. `predicted` and `unavailable` are new and are added to the helper's `TAGS` only when the first CUDA row needs them (section 7.4).

## 1. Purpose, and what this lane is not

**Question.** Cachalot has measured, on one Apple Silicon machine with unified memory, what an expert-streaming MoE runtime pays for storage, cache capacity, synchronization and overlap, and has a predictive model of the storage axis (`benchmarks/whatif.py`, `docs/LEDGER.md` section 7b). Which of those findings are properties of the workload (routing locality, expert size, working set) and therefore transfer to a discrete NVIDIA GPU, and which are properties of the Mac (unified memory, one USB or internal NVMe drive, the Metal driver, CPU-GPU synchronization cost)? A GPU with 32 GB of VRAM, a PCIe link and host memory is a different memory hierarchy running the same question, which makes it the cheapest honest test of the charter's rule C11 ("a laboratory for transferable concepts").

**It is** a separate, measured research lane with its own benchmark code, its own manifests and its own go or no-go gates.

**It is not**
- a port of Cachalot, a CUDA inference backend, or a promise of one (the only decision on that is N5, and a full DeepSeek V4.1 Flash port needs separate approval and a feasibility proof, section 14);
- a way to claim that Cachalot is fast on NVIDIA hardware, or to present the Mac as a datacenter equivalent or a consumer GPU as one (charter 8.1.10);
- a reason to touch `src/cachalot/`, the shell launchers, the Lab app or the existing priorities.

## 2. Resource, disclosure and privacy policy

**Access is potential, not proven.** Forera has been offered CloudRift GPU-compute credits for Cachalot's NVIDIA CUDA research. As reported to this session on 2026-10-17 (not verified by this session), CloudRift lists RTX 5090 32 GB instances in Seoul and V100 32 GB instances in Texas; A100 and H100 are not currently available. Availability, regions, instance types, persistence, network-volume support and prices change, and the credits may expire. Nothing here depends on a specific machine existing on a specific date, and nothing was provisioned, redeemed or installed in writing this plan.

**Public acknowledgement.** The only public wording is "GPU-compute credits provided by CloudRift for NVIDIA CUDA research", linked to `https://www.cloudrift.ai/`, as already in the README. Public text never states the promotional code, the grant's amount, a balance, prices, an account, a contact address of the provider, endpoints or billing details, never implies that CloudRift endorses a result, and never suggests it sponsors anything else.

**What never enters the repository or a published artifact.** Promo or redemption codes; the dollar value, balance or spend; API tokens, SSH keys, passwords; instance identifiers, hostnames, public or private IP addresses, MAC addresses, GPU or board UUIDs and serial numbers, volume or snapshot identifiers; environment variables that can hold secrets (the manifest keeps an allow-list, not a deny-list, section 7.3); the account's email. A region may appear only as a continent or country when the placement matters to a result (a network measurement).

**Private cost tracking is separate from publishable results.** A local, untracked file outside the repository (path chosen by Hamed, decision D2, never committed) holds: session start and stop times, instance type, hours billed, remaining-credit notes, and the per-session and cumulative caps. Published results carry hardware descriptions and GPU-hours as a count, never currency. The repository's `.gitignore` already excludes `benchmarks/results/`; raw CUDA outputs follow the same rule and a sanitized summary is what gets committed (section 12).

**Operational safeguards (hard rules for any session under this plan).**
1. No instance is started without Hamed's yes for that session, and every session states a **time cap** before it starts. The default cap is the shortest time that fits the session's registered experiments plus 25 %; the cumulative cap is D2.
2. No automatic or scheduled instance, no long-lived "just in case" instance. If the provider offers auto-shutdown, it is enabled; the session also ends with the teardown checklist in section 13 regardless.
3. No credits are redeemed by the agent. Redemption and account setup are Hamed's, outside the agent's reach.
4. No checkpoint download above the size cap in D5 without a written cost and storage assessment (bytes, transfer time on the instance's measured network rate, disk needed, idle cost of holding it) and his explicit yes. Prefer fixtures that are generated on the instance or tiny to fetch.
5. Remote software installation happens only in an execution phase Hamed has approved, from pinned versions, with the exact commands recorded in the manifest. Nothing is installed to test an idea that a Mac-only experiment can answer.
6. A session that cannot finish its registered experiments inside its cap stops, records what it has, and reports the shortfall. It does not extend the cap on its own.

## 3. Starting facts from the repository

All **recorded** unless marked. They are the baseline the CUDA lane is compared against and the reason each experiment is shaped as it is.

| fact | value | source |
|---|---|---|
| expert record, DeepSeek V4.1 Flash | 9.49 MiB (9.95 MB), 2-bit affine g128; 40 layers x 384 experts = 15,360; top-6, 240 expert uses a token | `docs/ARCHITECTURE-COMPARISON.md` section 1 |
| expert record, MiniMax-M3 | 21.1 MiB slot image (3-bit); 60 x 128 = 7,680; top-4 | same |
| expert record, GLM-5.3-Flash | 13.5 MiB (14.16 MB), 4-bit; 42 x 288 = 12,096; top-8 | same |
| DeepSeek expert bank on disk | 142 GiB | same |
| Mac expert cache, DeepSeek shipped | 48 GiB = 5,179 slots = 33.7 % of all experts resident | same |
| Mac cost of one decode miss, DeepSeek, internal SSD | 2.0 ms (fits 1.83-2.41) | LEDGER DS-MISS-48 |
| Mac all-resident token, DeepSeek, server | 79.2-80.3 ms; of which 40 router syncs and CPU graph work are a large part (GPU rows 49.9 ms of it) | LEDGER DS-FLOOR-SRV, WI-FLOOR-BW |
| share of miss read time hidden behind compute | 41-47 % (DeepSeek, 34-46 GiB), 77 % under budget 0 | LEDGER DS-LAYER-BLOCKED, DS-BUDGET0-LAYER |
| speculative loads that are wasted | about two thirds of predicted loads (precision 32-48 %) | charter section 3.2 |
| a token reads | 11.94 GB (trunk and shared weights plus 240 experts) | LEDGER WI-FLOOR-BW |
| Mac memory bandwidth | 819 GB/s chip peak; trunk GEMVs 20-77 % of it by projection | LEDGER MC-READ-SIZE, WI-FLOOR-BW |
| routing traces available, no weights needed | `benchmarks/results/routing-trace-{deepseek-long,deepseek-short,glm,glm-agent,glm-long,minimax,minimax-cold,minimax-long}` and `trace_routing_v8_hermes.trace.npz` (gitignored, local to Hamed's machine) | `benchmarks/cache_sim.py`, `RoutingTracer` |

Derived sizing that motivates N2 (arithmetic shown; budgets are placeholders until N0 reports free VRAM):

- If all 240 expert uses of a DeepSeek token missed, that is 240 x 9.95 MB = 2.39 GB a token. Over a PCIe 5.0 x16 link (vendor theoretical about 63 GB/s) that is at best about 38 ms; over PCIe 3.0 x16 (about 15.75 GB/s) about 152 ms. Realistic pinned-memory rates are lower than the theoretical figures, and N1 measures them.
- A 32 GB card cannot hold the 142 GiB bank. With, say, 24 GiB of VRAM given to experts, the resident share is 24 GiB / 9.49 MiB = 2,589 experts = 16.9 % of 15,360, half the Mac's 33.7 %. The hit-rate curve therefore has to be read at smaller resident fractions than the Mac records cover.
- The expert bank (142 GiB) is also larger than the host RAM of a typical rented GPU instance, so the second tier is **not** the same on the two platforms: on the Mac the miss source is an SSD; on a rented GPU it is host RAM if the working set fits there, and local NVMe or a network volume otherwise. That is a research finding about the hierarchy, not a defect to hide (R6, section 11).

## 4. Research questions

Numbered CUDA-Q1 to CUDA-Q8. Each maps to one phase and, in `docs/LEDGER.md` section 9, to a proposed row. They stay **open** until a phase produces a validated answer.

| id | question | phase | existing Mac evidence |
|---|---|---|---|
| CUDA-Q1 | At expert-record sizes (3-22 MiB), is a host-to-device copy bandwidth-bound or latency-bound, and how far does pageable memory lag pinned? | N1 | USB queue depth 1 plus bandwidth wall (ST-X10-QD); no discrete-GPU data |
| CUDA-Q2 | Is an expert-sized GEMV on a discrete GPU memory-bandwidth-bound, and what share of device bandwidth does a per-expert launch reach compared with grouped launches? | N1 | Metal: 60-85 % of 819 GB/s at 3-11 MiB (MC-READ-SIZE) |
| CUDA-Q3 | What do launch, event and stream synchronization cost per layer, and does the Mac's "40 syncs set the floor" finding reappear as a CPU-orchestration floor? | N1, N3 | 40 router evals 9-12 ms of an 80 ms floor |
| CUDA-Q4 | Does the hit-rate curve against resident fraction depend on the trace and policy only (so it transfers), while the cost per miss depends on the device and tier (so it does not)? | N2 | cache_sim reproduces live hit rate on the Mac (HANDOFF 18.54) |
| CUDA-Q5 | At resident fractions below the Mac's range (10-20 %), does a frequency- or weight-aware policy beat LRU by more than at 34 %? | N2 | policies compared at 34 % (`simulate_policies.py`), small gaps |
| CUDA-Q6 | How much of an expert fetch can copy engines hide behind compute, and how much do concurrent copies slow compute on a discrete GPU where copy traffic is a small share of device bandwidth? | N3 | 41-47 % hidden on the Mac; reads slow the GPU part of a token about 25 ms (HANDOFF 18.4) |
| CUDA-Q7 | Which bottleneck binds after storage is removed from the critical path: launch and sync, copies, compute, or VRAM bandwidth, and does it move between an RTX 5090 and a V100? | N3, N4 | bottleneck migration table (LEDGER section 6) |
| CUDA-Q8 | Which of Cachalot's findings are stated in architectural terms that survive the change of platform, and which must be restated as Mac-specific? | N4 | `docs/ARCHITECTURE-COMPARISON.md` |

## 5. Hypotheses (proposed, nothing measured)

Hypotheses are registered here and in the ledger as **proposed**. Each carries a falsifier. The numeric intervals are filled in **before the corresponding run** from the N0 and N1 data (the experiment record, charter 7.2 fields 1-3, is written and dated before the build), because quoting an interval now, without a manifest, would be false precision. What this section fixes is the direction, the quantity, and the test that would break it.

| id | hypothesis | quantity and direction | falsifier | phase |
|---|---|---|---|---|
| CUDA-H1 | Expert-sized pinned H2D copies sit on the bandwidth plateau of the link. | bandwidth at 8-24 MiB is at least 85 % of the plateau measured at 256 MiB, on both GPUs | below 85 % at the DeepSeek record size (9.95 MB) on either GPU | N1 |
| CUDA-H2 | Pageable memory lags pinned by a large factor at record size. | pageable bandwidth below 60 % of pinned at 8-24 MiB | pageable within 20 % of pinned | N1 |
| CUDA-H3 | Per-expert GEMV launches reach well under device bandwidth; grouping several experts into one launch raises it. | a single 3 MiB projection below 60 % of measured device bandwidth; a group of 8 or more at or above 80 % of the large-buffer rate | single-launch share at or above 80 %, or grouping gains under 10 % | N1 |
| CUDA-H4 | The GPU-side cost of a layer's router synchronization (kernel launch chain, event wait, D2H of expert ids) makes a per-layer-sync loop CPU-orchestration-bound at decode shapes. | with 40 layers x one sync, the sync term is at least 10 % of an all-resident token | under 5 % of the token | N1, N3 |
| CUDA-H5 | The hit-rate curve against resident fraction is a property of trace and policy, so a replay on the CUDA lane equals `cache_sim.py` exactly; only cost per miss changes. | hit and miss sequences of the CUDA replay equal the Mac simulator's, event for event | any difference in the sequences for the same trace, seed and policy | N2 |
| CUDA-H6 | At 10-20 % resident, a frequency-aware or router-weight-aware policy beats LRU by more than the same comparison at 34 % does on the same trace. | miss reduction vs LRU at 17 % resident at least twice that at 34 % | gap at 17 % under 1.5x the gap at 34 %, or inside the noise band of the trace | N2 |
| CUDA-H7 | On a discrete GPU, concurrent H2D copies interfere little with compute. | kernel slowdown while a continuous copy stream runs: under 5 % on the RTX 5090 (copy traffic is about 3.5 % of device bandwidth, derived from vendor figures) and under 5 % on the V100 (about 1.7 %, derived) | slowdown of 5 % or more on either card | N3 |
| CUDA-H8 | With copy hidden and storage off the critical path, the remaining token floor is orchestration (launch, sync, host work), as on the Mac. | after overlap, sync plus launch plus host time is the largest critical-path term in an expert-streaming decode loop on at least one of the two GPUs | compute or VRAM bandwidth is the largest term on both | N3, N4 |
| CUDA-H9 | The value of a unit of fast memory (ms a token per GiB of resident experts) scales with the cost per miss, so it is far smaller on a PCIe-attached host tier than on the Mac's SSD tier. | ms saved per GiB of VRAM at the host-RAM tier is at least 5x smaller than the Mac internal-SSD figure (LEDGER DS-BUDGET-CURVE) | ratio under 3x | N2, N4 |

Predictions that depend on the cache policy (CUDA-H5, H6, H9) can be tested **offline on the Mac against the existing traces** with no GPU time at all; N0a does that first (section 6), which is the cheapest way to be wrong.

## 6. Programme

Phases are ordered by dependency and by cost. N0a is the only phase that needs no credits. Every phase has an entry gate, an exit gate, deliverables and a stop rule. A phase that fails its exit gate ends the lane or forces a re-plan; it does not silently continue.

### Priority and ordering

1. **N0a** (Mac only, no GPU time): prepare. Highest value per effort; it can falsify H5, H6, H9 before any instance exists.
2. **N0b**: first instance, inventory and a clean manifest. Short.
3. **N1**: microbenchmarks. The core of the lane; small, cheap, makes every later prediction answerable.
4. **N2**: constrained-VRAM residency, trace replay first, live weights later.
5. **N3**: scheduling and concurrency.
6. **N4**: cross-architecture comparison (a synthesis phase, little new GPU time).
7. **N5**: integration decision.

N1 on the RTX 5090 comes before N1 on the V100 (decision D3, recommended), because a failure to run Blackwell kernels with the available toolchain is the biggest schedule risk and shows up there first; the V100 session then supplies the contrast (PCIe generation, no FP8, no native BF16).

### N0a. Offline preparation on the Mac (no credits, no remote software)

*Entry gate.* This plan merged. Hamed's lane order unchanged.

*Work.*
1. Write the **experiment records** for N1 and N2 (charter 7.2 fields 1-3, with intervals where Mac data supports them) as drafts in a HANDOFF section, to be dated and frozen at the start of each session.
2. Run `benchmarks/cache_sim.py` and `benchmarks/simulate_policies.py` on the existing routing traces at resident fractions that a 32 GB card implies (for example 8, 12, 17, 25 and 34 % of experts) for LRU, SLRU and LFU, with `--expert-bytes` set per architecture. This is an offline, Mac-only, GPU-free run and is the first test of H5/H6's premise. Stop rule: the trace has fewer than 1,000 decode tokens (the simulator warns) or the policy gap is inside the trace's bootstrap band.
3. Specify the synthetic routing generator (section 6, N2 stage 1) and its determinism test, as code that runs on the Mac with fakes. Nothing CUDA-specific is imported.
4. Draft the benchmark layout `benchmarks/cuda/` (proposed, not created by this plan) and the manifest helper's CUDA extension (section 7), unit-tested on the Mac with recorded fixtures of `nvidia-smi` output.
5. Pick the open-model candidates for N2 stage 3 and record, for each, license, size, expert layout and download cost, **without downloading** (D5).

*Deliverables.* A frozen prediction file; an offline policy table by resident fraction with intervals; tests for the generator and the manifest parser; a candidate-model sheet.

*Exit gate.* Offline table exists, and either H5/H6's premise is supported or the lane's N2 scope is reduced in writing.

*Cost.* Mac time only; follows the one-runtime rule and runs when the machine is idle (`benchmarks/guarded_run.sh` if a runtime process could start).

### N0b. Resource inventory and experiment infrastructure (first instance)

*Entry gate.* N0a exit gate; Hamed's yes and a session time cap; the private cost file exists (D2); a decision which instance first (D3).

*Verify and record, for each instance type used (never assume):*

| item | how | notes |
|---|---|---|
| provider, instance type, region (coarse), billing unit, minimum charge, idle cost per hour, stop vs terminate semantics, whether a stopped instance keeps billing for disk | provider console and docs | private file for money, manifest for the type only |
| GPU model, architecture, compute capability, VRAM total and free, ECC state, persistence mode, power limit, clocks and throttle reasons | `nvidia-smi -q`, `nvidia-smi --query-gpu=...`, `torch.cuda.get_device_properties` | expected: 5090 Blackwell (capability 12.0 expected), V100 Volta (7.0 expected); **expected values are vendor-based and must be verified** |
| driver, CUDA runtime, CUDA toolkit and compiler, cuDNN/cuBLAS versions | `nvidia-smi`, `nvcc --version`, `torch.version.cuda` | a single toolkit that targets both capabilities (Blackwell needs a recent 12.x; later major toolkit versions drop older architectures such as Volta, so the toolkit version is a gate, to be verified) |
| host: CPU model, cores, threads, NUMA, RAM, locked-memory limit (`ulimit -l`), huge pages | `lscpu`, `/proc/meminfo`, `ulimit` | pinned memory is bounded by the locked-memory limit; the host RAM size decides what the second tier can hold |
| PCIe: generation and width negotiated and maximum, GPU-to-CPU affinity | `nvidia-smi -q` (PCI section), `nvidia-smi topo -m` | the negotiated link, not the card's maximum, sets the H2D plateau; a virtualized lane can run narrower |
| storage: local disk type and size, network volume, filesystem, measured sequential read, whether it survives stop | `lsblk`, `fio` or `dd` with direct I/O, provider docs | network storage is **not** a PCIe transfer and never enters an H2D figure (section 9) |
| network: region, measured download rate to the model host | one timed fetch of a small public file | for the cost assessment of any download |
| virtualization: bare metal, VM or container; MIG or vGPU; profiler counter permission | `systemd-detect-virt`, `nvidia-smi -q`, a CUPTI/Nsight permission probe | decides whether Nsight Compute counters are available (section 7.5) |
| power telemetry | `nvidia-smi --query-gpu=power.draw,...` and NVML | board power only, sampling rate and averaging window recorded; may be **unavailable** in a VM |
| reproducible environment | pinned package list, container or lockfile, exact commands | recorded in the manifest and in the private log |

*Exit gate.* (a) Safe startup and shutdown demonstrated, including teardown (section 13) and the measured idle cost per hour recorded privately; (b) a **clean machine manifest** validates against the schema (section 7.2) with no private field; (c) a trivial kernel and a pinned copy run and verify bit-exact on the device; (d) the architecture gate table (section 10) is filled in from the instance, not from specifications.

*Stop rule.* If the toolchain cannot target the instance's architecture within the session cap, or the instance cannot report its PCIe link or free VRAM, the session stops and records `unavailable`.

*Deliverables.* `machine-manifest` JSON for the instance; a setup script that reproduces the environment from a clean image; a private-log entry.

### N1. Microbenchmarks

*Entry gate.* N0b exit gate on that instance.

*Measure, per device, in one process per condition with swapped orderings (the repo's ABBA discipline, `MEASUREMENT.md`):*

1. **H2D and D2H copy**: latency and bandwidth against size, from 4 KiB to 1 GiB in a geometric series that includes the three expert record sizes exactly (9.95 MB, 14.16 MB, 21.1 MiB) and the 3 MiB per-projection read size; pageable and pinned (`cudaHostAlloc` or the framework's pinned allocator); synchronous and asynchronous; both directions; also the **pinned-staging** pattern (a pageable source copied into a pinned ring then to the device), since weights arrive from disk into pageable memory first.
2. **Stream overlap of copy and copy**: two directions at once (full duplex), two H2D streams (queue depth), and the number of copy engines the device exposes.
3. **Device memory bandwidth**: device-to-device copy and a read-only reduction on buffers large enough to defeat the L2 cache, as the achievable peak that every "share of bandwidth" is divided by (never the vendor peak).
4. **Launch overhead and synchronization**: empty-kernel launch rate; launch-to-completion latency; `cudaEventRecord`/`cudaStreamSynchronize`/`cudaEventSynchronize` cost; D2H of a small result and host wait (the router-id read), each as a 40-layer chain, to answer CUDA-Q3.
5. **GEMV and GEMM at expert shapes**: the shapes come from the model configs in the repository (DeepSeek expert projections, MiniMax and GLM equivalents), one projection per launch and grouped launches of 2, 4, 8 and 16; fp16 on both devices, bf16 and fp8 only where the architecture gate allows it; a dequantizing kernel for the 2-bit g128 layout in plain CUDA C++ (no new framework), checked against a CPU reference (section 7.6). Report time, bytes read, achieved bandwidth and its share of the measured device peak.
6. **Cold against warm state**: first-touch allocation, first launch (context creation, module load, JIT/PTX compile if any), L2 flush between iterations, persistence-mode state, and the GPU clock ramp after idle. These are reported separately, never folded into a mean.
7. **Jitter**: n of at least 30 timed iterations per cell after a warm-up, p50 and p95 always, p99 only when n is at least 200 (the repo's rule, `run_manifest.percentiles`), bootstrap confidence interval of the median, and a **control cell** repeated at the start and end of the session to bound drift (the Mac's cross-process sd was 1.5 %, LEDGER DS-DRIFT; the rented GPU's is unknown and shared hosts are noisier).

*Do not mistake* network or remote storage for a PCIe transfer: a copy whose source is a file on a network volume is a storage measurement and is labelled so; the H2D rows use a buffer already resident in host memory, verified resident (touched, locked) before timing.

*Deliverables.* A per-device **bandwidth and latency curve** (size against rate, with p50/p95), the measured device peak, the launch and sync table, the GEMV table at expert shapes, the cold/warm table; a short note per hypothesis H1-H4 with the registered interval beside the result.

*Exit gate.* Every H1-H4 cell has a result or an `unavailable` with a reason; the measured peaks are internally consistent (a device-to-device copy cannot exceed the reduction's read rate by more than the copy-count arithmetic allows; a pinned H2D rate cannot exceed the negotiated link's theoretical rate); the control cell's drift is reported.

*Stop rule.* A cell whose timer overhead (an empty event pair, measured) exceeds 10 % of the measured time is reported as below the instrument's resolution, not as a number.

*Cost.* Estimate: a short session per device (order of a few GPU-hours including setup). **Estimate, not a measurement;** the session cap is set from the first N0b timing.

### N2. Constrained-VRAM MoE residency

*Entry gate.* N1 exit gate on at least one device (the copy and GEMV curves are the cost model's inputs); N0a's offline policy table.

*Separation of claims.* Three stages, and a result states which stage produced it:

| stage | what runs | what it can claim | what it cannot |
|---|---|---|---|
| 1 synthetic trace replay | a seeded, parameterized routing generator (layers, experts, top-k, skew, temporal locality) drives a real VRAM slot arena and real H2D copies of dummy expert-sized buffers | cache behaviour and transfer cost of a pattern, with controlled knobs | anything about real model routing |
| 2 extracted trace replay | the recorded Mac routing traces (DeepSeek, MiniMax, GLM) drive the same arena | the policy and fraction curves on real routing, and the exact-equality test of H5 against `cache_sim.py` | real compute, real quality, a live token time |
| 3 live inference on a manageable open model | a small open MoE with real weights, its expert cache **capped below its natural VRAM need** by a policy parameter, so the constraint is emulated and labelled "emulated cap" | end-to-end token time with real kernels at a capped working set | any statement about DeepSeek V4.1 Flash, whose bank does not fit; trace replay and live claims are never merged into one table |

*Measure.* Cache hit and miss, useful and wasted prefetch (an expert prefetched and used before eviction against one that is not), bytes per token moved H2D, evictions, exposed stall (critical-path wait for a copy) against hidden copy time (charter 4.4), working-set curve (resident fraction against hit rate and ms a token), and the hit-rate curve for **LRU, SLRU, LFU, router-weight-aware and prediction-informed** policies using the existing policy definitions where they exist (`simulate_policies.py`, `cache_sim.py`, the prefetch predictors recorded in the routing traces' predicted sets, LEDGER predicted-set rows). Expert sizes, top-k and expert counts come from the three Cachalot architectures plus the open model.

*Account explicitly for:* pinned host memory consumed by the staging ring and by the host-tier cache (counts against the locked-memory limit); the VRAM actually available (total minus context, workspace, trunk weights, KV and allocator reserve, read from the device after warm-up); allocator behaviour: a **fixed-size slot arena allocated once** (the repo's slot-image idea) so fragmentation is zero by construction, and the framework allocator's reserved against allocated bytes recorded if a framework is used; and the PCIe link's measured rate from N1, not its nominal one.

*Deliverables.* A replay harness with a determinism test; curves for stages 1 and 2 on both devices; the stage 3 curve for the chosen open model on one device; the experiment records with predictions beside results.

*Exit gate.* H5 tested (equality or the exact point of difference found); H6 and H9 answered with intervals; every curve carries the stage tag; for stage 3 the emulated cap and the natural footprint are both stated.

*Stop rule.* A policy gap inside the trace's bootstrap band is a null result and is published as such (section 12). A stage 3 run whose quality is not verified against the same model's reference output (section 7.6) is discarded.

### N3. Scheduling and concurrency

*Entry gate.* N1 exit gate (launch, sync and copy costs known); N2 stage 1 harness.

*Experiments.*
- **Streams and events**: a compute stream and one or two copy streams with event dependencies, a bounded **prefetch queue depth** (in-flight copies) swept from 1 to 8, and a bounded pinned staging ring for backpressure (the producer blocks when the ring is full; the blocked time is recorded).
- **Overlap of compute and copies**: layer-ahead prefetch of predicted experts against demand fetch on miss, with prediction precision swept using the recorded predicted-set traces, so wasted copies are charged.
- **Batching and expert grouping**: group tokens (or the selected experts of one token) into fewer, larger launches; measure the launch-count and bandwidth effect against CUDA-H3's grouped GEMV curve.
- **CPU orchestration**: the host-side cost of issuing a layer (Python or C++ driver loop), CUDA graphs against eager launch where the instance allows, and the router D2H sync per layer; this is the CUDA counterpart of the Mac's "40 syncs" finding.
- **Interference** (CUDA-H7): a compute kernel timed alone, then with a continuous H2D copy stream running, both devices.
- **Critical-path reconstruction.** Every timed run records timestamps of kernel, copy and host events with CUDA events and host monotonic clocks; the post-processor builds the dependency graph and computes, per token, **resource time** (per lane: copy engine, compute, host) and **critical-path time** (charter 4.4). **Overlapping durations are never summed into a wall time.** Overlap is the measured union of intervals per lane, as in `benchmarks/decode_anatomy.py`; hidden time is resource time minus exposed time. A reported token time is a measured wall time; a reported component is a lane or a path segment, and the two are never added.
- **Negative results and bottleneck migration.** Each change records the term it removed and the term that bound next (charter 8.1.3). An optimization that moves the wait from copy to sync without changing the token time is a result, recorded as such.

*Deliverables.* Overlap and queue-depth curves, the interference table, the critical-path breakdown per device, a migration table in the style of LEDGER section 6.

*Exit gate.* H7 and H8 answered; the sum of reported critical-path segments equals the measured token wall time within the timer's stated error (a closed accounting, otherwise the decomposition is rejected).

*Stop rule.* If instrumentation adds more than 5 % to the token time at the sampling used, the run is sampled (1 in 4, as the Mac tracer was) and the sampled overhead is reported with its interval; an instrumented run is never the speed baseline (charter 8.1.7).

### N4. Cross-architecture comparison

*Entry gate.* N1 on both GPUs; N2 stages 1-2 on both; the Mac figures cited from the ledger with their regime tags (a Mac re-measurement needs the idle machine and Hamed's go; the default is to cite).

*What is compared.* Apple M3 Ultra (MLX, Metal), RTX 5090 (CUDA), V100 (CUDA), **only** where the comparability rules of section 11 hold. The workload definitions are shared: one trace file, one set of tensor shapes, one stored-byte layout, one seed, each with a content hash in the manifest.

*Deliverables.* A comparison table whose every cell is tagged with its equivalence class (section 11); a section phrased in architectural terms (CUDA-Q8): which finding is a property of expert size, locality and working set, and which of the platform; value of a unit of fast memory (ms a token per GiB, the charter's C10) by tier; a list of Mac findings the CUDA lane **failed to reproduce**.

*Exit gate.* Every cross-platform number states its class, its regime and whether it is measured or derived; no cell mixes classes.

### N5. Evidence-led integration decision

*Entry gate.* N4 exit gate, and Hamed's request to decide.

The decision is one of four and is made from evidence, not from momentum:

| option | choose when the evidence shows | what is built |
|---|---|---|
| A. Independent CUDA benchmark library | the lane produced reusable, validated instruments and measured curves but no kernel worth shipping | `benchmarks/cuda/` stays standalone; documented and tested; no runtime import |
| B. Reusable low-level CUDA kernels | one kernel (for example the 2-bit g128 expert dequant-GEMV or the slot-arena copy path) beats the framework baseline by more than the instrument's noise on both devices and is bit-checked | a small, separately versioned kernel package, still no runtime dependency |
| C. Optional experimental CUDA inference backend | the first two plus a one-layer parity proof and a memory/cost feasibility proof (section 14) are accepted | a gated, opt-in backend behind an explicit switch, default off, not part of the Apple Silicon path |
| D. No integration | the transferable findings are the product; the CUDA code is evidence only | archive the code and results; the charter records the findings |

A shared interface between the MLX and CUDA code is considered **only** under the rule of two (section 16): two working implementations, a concrete duplication, and evidence that the abstraction removes it.

## 7. Reproducibility contract

### 7.1 What is reproducible

A result is reproducible when someone with the result file, the manifest it names and this repository can rerun the same experiment on an equivalent instance. Everything below exists to make that true, and to make a failure to reproduce attributable (instance, driver, clocks, neighbour) rather than anecdotal.

### 7.2 Machine manifest and run manifest (schema `cuda-manifest` v1)

Two JSON documents, versioned and extending the Mac conventions of `benchmarks/run_manifest.py` (commit and dirty flag, content hashes, an allow-listed environment, machine state, fields absent rather than guessed).

**Machine manifest** (once per instance session; many runs reference it by hash):

| group | fields |
|---|---|
| identity | schema version; `platform: "cuda"`; manifest id (hash of the canonical JSON without the id); UTC timestamp; instance class (provider-neutral type name, e.g. `rtx5090-32gb`); region (continent or country, optional) |
| code | repository commit and dirty flag; plan version; kernel source hashes; build flags; `nvcc` arguments; target architectures |
| GPU | name; architecture; compute capability; VRAM total and free after warm-up; SM count; memory type and bus width as reported; ECC; persistence mode; power limit; supported and tested data types (section 10); clock and throttle readings at start and end; a salted hash of the GPU identity (never the UUID) |
| software | driver; CUDA runtime and driver API versions; toolkit and compiler; cuBLAS/cuDNN where used; Python and framework versions; the pinned package list hash; container image digest if any |
| host | CPU model, cores, threads; NUMA nodes; RAM; locked-memory limit; kernel version; virtualization type |
| interconnect | PCIe generation and width, negotiated and maximum; GPU-CPU affinity; copy-engine count |
| storage | local disk type and size; measured sequential read; network-volume presence (a boolean and its measured rate, no identifiers); whether the working directory is local or network |
| profiler | which of CUDA events, NVTX, framework profiler, Nsight Systems and Nsight Compute are present, permitted and tested, each with `available` or `unavailable` and the reason |
| power | whether NVML power is readable, sampling interval, averaging window; `unavailable` otherwise |
| provenance | operator-supplied note; session time cap; hours used (count only) |

**Run manifest** (per experiment run):

| group | fields |
|---|---|
| link | machine manifest id; run id; instrument name and version; commit |
| workload | workload id and class (W1-W7 where it maps); model or fixture id and **content hash**; trace hash; seed(s); tensor shapes; stored dtype and compute dtype; quantization layout |
| mode | run mode (`timing`, `profiling`, `correctness`); cache state per tier (cold, warm, flushed, how); resident fraction and emulated cap if any; policy and prefetch settings |
| protocol | warm-up iterations; timed repetitions; ordering (ABBA or other); control cells; timer source (CUDA events, host monotonic); timer overhead measured |
| measurement | profiling mode and its measured overhead; sampling rate if sampled |
| results | scorecard rows (section 7.4) |
| gates | correctness-check outcomes (section 7.6); the registered prediction id and its result |

### 7.3 Redaction

The manifest writer uses an **allow-list** for environment variables (for example `CUDA_*`, `NCCL_*` tuning knobs, `OMP_NUM_THREADS`, `CACHALOT_*`), drops everything else, scrubs hostnames and addresses, and replaces any identifier with a salted hash where a stable identity is needed for deduplication. A publish step refuses a file that contains an address-shaped string, a token-shaped string, a UUID, a MAC, a path under a user's home, or an email address, and refuses an unredacted manifest outright. This is tested on the Mac with fixtures before any instance exists (N0a).

### 7.4 Result rows

One row is one measured, derived or predicted value tied to a run manifest id: `{run, arm, field, value, unit, tag, n, p50, p95, ci_method, ci_low, ci_high, source, overhead_ms, note}`. The `tag` is one of **measured / derived / predicted / hypothesis / unavailable**; an `unavailable` row has no value and a reason. Units are explicit. A field not measured in a run is absent, never zero. The rows are JSON lines so that the existing `append_rows` pattern and a future Lab importer need no new format. New field names are added to the schema list **before** a row may carry them, as `run_manifest.py` already requires.

### 7.5 Trace and profiler strategy, and overhead controls

| layer | tool | use | availability rule |
|---|---|---|---|
| base | CUDA events and host monotonic clock | every timing in N1-N3 | always present; timer overhead measured as an empty event pair and reported |
| annotation | NVTX ranges | name regions for later tools | free when no profiler attached; recorded as present |
| framework | `torch.profiler` (with CUPTI) | breakdown of framework runs in N2 stage 3 | recorded if the framework is used; overhead measured |
| system timeline | Nsight Systems | confirm overlap and find host gaps in N3 | **only if installed and permitted**; otherwise `unavailable`, and the CUDA-event reconstruction stands alone |
| kernel counters | Nsight Compute | explain a kernel's bandwidth share | **only if the instance grants performance-counter access** (often restricted in containers and VMs); otherwise `unavailable` |
| power | NVML / `nvidia-smi` | board power in N3 and N4 | board power only, sampling and averaging recorded; may be `unavailable` |

Overhead controls: (1) **timing runs and profiling runs are separate runs**; a profiled run is never used for a speed number; (2) the profiler's overhead is measured by running the same cell with and without it and reported; (3) event recording is placed outside the timed kernel region and its cost is subtracted only if it is measured and stated; (4) clocks, temperature and throttle reasons are sampled before and after each cell and a throttled cell is flagged; (5) GPU clock locking is applied only where the instance permits it and is recorded, never assumed; (6) a control cell bounds drift within a session.

### 7.6 Correctness checks (a run without them is void)

- **Copies**: round-trip checksum of the bytes (device to host to compare) for every size and mode.
- **Kernels**: compare against a float32 CPU reference on seeded inputs with a tolerance set per dtype and recorded; a kernel that fails does not produce a timing row.
- **Replay**: the CUDA replay's hit/miss/eviction sequence equals `cache_sim.py` for the same trace, seed and policy (H5); a difference is a bug until proven otherwise.
- **Live (stage 3)**: token ids and teacher-forced NLL against the same model's reference on the same device, using the repository's method (greedy text is near-tie sensitive and is not a gate, `MEASUREMENT.md`). Cross-device numerics are compared by NLL and max-KL within a stated band, never by bit equality.
- **Determinism**: two runs with the same seed give the same sequence of cache events.

### 7.7 Importable by Cachalot Lab, without Lab running CUDA

All outputs are JSON lines and JSON manifests with the schema above. Lab reads them as it reads Mac scorecards (read-only import); Lab never starts, manages or bills a GPU instance, never needs CUDA, and never holds provider credentials. Remote GPU management is **not** a Lab feature and this plan proposes none. If Lab later shows CUDA runs, the only runtime-side request is a new `platform` field and the evidence-tag vocabulary of section 0; that is a Lab proposal to be written in the Lab repository's own process (`docs/RUNTIME_REQUESTS.md` there) when N1 has produced files worth importing. No Lab repository change is made by this plan.

## 8. Repository layout (proposed, created when N0a starts)

```
benchmarks/cuda/          standalone microbenchmarks and replay harness (no import from src/cachalot)
benchmarks/cuda/kernels/  plain CUDA C++ sources
benchmarks/cuda/manifest.py   CUDA manifest builder, redaction, schema check; reuses run_manifest.row / append_rows
docs/cuda-results/        sanitized summaries and curves committed for the public record (benchmarks/results/ stays gitignored)
tests/test_cuda_*.py      Mac-runnable tests with recorded nvidia-smi fixtures; GPU tests skip when no device
```

Mac tests must pass unchanged on a Mac with no NVIDIA stack: CUDA tests skip when no CUDA device is present, and nothing in `src/cachalot/` imports the CUDA code.

## 9. What a measurement must not be confused with

- Network or remote storage is not local PCIe transfer.
- Host-to-device from pageable memory is a different quantity from pinned, and a copy from a freshly mapped file is a storage read.
- A kernel timed with stale L2 contents is not a cold-memory kernel; flush between iterations when the question is memory-bound.
- Summed event durations are not wall time (section 6, N3).
- A rate above the device's measured peak is a cache hit or a timing bug, not a result (the repository's page-cache lesson: a rate above the drive wall is a cache hit; the 0.61.6 and 0.61.7 raw-rate claims were withdrawn in 0.61.8, HANDOFF Start here 2026-10-07).
- A shared-host instance drifts; one run is one run.

## 10. Architecture gates

Features are enabled by what the device reports, never by the instance's name. A feature a device lacks yields `unavailable` for that device, not a slower number.

| feature | RTX 5090 (Blackwell, capability 12.0 expected) | V100 (Volta, capability 7.0 expected) | gate |
|---|---|---|---|
| fp16 tensor-core math | expected yes | expected yes | compute-capability query, then a verified kernel |
| bf16 compute | expected yes | not native on Volta; do not assume | query plus a bit-check of a bf16 kernel; else `unavailable` |
| fp8 (E4M3/E5M2) tensor-core math | expected yes | **no**; Volta has no fp8 | capability gate; the common arm decodes fp8 arithmetically to fp16, as the Mac's E4M3 path does |
| 4-bit float tensor-core formats | expected on Blackwell only | no | capability gate; device-specific arm only |
| async copy engines | expected multiple | expected fewer | queried (`asyncEngineCount`), measured in N1 |
| link generation | expected PCIe 5.0 x16 | expected PCIe 3.0 x16 for PCIe cards (SXM variants use NVLink; the rented type is to be verified) | read negotiated link in N0b |
| memory | GDDR7, 32 GB | HBM2, 32 GB | read in N0b |
| toolkit | a recent 12.x targeting capability 12.0 | the same 12.x still targets 7.0 | N0b gate; a later major toolkit may drop Volta |

"Expected" cells come from vendor documentation and are **derived** inputs only; N0b fills the real values.

## 11. Cross-architecture comparability rules

A number from one platform is placed beside a number from another only if all of these hold. Violations are not caveated, they are not compared.

1. **R1 identity of workload.** Same trace or fixture hash, same tensor shapes, same stored-byte layout, same seed, same context and batch. Different model, context, quantization or topology means a different workload.
2. **R2 equivalence classes.** Every comparison cell is tagged: **E0** identical stored bytes and identical arithmetic result (for example byte copies); **E1** same stored bytes, same algorithm, numerically equivalent within a stated tolerance (the common fp16-compute arm); **E2** same algorithm, device-native precision or kernel (5090 native fp8 vs V100 fp16); **E3** different model or quantization. Performance is compared in E0/E1; E2 only with the difference stated beside it; E3 never.
3. **R3 common arm and native arm.** Each device runs a *common* arm (lowest shared capability, so the comparison is fair) and may run a *native* arm (best it supports). Only common arms are cross-device comparable; native arms are labelled device-specific.
4. **R4 normalize by the measured peak, not the spec.** Achieved bandwidth is reported with its share of the device's own N1 peak. Cross-platform "efficiency" is that share, labelled **derived**.
5. **R5 same cache state definition.** Cold, warm and flushed mean the same thing on every platform and per tier; the manifest says how each was produced.
6. **R6 tiers are named, not equated.** A "miss" is tier-specific: SSD to unified memory (Mac), host RAM to VRAM, local NVMe to host to VRAM, or network volume to host to VRAM. Cost per miss is compared only between equal tier pairs, and the hierarchy difference itself is a finding.
7. **R7 topology differences are stated.** Unified memory against discrete VRAM; the Mac's driver-managed GPU allocation and wired-memory cliff against VRAM exhaustion; PCIe generation and width; CPU model (orchestration cost depends on the host CPU).
8. **R8 energy.** NVML board power is GPU-board only; the Mac's `PSTR` is whole-system and `powermetrics` reports package components. Joules a token are compared only between like scopes (GPU component against GPU component) and carry the sampling interval; a system-level comparison is `unavailable` unless a like-for-like scope exists. Energy tables from the Mac (LEDGER ENERGY-*) are not re-labelled.
9. **R9 clocks and thermals.** Recorded per cell; a throttled cell is flagged and not averaged into a curve silently.
10. **R10 no datacenter extrapolation.** RTX 5090 is a consumer GPU and V100 an older datacenter GPU; neither stands in for A100/H100. Any scaling to other hardware is a **hypothesis** labelled as such, with the model it comes from (`whatif.py`-style, validated only at measured points).

## 12. Publication

- Raw outputs stay local (`benchmarks/results/` is gitignored). What is published is a **sanitized summary**: curves, tables and manifests passed through the redaction check, committed under `docs/cuda-results/` with the plan's hypothesis ids.
- Every published number states label, class, regime, n and interval, and which instance class and date it came from (section 0). "RTX 5090" and "V100" appear as hardware descriptions only.
- **Negative and null results are published** on the same terms as positive ones, in the ledger, when a registered hypothesis was tested with a correct method: the hypothesis, the prediction, the falsifier, the interval of the result, and the reason the method was adequate. A null is also published when it removes an expectation the Mac lane had (for example "grouping does not help on the 5090"). A result is withheld only if its correctness gate failed (then the gate failure is recorded instead) or if it contains data the redaction rules forbid.
- Prediction misses are reported in the record's "Against the prediction" paragraph, as in HANDOFF 18.162-18.164. A prediction that was edited after its run is void.

## 13. Cost discipline and teardown

**Budget discipline.**
1. The unit is **GPU-hours** (a count) and a **time cap per session**; currency lives only in the private file (D2).
2. Every session has an itinerary: the experiments from N0b-N3 it will run, in priority order, each with an estimated duration, set before starting. Stop when the cap is reached; unfinished items are carried forward.
3. Work that does not need a GPU is done on the Mac (code, tests, analysis, manifest checks, offline policy tables).
4. A cheap contrast beats an expensive confirmation: run N1 on both cards before any N2 stage 3.
5. The idle cost per hour of each instance type is measured in N0b and recorded privately, so the cost of a pause is known before it is taken.
6. Do not hold state on a billed instance: results are copied off the instance at the end of each experiment, and the instance is stopped or terminated per its billing semantics.
7. Estimates in this plan ("a few GPU-hours") are **estimates**, replaced by measured session durations as they accrue.

**Teardown checklist (every session, end).**
1. All results and the manifests copied off the instance and checksum-verified.
2. Redaction check run on the files that will be committed.
3. Processes stopped, GPU memory released (`nvidia-smi` shows none in use).
4. Any attached volume or snapshot: kept or deleted on purpose; the billing consequence checked against the provider's current terms and noted privately.
5. Instance stopped or terminated, and the console state confirmed as stopped or terminated (not "stopping").
6. Private log updated with end time and hours; the cumulative cap checked.
7. Credentials and keys used in the session rotated or removed if they were issued for it.

**Availability and expiration uncertainty.** The instance types, their regions and the credits' expiry are provider facts that can change without notice. The plan's order puts the cheapest, falsifiable work first (N0a on the Mac), keeps N0b-N1 short, and treats each session as potentially the last: results are written off the instance at once and each phase's deliverable is useful on its own.

## 14. Risks

| risk | effect | mitigation |
|---|---|---|
| the toolchain does not target the card (Blackwell needs a recent toolkit and driver; a later toolkit drops Volta) | N1 cannot run | N0b exit gate checks it with a trivial kernel before any experiment; pin to one toolkit that targets both |
| rented hardware is shared, throttled, virtualized or restricted | noisy or incomplete counters | control cells, repeat on a second day, flag throttle, `unavailable` for blocked counters |
| host RAM too small for the second tier | the planned stage 2/3 working sets do not fit | decided in N0b from the real RAM; scale the emulated cap and state it |
| a 142 GiB bank is not representable on the card | a faithful DeepSeek port is not feasible on 32 GB without an NVMe/network tier | out of scope until N5; the lane studies the hierarchy with traces and a small open model |
| replay is mistaken for inference | wrong claims | stage tags on every result (section 6, N2); a live claim needs the live stage |
| differing numerics make cross-device comparison unfair | false cross-platform conclusions | R2/R3 classes, common arm, NLL-based checks |
| credits expire or instances disappear | lost work | off-instance results at once; every phase's deliverable stands alone |
| scope creep into a port | the lane becomes a rewrite | N5 gate and the rule of two; no `src/` import |
| a sponsor mention reads as an endorsement | reputational risk | wording rule (section 2) |
| time competes with Hermes, vision and speed | priorities violated | the lane is a research lane, scheduled by Hamed; the plan changes no priority; a conflict is escalated, not resolved silently |

**What a full DeepSeek V4.1 Flash CUDA port would require (N5, separately approved).** A memory proof (non-expert weights plus KV plus workspace in VRAM, the 142 GiB expert bank across host RAM, NVMe and network tiers with measured rates); a kernel inventory of what the runtime uses on Metal (FP8 trunk GEMV, 2-bit expert path, compressed sparse attention with the sliding window, Engram n-gram lookups, hyper-connection mix) with a CUDA plan for each; a one-layer numerical parity proof against the Mac output by NLL and KL; a cost estimate in GPU-hours; and Hamed's written approval. None of this is begun by this plan.

## 15. Acceptance criteria for this plan, and for each phase

This document is accepted when: it states hypotheses with falsifiers, phases with entry and exit gates, a manifest and redaction contract, comparability rules, cost and teardown rules, and the decisions below; it separates planned work from measured work; it asserts no benchmark was run; and it is consistent with `docs/RESEARCH-DIRECTION.md` (section 14 there).

A **phase** is accepted when its exit gate is met, its deliverables exist, every number in them carries a label from section 0, every registered hypothesis it covers has a result or an explicit `unavailable`, its experiment record has fields 1-3 dated before the run, and its manifests pass the redaction check.

## 16. Relationship to the charter and the portability rule

Charter rule C11 and the last line of 8.2 say: keep platform-neutral what is already neutral and build **no speculative portability layers**. This plan keeps that rule and applies it:

- What is shared across platforms is **data**, not code: the routing-trace format (`RoutingTracer`), the policy definitions as specified in `cache_sim.py`, the run-manifest and scorecard JSON lines, and the workload class names W1-W7. These were already neutral.
- The CUDA benchmarks are **standalone code** under `benchmarks/cuda/`. They do not import the runtime and the runtime does not import them. There is no `Backend` interface, no device abstraction layer, and none is proposed.
- **Rule of two.** A shared abstraction may be proposed only when two working implementations (the MLX/Metal path and a CUDA path) exist, a specific duplication is demonstrated, and the abstraction is measured not to cost performance or clarity. Before that, duplication is cheaper than the wrong abstraction.
- The lane is a measurement of the rule, not an exception to it: if N5 shows the transferable findings are statements about expert size and locality, the neutral artifacts are all the project needs; if it shows reusable kernels, they stay a separate package.

## 17. Decisions that need Hamed

| id | decision | recommended | why |
|---|---|---|---|
| D1 | Start N0a now, and when to spend the first instance session (credit expiry is unknown) | N0a immediately (no credits); N0b only after its exit gate | N0a can falsify H5/H6 for free; it also writes the predictions that the first session needs |
| D2 | Where the private cost file lives and the per-session and cumulative GPU-hour caps | a file outside the repository; first-session cap about 3 GPU-hours (an estimate to be replaced by N0b's measurement) | keeps currency out of the repository and bounds the first spend |
| D3 | Instance order | RTX 5090 first for N0b and N1, V100 second for contrast | the modern toolchain risk shows first; V100 supplies the PCIe 3 / no-fp8 contrast |
| D4 | Permission to install software on an instance (pinned versions, commands recorded) | allowed per session, listed in the session itinerary | N0b needs a toolkit and Python stack |
| D5 | Open model for N2 stage 3 and the checkpoint size cap | choose from the candidate sheet; cap at a size that fits the instance's local disk and the session cap; assessment required above it | avoids giant downloads on a billed clock |
| D6 | Publication path: `docs/cuda-results/` for sanitized summaries; raw outputs stay local | yes | matches the repository's gitignore and redaction rules |
| D7 | Lab: when to open the proposal in the Lab repository's own process | after N1 produces importable files | nothing to import before then; no Lab change now |
| D8 | N4 Mac re-measurements: cite the ledger or re-run on the idle Mac | cite by default; re-run only a regime-stale row and only with his go | respects the one-runtime rule and the stale-constant rule (charter 8.1.4) |

## 18. Honest limits

- This is a plan. No hypothesis in it has been tested; every "expected" device property is vendor-derived and unverified for the rented machine.
- The estimates of GPU-hours are guesses until N0b measures a session.
- The Mac numbers cited are recorded values from earlier sessions in their stated regimes; this document did not reproduce any of them.
- The offline policy work in N0a uses traces recorded on the Mac with its own workload mix; a CUDA-lane conclusion drawn from them is a statement about those traces.
- A discrete-GPU hierarchy differs from unified memory in more than one way at once; the lane can attribute a difference to tier, link, driver or kernel only where an experiment varies one of them.
