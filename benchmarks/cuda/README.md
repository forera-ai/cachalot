# CUDA research lane: N0/N1 scaffold

Status: **scaffold, never run on a GPU.** This directory implements the instruments of phases N0b and N1 of
[`docs/CUDA-RESEARCH-PLAN.md`](../../docs/CUDA-RESEARCH-PLAN.md): an environment probe, a bounded microbenchmark suite
and a manifest and redaction contract. It measures nothing yet, proves no hypothesis, and is **not** a CUDA backend for
Cachalot. Nothing under `src/cachalot/` imports it, and it imports nothing from `src/cachalot/`.

Where this file and the plan differ, the plan wins. The plan's layout is `benchmarks/cuda/`, so that is where this lives.

## Two environments, kept apart

| | Mac development host (this repository's normal machine) | Provisioned Linux NVIDIA GPU host |
|---|---|---|
| purpose | write and test the instruments | take the measurements |
| CUDA execution | **none**, ever | yes |
| what runs | `plan`, dry-run `run`, `validate`, `doctor` (reports "no CUDA"), the unit tests | `doctor`, then `run --execute` |
| dependencies | the repository's normal dev environment | python 3.12+, a CUDA build of `torch` matching the driver |
| `run --execute` | exits 3 with an informative SKIP | runs under the caps |

The Mac needs no CUDA toolkit, no NVIDIA package and no change to `pyproject.toml`. `torch` is imported lazily and only
inside the GPU cells, so every Mac test and the checkpoint-free `pytest -q` pass without it.

## Files

| file | role |
|---|---|
| `manifest.py` | schema `cuda-manifest` v1: sealing (content-addressed id), validation, result rows, the redaction gate. Standard library only. |
| `doctor.py` | non-secret environment probe; pure parsers for `nvidia-smi`, `nvcc`, `lscpu`, `/proc/meminfo` |
| `suite.py` | plan, limits, statistics, runner, and the torch cells (`TorchBench`) |
| `cuda_bench.py` | command line: `doctor`, `plan`, `run`, `validate` |
| `../../tests/test_cuda_*.py`, `../../tests/fixtures/cuda/` | Mac tests with recorded (synthetic) `nvidia-smi` and `lscpu` output |

## Commands

On the Mac (all safe, none touches a GPU):

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
~/venvs/deepseek-v41/bin/python -m pytest tests/test_cuda_manifest.py tests/test_cuda_suite.py -q
~/venvs/deepseek-v41/bin/ruff check benchmarks/cuda tests/test_cuda_manifest.py tests/test_cuda_suite.py
~/venvs/deepseek-v41/bin/python benchmarks/cuda/cuda_bench.py plan --profile smoke
~/venvs/deepseek-v41/bin/python benchmarks/cuda/cuda_bench.py run --profile smoke
~/venvs/deepseek-v41/bin/python benchmarks/cuda/cuda_bench.py run --profile smoke --execute
```

The last command must print `SKIP: no CUDA run possible here: ...` and exit 3.

On a provisioned GPU host, **only after the operator has approved the session** (plan sections 2, 13 and 17: time cap, D2, D3, D4):

```bash
python3 benchmarks/cuda/cuda_bench.py doctor --out machine-manifest.json --region <continent-or-country>
python3 benchmarks/cuda/cuda_bench.py plan --profile smoke
python3 benchmarks/cuda/cuda_bench.py run --profile smoke --execute --note "N0b smoke"
python3 benchmarks/cuda/cuda_bench.py validate benchmarks/results/cuda/<run-id>/*
```

Order of work: `doctor` (inventory, read it, stop if the toolchain or link is not what the plan expects), then the smoke
profile (30-60 s wall-clock cap), then **inspect** `rows.jsonl` and `run-manifest.json` (gates, `failed_cells`, `skipped`,
`sanity_violations`), and only then the baseline profile with an explicit cap, for example
`--profile baseline --max-seconds 300 --max-alloc-mib 2048 --execute`.

## Caps and what they mean

| control | default smoke / baseline | hard ceiling (not liftable from the command line) |
|---|---|---|
| `--max-seconds` wall clock for the whole run | 45 / 300 | 900 |
| `--max-iters` timed iterations per cell | 12 / 30 | 1000 |
| `--max-alloc-mib` device or pinned allocation per cell | 1024 / 2048 | 4096 |

* The wall-clock cap is checked before every cell and before every timed iteration. A run that hits it stops, records
  `status: truncated` and lists the cells it skipped. It never extends itself.
* A cell whose allocation exceeds the cap is **skipped with a reason**, never shrunk.
* `run` without `--execute` is a dry run: it prints the plan and writes nothing. The estimate it prints is a conservative
  upper bound; the cap is what bounds the run. `gpu_hours_cap` is a count of GPU-hours, never currency.
* Smoke-grade results (fewer than 30 timed iterations) carry a note saying so. Only the baseline profile reaches the plan's n >= 30.
* Nothing here starts, stops or bills an instance. The operator provisions and tears down the host with the provider's own tools.

## What the suite measures (N1)

| family | cells | timer |
|---|---|---|
| copy | H2D and D2H, pageable and pinned, from 4 KiB to 256 MiB, including the exact record sizes 9.95 MB, 14.16 MB, 21.1 MiB and 3 MiB | CUDA events (device) and host monotonic, both reported |
| device bandwidth | device-to-device copy on a buffer far larger than L2; its rate is the **measured peak** that other shares are divided by | CUDA events |
| launch and sync | tiny-kernel launch rate, idle sync, event pair, and a 40-layer chain of (kernel, 4-byte D2H, host wait) | host monotonic |
| matmul | fp16 GEMV (m=1) and GEMM at an expert-like projection shape, loop of g launches against one batched launch, g in {1,2,4,8,16}, L2 flushed between iterations | CUDA events |
| overlap | compute alone, copy alone, compute while pinned H2D copies cover the whole compute window | CUDA events on each stream |
| control | the pinned H2D copy at the DeepSeek record size, first and last cell; the difference bounds drift within the session | CUDA events |

Rules enforced in code: warm-up before timing; `synchronize` before reading a result; p50 and p95 always, p99 only when
n >= 200; seeded bootstrap interval of the median; a cell whose timer overhead (measured as an empty event pair) exceeds
10 % of its median is reported `unavailable (below instrument resolution)`, not as a number.

Correctness gates (a failed gate produces **no timing row**, only a `validation_failed` row):
copies are round-trip byte-compared; matmul is compared to a float32 CPU reference with a recorded tolerance; the overlap
cell byte-compares its copied buffer.

Sanity bounds (a violation is recorded and fails the `sanity` gate, it is never silently dropped): a host-device rate above
70 GB/s, a device-memory rate above 4500 GB/s, a matmul rate above 1.2x the measured device peak. Plan section 9: a rate above
the measured peak is a cache hit or a timing bug, not a result.

Hardware differences are explicit. Only **fp16** kernels run. `suite.dtype_support` gates bf16 (compute capability >= 8.0)
and fp8 (>= 8.9) by what the device reports, so a V100 (7.0) never receives a bf16 or fp8 kernel; the common arm between
V100 and RTX 5090 is fp16. Expert-like shape 4096 x 2880 is **derived** from the 9.95 MB record at 2.25 bit per weight and has
not been read from the model config; the weights here are fp16 (7x the bytes of the 2-bit record), so these are bandwidth
probes at expert scale, not expert kernels. A 2-bit dequantizing CUDA kernel (plan N1.5) is **not** in this scaffold.

## Artifacts (per run, under `benchmarks/results/cuda/<run-id>/`, gitignored)

| file | content |
|---|---|
| `machine-manifest.json` | the doctor report, sealed with its content id |
| `run-manifest.json` | limits, protocol, gates, skipped cells, status, link to the machine manifest id |
| `rows.jsonl` | one result row per line (plan 7.4) |
| `raw-samples.json` | every per-iteration sample, by cell and arm |

Raw outputs stay local. A sanitized summary is what gets committed, under `docs/cuda-results/` (plan section 12).

### Field definitions

Schema `cuda-manifest` version 1. A field that was not measured is **absent**, never zero. Units are in the row.

Manifest (both kinds): `schema`, `schema_version` (1), `platform` (`cuda`), `kind` (`machine` or `run`), `created_utc`,
`manifest_id` (first 16 hex of sha256 of the canonical JSON without this field; tamper-evident).

* machine: `instance_class`, `region` (coarse only), `code.git` (`sha`, `dirty`, `dirty_file_count`), `gpu` (driver view from
  `nvidia-smi`, framework view from torch, `count`), `software` (driver, `nvcc`, torch), `host` (CPU, RAM, kernel, virtualization,
  memlock limit), `interconnect` (negotiated and maximum PCIe generation and width), `environment` (allow-listed variables only),
  `profiler`, `power`, `provenance`. No GPU UUID, serial, bus id, hostname, address, account, cost or credential is collected.
* run: `machine_manifest_id`, `run_id`, `instrument`, `workload` (id, seed, dtypes, shape provenance, an explicit "no model,
  no inference" claim), `mode`, `protocol` (warm-up, caps, timer sources, measured timer overhead), `limits` (caps, hard ceilings,
  elapsed), `gates` (`copy_checksum`, `kernel_reference`, `sanity`, `sanity_violations`, `failed_cells`), `status`
  (`complete`, `truncated`, `failed`, `skipped`, `dry-run`), `skipped` (cell and reason), `operator_note`.

Row: `run`, `arm` (`<cell>|<arm>`), `field`, `value`, `unit`, `tag`, `source`, and optionally `n`, `p50`, `p95`, `ci_low`,
`ci_high`, `ci_method`, `overhead_ms`, `note`. `tag` is one of `measured`, `derived`, `predicted`, `hypothesis`,
`unavailable`; an `unavailable` row has **no value** and a reason in `note`. Bandwidth, shares of peak and slowdowns are
`derived` (from the measured times); times are `measured`.

### Redaction

Environment variables pass an allow-list (`CUDA_*`, `NCCL_*`, `CACHALOT_*`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`PYTORCH_CUDA_ALLOC_CONF`) and a name deny-list (`KEY`, `TOKEN`, `SECRET`, `PASS`, `CRED`, `AUTH`, `CODE`, `PROMO`). Every file
written passes `assert_publishable`, which refuses IPv4 and IPv6 addresses, MACs, UUIDs, GPU UUIDs, e-mail addresses, home
paths, provider-token-shaped strings, long secret-shaped strings and private-key or public-key text. `validate` re-applies
the same check to files that arrive from elsewhere. The refusal message names the JSON path, never the text.

## Operator notes and teardown checklist

Credentials: this tooling never reads, stores or sends a provider API key, promo code or SSH key. Provisioning, SSH and
teardown are done by the operator with the provider's console or CLI. Keep keys in the operator's own secret store, not in
this repository, not in `--note`, not in shell history. If a key was pasted anywhere shared, rotate it.

Before the session: cap and approval agreed (plan section 17 D2, D3, D4); the private cost file exists outside the repository;
install only pinned versions in the approved session and record the exact commands.

Teardown (plan section 13), every session:
1. Results and manifests copied off the instance and checksum-verified (`sha256sum`).
2. `validate` run on every file that will be committed.
3. No process left on the GPU (`nvidia-smi` shows none in use).
4. Attached volumes and snapshots kept or deleted on purpose; the billing consequence checked.
5. Instance stopped or terminated, and the console confirms it is stopped or terminated (not "stopping").
6. Private log updated with end time and GPU-hours.
7. Credentials issued for the session rotated or removed.

## Troubleshooting

| symptom | meaning and next step |
|---|---|
| `SKIP: ... torch not usable (not installed)` | install a CUDA build of torch on the GPU host (version matched to the driver, pinned, recorded) |
| `SKIP: ... torch reports no CUDA device` | driver missing or too old, CPU-only torch wheel, or a container without GPU access; check `nvidia-smi` first |
| `no kernel image is available for execution on the device` in `failed_cells` | the torch wheel does not target this compute capability (Blackwell needs a recent CUDA 12.x build; a very new toolkit can drop Volta). The doctor's `arch_list` shows what the wheel targets. This is the N0b toolchain gate: stop and record it |
| `CUDA out of memory` in a cell | lower `--max-alloc-mib`; the cell is recorded as failed, the run continues |
| `unavailable ... below instrument resolution` | the interval is too short for events; not a number, by design |
| `sanity_violations` not empty | a rate beyond physical bounds: cache hit or timing bug; do not use the affected rows |
| `status: truncated` | the wall-clock cap was reached; read `skipped`, re-run only the missing cells in a new approved session |
| `FAIL` from `validate` with `redaction` | a forbidden string shape is in the file; fix the source, do not edit the file by hand |
| nvidia-smi fields `None` | the platform reports `[N/A]` (common for ECC or power in VMs); recorded as absent |

## Unsupported and out of scope

* Execution on macOS, on AMD or Intel GPUs, or without the NVIDIA driver.
* bf16 and fp8 kernels, 4-bit float formats, a 2-bit dequantizing expert kernel, CUDA graphs, Nsight counter collection,
  NVML power sampling, multi-GPU, and phases N2 to N5 (residency, trace replay, scheduling, comparison, integration).
* Any statement about DeepSeek V4.1 Flash, MiniMax or GLM performance on NVIDIA hardware, any Cachalot inference backend,
  or any datacenter extrapolation (plan rule R10). Working microbenchmarks do not make a CUDA Cachalot.
* Network or remote storage as a PCIe figure. H2D cells use host buffers already resident in memory.
