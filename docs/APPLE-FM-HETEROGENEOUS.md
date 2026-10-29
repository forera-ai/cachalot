# Heterogeneous execution: the system model on the Neural Engine beside an MLX GPU workload (levers E and D)

Written 2026-10-29 against runtime 0.69.0. Numbers from `benchmarks/apple_fm_bench.py contention` (results in
`docs/APPLE-FM-RESULTS.md` section 4). Labels: measured, derived, hypothesis.

## 1. The question, and what was measured

Can auxiliary work on the Neural Engine run without materially slowing the GPU-bound target model? Separate compute
units do not imply separate memory bandwidth or power: the ANE, GPU and CPU share the M3 Ultra's unified memory and the
package's power budget.

Measured (this session): a GPU stand-in in its own process, alone, then beside the system model generating
continuously in another process, then the system model alone; cells in the order gpu, fm, both, both, fm, gpu (ABBA),
20 s measuring windows after an 8 s warm-up, for two GPU loads:

* **GEMV**: an fp16 1 x 16,384 by 16,384 x 32,768 product, eight per eval (8 GiB read an eval) — bandwidth-bound, the
  shape of batch-1 decode.
* **GEMM**: an fp16 4,096-cube product, eight per eval — compute-bound, the shape of prefill.

| load | GPU alone | GPU beside the system model | GPU change | system model alone | beside the GPU | change |
|---|---:|---:|---:|---:|---:|---:|
| GEMV | 659.4 GB/s (659.1, 659.7) | 645.8 GB/s (646.6, 645.1) | -2.1 % | 52.9 tok/s (53.3, 52.4) | 51.6 tok/s (51.7, 51.6) | -2.3 % |
| GEMM | 18.21 TFLOP/s (18.24, 18.18) | 18.11 TFLOP/s (18.10, 18.13) | -0.5 % | 52.5 tok/s (52.6, 52.4) | 52.5 tok/s (52.3, 52.7) | 0.0 % |

Each cell is the median of one 20 s window; the two values in brackets are the two ABBA repeats. The GEMV arms
separate cleanly (13.6 GB/s between them, 0.6-1.5 GB/s within); the GEMM difference (0.5 %) is barely above its
within-arm spread (0.3 %).

**Reading (derived, hypothesis).** The interference follows memory bandwidth, not compute: the system model takes
about 13.6 GB/s from a bandwidth-bound GPU stream, about 0.26 GB for each of its ~52 tokens a second (consistent with
a sparse ~3B model at low precision reading a fraction of its weights a token; not verified), and nothing measurable
from a compute-bound one. The system model itself slows only where the GPU saturates the memory system.

The GPU process ran MLX; the system model ran in its own service; neither was Cachalot's runtime.

## 2. What it means for Cachalot (derived and hypothesis)

* Cachalot's DeepSeek decode floor is ~53 ms of GPU work in an 80 ms token (0.60.9), of which the expert and attention
  GEMVs are bandwidth-bound like the stand-in. **If** the stand-in's -2 % carries over, an auxiliary request served
  during decode costs the target about 1-2 ms a token while it runs (derived), and the auxiliary loses about 2 %.
  This is a stand-in result: the real decode also streams experts from the SSD, synchronizes with the host 44 times a
  token and holds 48+ GiB wired; it must be confirmed with a real decode arm (section 4) before any scheduling policy.
* The system model serializes its own requests (two parallel sessions take as long as two in a row), so there is no
  auxiliary batching to exploit; at most one auxiliary request runs beside the target at a time.
* The opportunity is not throughput of the target: it is doing auxiliary work in the target's idle windows (between
  turns, during the drive-bound part of a cold load, during Hermes's tool execution) at a measured cost of a few
  percent if it overlaps.

## 3. Lever D, semantic cache assistance (research, nothing built)

Cachalot's caches are the routed-expert slot store (LRU with quotas and deterministic admission), prefix snapshots
(in memory and on disk, keyed on token prefixes) and system-block pins. None of them can use a semantic label without
a change to its own policy code, and Apple's API cannot touch target KV tensors. The measured costs say where a
semantic signal could matter and where it cannot:

* Per decode token (80-100 ms) a 0.33-0.9 s auxiliary call is 4-10 tokens of time: no per-token or per-layer policy.
* Per request: classification at 0.82 s could tag a request (agent turn, side request, tool result) to decide which
  prefix snapshot to keep when the byte budget evicts. The existing evidence (HANDOFF 18.66, the system-block pins) is
  that exact-prefix reuse already captures the agent's repeated blocks; a semantic tag would only help when entries
  compete, which no measured session has shown.
* **Next step, offline and cheap:** replay a dumped Hermes session (`CACHALOT_SERVER_DUMP`) through the prefix-cache
  policy with recency, a deterministic heuristic (request kind from the message structure) and the auxiliary label,
  and count re-prefilled tokens. Only if the auxiliary label beats the deterministic one is there a lever.

## 4. Next experiments (each needs Hamed's go and an estimate first)

1. **E-1, real decode beside the auxiliary: done (0.69.2, `docs/APPLE-FM-E1-RESULTS.md`).** DeepSeek through `serve.sh`
   at 48 GiB, 12 replies an arm in ABBA order: +0.57 ms a token (+0.7 %) with classification calls running back to
   back beside it, slower in 6 of 6 cycles; the calls themselves +5 %. The stand-in's -2.1 % over-predicted it.
2. **E-2, the same for GLM read-bound** (decode is drive-bound on the X10Pro, 1.5-1.6 s a token): the ANE and the GPU
   should then contend even less; ~30 minutes.
3. **E-3, power.** The stand-in cells with Hamed's `sudo powermetrics --samplers cpu_power,gpu_power,ane_power`
   beside them, to see the ANE's watts and whether the GPU's clock drops under the shared budget.

No scheduler is built: the measured interference is small and there is no measured opportunity yet that a scheduler
would capture.
