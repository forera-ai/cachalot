# The Neural Engine as a second matmul engine for Cachalot's dense trunk (experiment record; predictions before any build or run)

Written 2026-10-30 against runtime 0.69.6, **before `coremltools` is installed, before the benchmark is written, before
anything is run**. Charter section 7.2 fields 1 to 3; results go in `docs/ANE-MATMUL-RESULTS.md`. A prediction edited
after its run is void. Hamed: "Go: run the ANE matmul microbenchmark (install coremltools in a scratch venv, ~30-45 min).
Register predictions first." The decision rule is the one written before this record, `docs/APPLE-FM-ANE-LEVERS.md`
section 4.

## 1. Baseline (dated, measured)

* GPU, MLX fp16 stand-ins on this Mac: GEMM 4096^3 18.2 TFLOP/s, GEMV 1 GiB 659 GB/s (0.69.0); the 0.69.5 run read
  16.5 TFLOP/s and 586 GB/s with powermetrics sampling.
* The system model runs on the ANE at 4.3-4.7 W and takes ~2-4 % from a GPU GEMV beside it (0.69.0-0.69.5). No
  measurement of the ANE running *our own* programs exists in this repository.
* DeepSeek V4.1 Flash shapes: hidden 5120; shared expert 5120 x 2304 (fused gate+up 5120 x 4608, down 2304 x 5120);
  attention q low-rank 5120 x 1280, then 1280 x 32768 (64 heads x 512); prefill runs 4,096-token chunks.

## 2. Hypothesis

**H-ane:** the ANE sustains a useful fraction of the GPU's fp16 matmul rate (several TFLOP/s) on prefill-shaped GEMMs
with weights compiled in, its decode-shaped GEMVs are limited by a memory path well below the GPU's 659 GB/s, a Core ML
call costs a fraction of a millisecond, and the two engines contend mainly through memory bandwidth.

## 3. Method

A scratch venv (Homebrew Python 3.12, `coremltools` 9.0 and `mlx` 0.32.4 from PyPI, hashes recorded), outside the
repository and the runtime venv. `benchmarks/ane_matmul_bench.py` (new) builds fp16 MIL programs with constant weights
and fixed shapes, saves them as `.mlpackage`, loads them with `compute_units=CPU_AND_NE`, and reads each op's device
from `MLComputePlan`:

| model | input | weight | shape class |
|---|---|---|---|
| `gemm4096` | 512 x 4096 | 4096 x 4096 | throughput reference (same K and N as the GPU stand-in) |
| `w13_gemm` / `w13_gemv` | 512 x 5120 / 1 x 5120 | 5120 x 4608 | shared expert, prefill / decode |
| `qup_gemm` | 512 x 1280 | 1280 x 32768 | attention q up-projection, prefill |
| `w13_gemv_int8` | 1 x 5120 | 5120 x 4608, int8 linear-quantized | decode, half the weight bytes |
| `tiny` | 1 x 5120 | add a constant | call overhead (the activation hand-off) |

Per model: warm-up, then back-to-back `predict` calls for 15 s; per-call median and spread; TFLOP/s for GEMMs, weight
GB/s for GEMVs. Hand-off: `tiny`'s round trip, plus converting an evaluated MLX array (1 x 5120 and 512 x 5120 fp16) to
NumPy and back in the same process. Contention: an ANE `gemm4096` loop (scratch venv) and the MLX GEMM stand-in
(runtime venv) in separate processes, cells gpu, ane, both, both, ane, gpu, 20 s each. No Cachalot runtime process.
Whole-system watts by SMC `PSTR` if read; the IOReport ANE counter only if Hamed's root `powermetrics` runs (not required).

## 4. Predictions

All **predicted**; probabilities mine.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| AN-1 | placement | `MLComputePlan` puts every fp16 matmul on the Neural Engine (0.65) | any matmul placed on CPU or GPU |
| AN-2 | `gemm4096` on the ANE | 4 to 12 TFLOP/s sustained (0.50) | below 2 TFLOP/s |
| AN-3 | `w13_gemm`, `qup_gemm` | 3 to 10 TFLOP/s each (0.45) | either below 1.5 |
| AN-4 | `w13_gemv` (47 MB of weights) | 30 to 150 GB/s effective (0.45) | above 400 GB/s or below 10 GB/s |
| AN-5 | int8 weights, `w13_gemv_int8` against fp16 | 1.2 to 2.0 times faster (0.40) | slower than fp16 |
| AN-6 | hand-off | `tiny` round trip median 0.1 to 0.6 ms; MLX-to-NumPy 1 x 5120 under 0.1 ms (0.50) | round trip above 2 ms |
| AN-7 | contention, GPU GEMM beside the ANE GEMM | GPU slowed 0 to 10 %, ANE slowed 0 to 15 % (0.50) | GPU slowed more than 20 % |
| AN-8 | the decision rule (`docs/APPLE-FM-ANE-LEVERS.md` section 4: ANE at least ~5 TFLOP/s on the prefill GEMM, GPU under 10 % slower beside it, hand-off under ~0.5 ms) | all three met (0.30) | — (the rule is the outcome) |

## 5. What each outcome means

* **AN-8 met:** lever 2 earns a design: which trunk ops move, the second copy of their weights (and the expert cache it
  costs), and two hand-offs a layer a chunk; the next step is pricing a 4,096-token DeepSeek chunk with these rates.
* **AN-2 or AN-3 below ~2-3 TFLOP/s, or AN-7 past its falsifier:** the ANE cannot add enough matmul throughput to pay
  for the duplicated weights and the hand-offs; lever 2 closes, and the Neural Engine's value to Cachalot is the
  side-request offload only.
* Nothing here is about a real model's numerics, the routed experts (streamed, not compilable), or a default.

## 6. Cost and stop rules

Install ~2-5 min, build and compile ~5 min, runs ~10-12 min, write-up ~10 min: within Hamed's 30-45 minutes. Stop if
the install fails or `MLComputePlan` is unavailable (then placement is inferred from timing against `CPU_ONLY` and the
deviation is recorded).
