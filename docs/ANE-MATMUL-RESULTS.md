# The Neural Engine as a second matmul engine: results (AN-1 to AN-8) — about 4 TFLOP/s, a quarter of the GPU; the go rule is not met

Written 2026-10-30 against runtime 0.69.6. Record: `docs/ANE-MATMUL-RECORD.md` (AN-1 to AN-8, committed first in
9dd1e5d before any install); benchmark `benchmarks/ane_matmul_bench.py`, built in ee9bdb9 before the measured run. Raw
rows (local): `benchmarks/results/ane-matmul/20261030T1448*.jsonl`, `...T145037Z.jsonl`, `...T145111Z.jsonl`.
Machine: Mac Studio M3 Ultra, 96 GB, macOS 27.2; scratch venv (Homebrew Python 3.12.16, `coremltools` 9.0, wheel
sha256 `7079e8b6ff5a63f0...` matching PyPI, `mlx` 0.32.4); no Cachalot runtime running; Hamed's applications open.
Numbers measured unless tagged derived.

## 0. Deviation

The record's hand-off probe `tiny` (an elementwise add) was placed on the **CPU** by Core ML (`MLComputePlan`), so it
times a CPU call, not an ANE dispatch. A second probe, `tiny_ane` (a 1 x 5120 by 5120 x 256 linear, placed on the Neural
Engine), was added before the measured run and is the one scored for AN-6. Both are reported.

## 1. Results

**Placement (`MLComputePlan`, CPU_AND_NE).** Every `linear` (fp16 and int8-weight) was placed on
`MLNeuralEngineComputeDevice`; the int8 model's weight dequantization (`constexpr_blockwise_shift_scale`) is a compile-time
constant; the add was placed on the CPU.

**Alone, 15 s of back-to-back `predict` calls each:**

| model | shape | per call, median (p25-p75) | rate |
|---|---|---|---|
| `gemm4096` | 512 x 4096 x 4096 | 4.228 ms (4.170-4.291) | **4.06 TFLOP/s** |
| `w13_gemm` (shared expert gate+up) | 512 x 5120 x 4608 | 6.577 ms (6.455-6.706) | **3.67 TFLOP/s** |
| `qup_gemm` (attention q up-projection) | 512 x 1280 x 32768 | 8.407 ms (8.314-8.552) | **5.11 TFLOP/s** |
| `w13_gemv` on the ANE | 1 x 5120 x 4608, 47 MB | 0.702 ms | 67.3 GB/s of weights |
| `w13_gemv`, CPU_ONLY | same | 0.736 ms | 64.1 GB/s |
| `w13_gemv_int8` on the ANE | same, int8 weights (24 MB) | 0.608 ms | 38.8 GB/s (1.15x the fp16 call) |
| `tiny_ane` (ANE dispatch probe) | 1 x 5120 x 256 | **0.258 ms** round trip | |
| `tiny` (CPU add) | 1 x 5120 | 0.040 ms | |

**Hand-off from MLX:** an evaluated 1 x 5120 fp16 array to NumPy 0.0007 ms, back to MLX and evaluated 0.0010 ms; a
512 x 5120 array 0.086 ms and 0.089 ms.

**Contention** (cells gpu, ane, both, both, ane, gpu, 20 s each; the GPU running MLX's fp16 4096^3 GEMM in the runtime
venv, the ANE running `gemm4096` in the scratch venv, separate processes):

| | alone | beside the other | change |
|---|---:|---:|---:|
| GPU | 16.63 TFLOP/s (16.66, 16.61) | 17.30 (17.28, 17.33) | **+4.0 %** (faster) |
| ANE | 4.01 TFLOP/s (4.04, 3.98) | 3.66 (3.62, 3.70) | -8.7 % |
| both engines together | | **20.96 TFLOP/s** | +26 % over the GPU alone |

## 2. Scores

| id | result | score |
|---|---|---|
| AN-1 | every matmul on the Neural Engine | held |
| AN-2 | `gemm4096` 4.06 TFLOP/s | held (at the interval's low edge, 4-12) |
| AN-3 | `w13_gemm` 3.67, `qup_gemm` 5.11 TFLOP/s | held |
| AN-4 | `w13_gemv` 67.3 GB/s | held |
| AN-5 | int8 weights 1.15x faster than fp16 | outside (1.2-2.0), not slower |
| AN-6 | ANE round trip 0.258 ms; MLX to NumPy 0.0007 ms | held |
| AN-7 | GPU +4.0 % faster beside the ANE, ANE -8.7 % | outside (the GPU sped up), short of the falsifier |
| AN-8 | the go rule: ANE at least ~5 TFLOP/s on the prefill GEMM — **not met** (3.67 and 4.06; only the q up-projection reached 5.11); GPU under 10 % slower — met; hand-off under ~0.5 ms — met | **not met** |

**Five held, two outside, and the decision rule not met** on its throughput criterion.

## 3. What it means for Cachalot

1. **The Neural Engine is a real matmul engine, about a quarter of the GPU's**: 3.7-5.1 TFLOP/s on prefill shapes
   against the GPU's 16.6-18.2 on the 4096 reference, and running both at once gives ~21 TFLOP/s, +26 % (measured, on a
   synthetic pair of loops).
2. **That +26 % is a ceiling the trunk cannot reach (derived).** Splitting matmul work so both engines finish together
   gives the ANE about 4 / (4 + 17) = 19 % of it, so a fully matmul-bound prefill whose every matmul could move would gain
   at most ~1.23x. Only the dense trunk can move: the routed experts are streamed from the SSD in a 2-bit format into
   buffers chosen at run time, and an ANE program's weights are compiled in. The gain is therefore at most the trunk's
   share of prefill matmul time times that 19 %, which on these shapes is single-digit to ~15 % (derived; the trunk's
   share of a DeepSeek chunk is not measured here).
3. **The costs are large and certain (derived).** The ANE takes fp16 or int8/palettized weights compiled into its own
   program, so the trunk would need a second copy in another format next to the GPU's: DeepSeek's ~12 GiB trunk is FP8
   and bf16, so an fp16 copy is up to ~2x its FP8 size and an int8 copy about the same size, taken from a 96 GB machine
   whose expert cache already sets the speed (48 GiB budget, wired ~73 GiB). The numerics would change (fp16 or int8 for
   FP8), so outputs would change. Every layer of every chunk would hand activations to the ANE and back (0.26 ms a call
   plus 0.09 ms a 512 x 5120 conversion each way; scaled to a 4,096-token chunk, about 3 ms a layer and ~0.13 s a chunk
   of 40 layers against a chunk time of tens of seconds: small), and chunk shapes must be fixed or padded.
4. **Decode gains nothing:** on the decode-shaped GEMV the ANE moves 67 GB/s of weights, no faster than the CPU alone
   (64 GB/s) and a tenth of the GPU's 586-659 GB/s; int8 weights help only 1.15x.
5. **Unexplained:** the GPU ran 4 % faster with the ANE busy beside it (both cells, both repeats). A busy CPU or memory
   clock state is a candidate (compare 0.69.3's E-1 exact path, where the auxiliary arm was also ~1 ms faster); not
   tested.

**Verdict:** lever 2 closes under the rule registered in 0.69.6. The Neural Engine adds about a quarter of the GPU's
matmul rate, but only on weights compiled into its own programs, so for Cachalot it would cost a second copy of the
trunk and a numerics change for at most a single-digit-to-~15 % prefill gain. The Neural Engine's value to Cachalot
remains the side-request offload (`docs/APPLE-FM-ANE-LEVERS.md` lever 1).

## 4. Limits

Synthetic constant weights, one batch size (512) for the GEMMs, `predict` from Python (its overhead is in every call;
`tiny_ane` bounds it at ~0.26 ms), no power numbers (no root `powermetrics` this time), Hamed's applications open. Not
tried: 4,096-token GEMMs, conv-1x1 layouts, palettized 4-bit weights, batching several layers into one ANE program.
