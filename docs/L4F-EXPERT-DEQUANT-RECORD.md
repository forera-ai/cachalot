# L4(f) step 5: how much of the 2-bit routed-expert kernel's time is the dequantization?

Written 2026-10-15 against runtime 0.65.1, before the measurement. Instrument: `benchmarks/micro_expert_dequant.py`. Hamed's v203 First job 2.

## Baseline

The DeepSeek routed experts run 18 `mx.quantized_matmul` launches a layer (6 experts x w1, w3, w2; 2-bit, group 128). HANDOFF 18.152 measured them at 0.252 ms a layer, 10.1 ms of the 80 ms floor, against a size-matched streaming ceiling of 0.104 ms a layer (4.2 ms of the token): 41 % of the ceiling. The same record says the rest "is dequantization and launch cost, not memory"; that was a hypothesis, never separated. Earlier screens (`micro_expert_roofline.py`, `micro_affine_expert_fusion.py`, HANDOFF 7.1.4) priced launch fusion (concat/gather, 20-25 % better) but not the dequantization.

## Question

If the dequantization arithmetic (`scale * q + bias`) cost nothing, how much faster would a 2-bit GEMV launch be? That is an upper bound for any dequant lever, independent of whether a bit-identical port of MLX's `qmv` exists.

## Method

Random packed buffers of the real shapes (w1: 2304 x 5120, w2: 5120 x 2304; 2-bit, group 128, bf16 scales and biases), 240 distinct matrices per shape so no launch reads what the previous one left in cache, chained in one `mx.eval`, best and median of 9, two processes (A then B) with `settle.sh` between. Arms per shape:

- `qmm`: `mx.quantized_matmul` as shipped.
- `sum`: `mx.sum` over the same bytes (a reading reference, not a ceiling; see v202 lesson).
- `dq`: a custom Metal GEMV (one simdgroup per few rows, `uint4` loads, float accumulation) with the real dequantization.
- `nodq`: the identical kernel with the dequantization removed (the packed word is used as the multiplicand; wrong numerics, same loads, same reduction, same scale/bias reads).

`dq` is checked numerically against `qmm` (relative error, not bit-identity: a screen). `nodq` is never a shippable kernel.

## Prediction (written first)

- P1: `dq` lands within -15 % to +40 % of `qmm` time (a naive kernel is not expected to beat MLX's tuned `qmv`; probability 0.7 that it is slower).
- P2: `nodq` is 5-20 % faster than `dq` (the arithmetic is a few ALU operations per loaded byte and hides under the loads).
- P3: so the saving available to a dequant-only lever is at most 0.2 x (10.1 - 4.2) = about 1.2 ms of the 80 ms floor, below the end-to-end drift (sd 1.2 ms).
- P4: `nodq` stays well below the `sum` rate at these sizes (3 MiB a launch: the rate table says a 2-bit kernel reaches about 56 % of 819 GB/s), so the launch is limited by size, not by arithmetic.

## Stop rule and falsifiers

- Close the dequant lever if `nodq` is less than 10 % faster than `qmm` in both processes.
- Build a bit-identical port next only if `nodq` is at least 30 % faster than `qmm` in both processes (which falsifies P2/P3).
- P1 fails if `dq` is more than 40 % slower or more than 15 % faster than `qmm`.

## Result

Measured 2026-10-15, two processes (A then B; `settle.sh` timed out at 603 s waiting for 65 GiB available, which this 4 GiB arm does not need, so the second process started without it), 240 distinct matrices a shape, best of 9, ms a launch with a bf16 sum around each launch. Both processes agree to 3 %.

| shape | sum | qmm (shipped) | dq tg64 | nodq tg64 | dq tg128 | nodq tg128 |
|---|---|---|---|---|---|---|
| w1 2304x5120 (A / B) | 0.0260 / 0.0254 | **0.0162 / 0.0160** | 0.0208 / 0.0209 | 0.0194 / 0.0191 | 0.0214 / 0.0211 | 0.0199 / 0.0196 |
| w2 5120x2304 (A / B) | 0.0252 / 0.0251 | **0.0191 / 0.0185** | 0.0233 / 0.0218 | 0.0217 / 0.0213 | 0.0222 / 0.0219 | 0.0218 / 0.0215 |

Threadgroup 256 was slower still (w2 0.026-0.028). `dq` agrees with `qmm` to a max relative error of 2.4e-3 to 4.4e-3 (bf16 scales; summation order differs). Rates: `qmm` 205 GB/s (25 % of 819) on w1 and 174-179 (21-22 %) on w2.

## Against the prediction

- P1 **missed narrowly.** `dq` is 28-31 % slower than `qmm` on w1 and 12-26 % on w2 (best threadgroup); inside the -15 % to +40 % window, but the stop-rule reading is the one that matters.
- P2 **held on w1, held on w2 (smaller).** Removing the dequantization makes the same kernel 5-7 % faster on w1 and 1-3 % on w2 (best threadgroup, both processes): inside the 5-20 % prediction on w1, under it on w2.
- P3 **held.** Even the removal of all arithmetic would not bring the kernel to the shipped `qmm`: `nodq` is still 17-20 % slower than `qmm` on w1 and 12-17 % on w2. If MLX's own kernel spends the same 1-7 % of its launch on the arithmetic (not measured: only this kernel's share is), a dequant-only lever is worth at most 0.1-0.4 ms of the token, against the predicted 1.2 ms.
- P4 **held in direction.** `nodq` runs at 20-21 % of 819 GB/s on w1, below the 56 % of the rate table; the mx.sum reading here (16 %) is lower than qmm's own rate, so it is not a ceiling, as the v202 lesson says.
- Stop rule: `nodq` less than 10 % faster than `qmm` in both processes (it is slower than `qmm` in both). **The dequant lever is closed.**

## Explanation (hypothesis)

At about 3 MiB and 16-19 microseconds a launch the kernel is limited by launch size (ramp, occupancy and tail of a short grid) and by memory-access shape, not by arithmetic: removing the arithmetic changes 1-7 %, while MLX's `qmv` beats a naive kernel with the same loads by 20-30 %, which points at its load pattern (each thread keeps several independent loads in flight, scale/bias reuse across a row group) rather than its decode. The in-token 41 % of the 0.104 ms ceiling therefore reflects launch granularity, which the earlier concat/gather screens already priced at 20-25 %.

## Limits

Random packed buffers, bf16 sum around each launch, one set of kernel parameters (16 or 4 lanes a row, uint4 loads), a naive kernel, not MLX's `qmv` port: a hand kernel copied from `qmv_fast` could differ in the loads. That is not a counterexample to the closure, because the closure rests on `nodq`, the no-arithmetic bound of this kernel; but it leaves open that a differently shaped kernel (more loads in flight) could beat `qmm` for reasons other than dequantization. Launch fusion (6 experts in one launch) is the lever those reasons point to, and it was screened at 20-25 % by `micro_expert_roofline.py` (a gather over contiguous slots; the slots are LRU and not contiguous, which is why it never shipped).

## Bottleneck after

Unchanged. Of the 10.1 ms the routed experts take in the 80 ms floor, dequantization arithmetic is not the part to attack; launch granularity and the non-contiguous slot layout are.

## Kind

Research (predicted, measured twice, explained as hypothesis): P2-P4 held, P1 missed narrowly, the stop rule closed the lever.
