# CUDA session N2S1 runbook: stage 1 replay with a real slot arena and real copies

Predictions, design and stop rules: `docs/CUDA-N2S1-EXPERIMENT-RECORD.md` (committed in b1974e2 before any harness code). Harness: `benchmarks/cuda/replay_arena.py`; its bookkeeping equals the simulator event for event on the registered cells (checked on the Mac, `benchmarks/cuda/n2s1-expected.json`). The S1, S2 and V1 runbooks apply for the provider, key, private ssh config, cost log and teardown; this file lists what differs. **Nothing here starts an instance.** Hamed creates it and says go for a named live instance; the agent never runs sudo. Cap: **1 GPU-hour** from Ready; the instance takes about seven minutes to initialize, so create it when the agent asks.

## Card and wheel

* RTX 5090: the default pin (CUDA 12.8 wheel, `MIN_DRIVER_MAJOR=570`). V100: `TORCH_INDEX=https://download.pytorch.org/whl/cu126 MIN_DRIVER_MAJOR=525`. The setup's arch gate stops if the wheel has no kernels for the card.
* A 17 % arena is 24.2 GiB: both cards have 31.4 GiB free; 25 and 34 % do not fit and are not run.

## Steps

1. Pack from a clean commit on the Mac: `cd /Users/hamedprooshani/Projects/deepseek-v41-mac && bash benchmarks/cuda/pack.sh` (REVISION must say `dirty=0`); upload and unpack as in S2.
2. Copy the trace (11 MB, not in the repository) to the host: `scp -F ~/cachalot-private/ssh_config_NAME benchmarks/results/routing-trace-deepseek-long/trace.npz NAME:~/cuda-n2/n2-trace.npz`. The host refuses to run if its sha256 differs from the one in `n2s1-expected.json`.
3. `check`, `setup`, `doctor`, `smoke` as before (all clean), then **`bash benchmarks/cuda/session.sh n2smoke`**: two 40-token cells (trace and generator, 17 %). **A GPU smoke of the stream code is required before any stage 1 number counts** (the S2 d2d race: the CPU rehearsal cannot test stream ordering). Read the gates: `event_hash`, `determinism`, `readback`, `sanity` all true, no failed cell, not truncated.
4. Only if the smoke is clean: `bash benchmarks/cuda/session.sh n2full` (four cells: the trace at 8, 12 and 17 %, the generator at 17 %; the 2,400 s cap is a bound, the estimate is 3 to 6 minutes on a 5090 and 6 to 12 on a V100 of copy time a cell group).
5. Copy back with `results`; score by **field name**, never by column position.

## Stop rules

Stop on a failed cell, a false gate, truncation, an event-hash mismatch (a harness bug: fix on the Mac), or 80 % of the cap before `n2full` starts.
