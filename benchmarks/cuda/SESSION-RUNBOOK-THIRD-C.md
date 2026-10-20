# CUDA third-C session runbook: the compute-slowdown slope at C = 0.6 ms (RTX 5090)

Predictions, design and stop rules: `docs/CUDA-THIRD-C-RECORD.md` (nine predictions G-1 to G-9, committed first in 32bd9e2; erratum in section 7 before any run). The harness is the row-timing file with two new profiles: `g-smoke` (8 cells, 40 tokens) and `g-full` (12 cells a run: levels 3 with per-row records, 6 and 4 at C = 0.6 ms only, arms `N`, `D`, `P0L`, `N`, 400 decode tokens). The earlier runbooks apply for the provider, key, private ssh config, cost log and teardown. **Nothing here starts an instance.** Hamed creates it and says go for the named live instance; the agent runs over SSH only for that one instance and never runs sudo. Cap: **1 GPU-hour** from Ready; every run passes an explicit `--max-seconds` (600 a run by default). Estimate: 4 to 6 minutes a run plus calibration and arena allocation, 12 to 18 minutes of GPU time for the smoke and both runs.

Which card: an **RTX 5090**, one instance, the card of every earlier stage. Only the **new multi-lead trace** is needed (`n3d-trace.npz`, 39.8 MB).

## Steps

1. Pack from a clean commit on the Mac (`bash benchmarks/cuda/pack.sh`, REVISION `dirty=0`); upload and unpack as before. Copy the trace: `scp -F ~/cachalot-private/ssh_config_NAME benchmarks/results/routing-trace-deepseek-leads/trace.npz NAME:~/cachalot-cuda/n3d-trace.npz`. The run refuses a trace whose sha256 differs from `n3d-expected.json`.
2. **First check the instance**: `nvidia-smi` must show the GPU and driver, and `getent hosts pypi.org` must resolve. A GPU that does not initialize is replaced, and the replacement is created **before** the faulty instance is terminated. Then `check`, `setup`, `doctor`, `smoke` (all clean). Note the source qualification of the first run: a median source-record copy time of 0.30 ms or more means a slow-population instance, and the record's section 5 slow-population rule applies (report; ask Hamed whether to replace the instance before the full run).
3. **`bash benchmarks/cuda/session.sh n3gsmoke`**. Read every gate of the run manifest: `event_hash`, `determinism`, `counts`, `chain2`, `readback`, `calibration`, `drift`, `closure`, `sanity`, `floor`, `rowsum`, `copyevents` (true), no failed cell, not truncated. The `floor` gate is the check that the stand-in is tuned to C = 0.6 ms.
4. Only if the smoke is clean: `MAXS=600 bash benchmarks/cuda/session.sh n3gfull` (two runs, A then B, 12 cells each). A truncated run is reported as truncated.
5. Copy back with `results` (compress the raw samples first: the instance link is about 30 KB/s), then on the Mac, with the two run directories (each holds `raw-samples.json`):
   `~/venvs/deepseek-v41/bin/python benchmarks/cuda/third_c_score.py RUN_A_DIR RUN_B_DIR --json docs/cuda-results/third-c-DATE.json`
   The script prints G-1 to G-9 beside the registered intervals and the three shapes on both slope bases (record section 7). Score by hand in `docs/CUDA-THIRD-C-RESULTS.md`.

## Stop rules

As the record, sections 5 and 6. Housekeeping stands (delete the ssh configs, known_hosts files and key of every earlier instance; enter the billed hours).
