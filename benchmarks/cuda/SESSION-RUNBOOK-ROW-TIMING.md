# CUDA row-timing session runbook: per-row and per-copy timing of a miss (RTX 5090)

Predictions, design and stop rules: `docs/CUDA-ROW-TIMING-RECORD.md` (committed in 6502a50, erratum in 21b65ac, both before any code). The harness is the stage D file with two additions: per-row event records (level L3 with the profile flag `rows`) and level L6 (L3 plus one CUDA event after every enqueued copy). Profiles `f-smoke` (8 cells, 40 tokens) and `f-full` (24 cells a run: L3, L6 and L4 at C = 1.0 and 0.3 ms, arms `N`, `D`, `P0L`, `N`, 400 decode tokens). The earlier runbooks apply for the provider, key, private ssh config, cost log and teardown. **Nothing here starts an instance.** Hamed creates it and says go for the named live instance; the agent runs over SSH only for that one instance and never runs sudo. Cap: **1 GPU-hour** from Ready, and every run passes an explicit `--max-seconds`.

Which card: an **RTX 5090** (the card of every earlier stage; the new per-row cells compare with stages C, D and the bridge). One instance. Only the **new multi-lead trace** is needed (`n3d-trace.npz`, 39.8 MB); the old trace is not used.

## Steps

1. Pack from a clean commit on the Mac (`bash benchmarks/cuda/pack.sh`, REVISION `dirty=0`); upload and unpack as before. Copy the trace: `scp -F ~/cachalot-private/ssh_config_NAME benchmarks/results/routing-trace-deepseek-leads/trace.npz NAME:~/cachalot-cuda/n3d-trace.npz`. The run refuses a trace whose sha256 differs from `n3d-expected.json`.
2. **First check the instance**: `nvidia-smi` must show the GPU and driver, and `getent hosts pypi.org` must resolve. A GPU that does not initialize is replaced, and the replacement is created **before** the faulty instance is terminated. Then `check`, `setup`, `doctor`, `smoke` (all clean). Note the source qualification of the first run: a median source-record copy time of 0.30 ms or more means a slow-population instance, and the record's section 5 rule applies (report; ask Hamed whether to replace the instance before the full run).
3. **`bash benchmarks/cuda/session.sh n3fsmoke`**. Read every gate of the run manifest: `event_hash`, `determinism`, `counts`, `chain2`, `readback`, `calibration`, `drift`, `closure`, `sanity`, and the two new ones **`rowsum`** and **`copyevents`** (true), no failed cell, not truncated.
4. Only if the smoke is clean: `MAXS=900 bash benchmarks/cuda/session.sh n3ffull` (two runs, A then B, 24 cells each; estimate 6 to 8 minutes a run plus calibration and arena allocation). A truncated run is reported as truncated.
5. Copy back with `results`, then on the Mac, with the two run directories (each holds `raw-samples.json`):
   `~/venvs/deepseek-v41/bin/python benchmarks/cuda/row_timing.py RUN_A_DIR RUN_B_DIR --json docs/cuda-results/row-timing-DATE.json`
   The script prints the quantities of RT-1 to RT-7 beside the registered intervals. Score from it by hand in `docs/CUDA-ROW-TIMING-RESULTS.md`.

## Stop rules

As the record, sections 5 and 6. Housekeeping stands (delete the ssh configs, known_hosts files and key of every earlier instance; enter the billed hours).
