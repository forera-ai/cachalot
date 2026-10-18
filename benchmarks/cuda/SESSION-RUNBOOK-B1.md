# CUDA session B1 runbook: pinned-buffer backing and a qualified copy curve (RTX 5090)

Predictions and stop rules: `docs/CUDA-BACKING-EXPERIMENT-RECORD.md` (committed in bc72198 before any instrument code). Instrument: `benchmarks/cuda/backing_probe.py`. The S1, S2, V1 and N2 runbooks apply for the provider, key, private ssh config, cost log and teardown; this file lists what differs. **Nothing here starts an instance.** Hamed creates it when the agent says it is ready (about seven minutes to initialize) and says go for the named live instance; the agent runs over SSH only for that one instance and never runs sudo. Cap: **1 GPU-hour** from Ready; the instrument itself runs 60 to 120 s a process.

## Differences

1. Same listing as S2 and N2 (RTX 5090, 32 GB, 24 GB RAM, 800 GB disk) and the same key; user `riftuser`; port from the console.
2. Pack from a clean commit on the Mac: `cd /Users/hamedprooshani/Projects/deepseek-v41-mac && bash benchmarks/cuda/pack.sh` (REVISION must say `dirty=0`); no trace file is needed.
3. On the host: `check`, `setup`, `doctor`, `smoke` as before, then **`bash benchmarks/cuda/session.sh backing`**: four separate processes in order (`b1` twice, `rest`, `curve`), stopping at the first failure. The first process is a fresh-process allocation probe and is the GPU smoke of the new instrument; read its gates before the later steps matter.
4. Read the huge-page cells with care: B-P4 must hold (at least 14 of 16 buffers show `AnonHugePages` at 90 % or more of their registered range) before B-P5 says anything about H-backing; if `madvise` or `cudaHostRegister` fails the cell reports `unavailable` with the error and the session continues. The registered range is the 9.95 MB rounded up to whole 2 MiB extents (10 MiB), so the 90 % rule is over that range.
5. Copy back with `results`; score by **field name**, never by column position. The `raw-samples.json` of each run carries every per-buffer time.

## Stop rules

Stop on a failed cell, a false gate, truncation, or 80 % of the cap before the first process. A defect found on the instance is fixed on the Mac.
