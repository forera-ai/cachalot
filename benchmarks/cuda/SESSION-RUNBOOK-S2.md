# CUDA session S2 runbook: overlap variants and batched expert copies (RTX 5090)

Predictions and stop rules: `docs/CUDA-S2-EXPERIMENT-RECORD.md` (committed in 3d044d4 before any S2 code). Everything in `SESSION-RUNBOOK.md` about the provider, the key, the private ssh config, the cost file, teardown and "if something goes wrong" applies unchanged; this file lists only what differs. **Nothing here starts an instance.** An instance needs Hamed's yes for this session; the agent runs over SSH only after Hamed instructs it for a specific live instance, and never runs sudo.

Cap: **1 GPU-hour** of billed time, setup included. The S2 profile itself is 13 cells; its estimated upper bound is 29 s and its wall-clock cap 150 s.

## Differences from S1

1. Pack from a clean commit on the Mac: `cd /Users/hamedprooshani/Projects/deepseek-v41-mac && bash benchmarks/cuda/pack.sh` (the REVISION file must say `dirty=0`).
2. On the host: `check`, `setup`, `doctor` and `smoke` as in S1. The smoke run must be clean (no failed cell, all gates true, not truncated) or the S2 profile refuses to start.
3. Instead of `baseline`: `SESSION_TAG=S2 bash benchmarks/cuda/session.sh s2`. It runs the profile `s2` (`--max-seconds 150 --max-alloc-mib 2048`): control, the four overlap variants (`overlap.{resident,large}.{h2d,d2d}`), the batched-copy cells (`batchcopy.k{1,2,4,8}.{single,loop}`) and the closing control.
4. Copy back with `results`, and score the rows by **field name**, never by column position (S1's first summary mislabelled columns). The fields of interest: `compute_slowdown_with_copy` on each overlap cell, `per_expert_ms` and `bandwidth_gb_s` on each batched-copy cell, `control_drift`, and `l2_cache_bytes` if the framework reports it.

## Estimated session time

Provision and upload 10, setup 10 to 15, inventory and smoke 3, S2 profile 3, copy back and teardown 10, reading between steps 10 to 20: about 50 minutes, cap 60.
