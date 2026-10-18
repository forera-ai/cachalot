# CUDA session V1 runbook: V100 32 GB (PCIe 3 and no-fp8 contrast)

Predictions and stop rules: `docs/CUDA-V1-EXPERIMENT-RECORD.md` (committed in d507e14 before any V100 instance or code change). The S1 runbook (`SESSION-RUNBOOK.md`) and `SESSION-RUNBOOK-S2.md` apply for provider, key, private ssh config, cost log and teardown; this file lists what differs. **Nothing here starts an instance.** Hamed creates it; the agent runs over SSH only after Hamed names the live instance, and never runs sudo. Cap: **1 GPU-hour** from Ready. Initialization takes about seven minutes before the console says Ready (S2 note); create the instance when the agent asks, not earlier.

## Differences from S1 and S2

1. **Wheel.** The pinned CUDA 12.8 wheel has no `sm_70` and cannot run on a V100. Use the CUDA 12.6 build of the same torch version and a lower driver floor:
   `TORCH_INDEX=https://download.pytorch.org/whl/cu126 MIN_DRIVER_MAJOR=525 bash benchmarks/cuda/session.sh setup`
   `setup` now ends with an arch gate that stops with an explicit message if the installed torch lists no kernels for the device; if it stops, copy the printed arch list, do not run any cell, tell the agent.
2. **Network.** S2's first VM had no outbound network until a console reboot. Test `curl -sI https://pypi.org | head -1` from the instance before `setup`; if it fails, reboot from the console menu.
3. **Order.** `check`, `setup`, `doctor`, `smoke`, then `baseline` (smoke must be clean), then `SESSION_TAG=V1 bash benchmarks/cuda/session.sh s2` up to four times if the first is clean. All cells already exist and ran on the 5090; bf16 and fp8 are gated off by the device's compute capability (7.0) and must report unavailable, not a slower number.
4. **Tag.** Use `SESSION_TAG=V1` in the environment for every command so notes and run ids say V1.
5. **Score by field name**, never by column position.
