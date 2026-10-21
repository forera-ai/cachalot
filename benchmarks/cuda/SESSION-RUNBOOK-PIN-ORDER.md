# CUDA follow-up M runbook: why an allocate-then-copy pinned buffer is slow, and the huge-page, route and classification answers in the fast pattern (RTX 5090)

Predictions and stop rules: `docs/CUDA-PIN-ORDER-RECORD.md` (nine predictions M-1 to M-9, committed first in debaddb). Audit that led to it: `docs/CUDA-PIN-PATTERN-AUDIT.md`. Instruments: `benchmarks/cuda/pin_route.py` groups `m0`, `m1`, `m4`, `m6`, `m7`, `m8`, `m9` (and the reused controls `k4` and `l1`), `session.sh pinroute3`, `pin_route_order_score.py`. The session K and follow-up L runbooks apply for the provider, key, private ssh config, cost log and teardown. **Nothing here starts an instance.** Hamed creates it and says go for the named live instance; the agent runs over SSH only for that one instance and never runs sudo. Cap: **1 GPU-hour** from Ready; GPU run time is under 2 minutes (nine processes of 1 to 2 s and the 25 s smoke).

Which card: an **RTX 5090 32 GB** (the Korean listing). The trace `n3d-trace.npz` is needed only for the smoke (the arena cross-check M-8).

## Steps

1. Pack from a clean commit on the Mac: `cd /Users/hamedprooshani/Projects/deepseek-v41-mac && bash benchmarks/cuda/pack.sh` (REVISION `dirty=0`); upload and unpack as before; copy the trace as in the session K runbook.
2. **First check the instance**: `nvidia-smi` must show the GPU and driver **and `lsmod | grep nvidia` must show the module** (a GPU on the PCI bus with no module, as on `giant-scarf-3095`, is replaced; create the replacement **before** terminating the faulty one). Then `check`, `setup`, `doctor`, `smoke` (all clean).
3. **`bash benchmarks/cuda/session.sh pinroute3`**: `m0`, `k4`, `l1`, `m1`, `m4`, `m6`, `m7`, `m8`, `m9`, each its own process, stopping at the first failure. Look at `k4` and `l1` first: M-1 needs the interleaved control slow and the batch-first control fast on this instance; if they do not separate, say so (M-2 to M-7 are then not testable) before spending anything else.
4. **`bash benchmarks/cuda/session.sh n3ksmoke`** (all gates true; note the `source_copy_ms_median` row for M-8).
5. Copy back with `results` and the follow-up directory (`tar -czf ~/order.tar.gz -C benchmarks/results cuda-order`), verify the hashes, then on the Mac:
   `~/venvs/deepseek-v41/bin/python benchmarks/cuda/pin_route_order_score.py ORDER_DIR --smoke N3K_SMOKE_RUN_DIR --json docs/cuda-results/pin-order-DATE.json`
   Score by hand in `docs/CUDA-PIN-ORDER-RESULTS.md`.

**Waiting on a run:** never `pgrep -f` inside the ssh command; use `ps aux | grep '[p]in_route'` or read the run directory.

## Stop rules

As the record, section 5. Housekeeping stands.
