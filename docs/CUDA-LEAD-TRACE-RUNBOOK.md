# Runbook: record a DeepSeek routing trace with predictions at leads 1, 2 and 3

Record and predictions: `docs/CUDA-LEAD-TRACE-RECORD.md` (read its sections 1 and 3 first). This is for Hamed to run; the agent starts nothing. Estimate: about 10 to 15 minutes, worst case 25. One runtime process at a time: quit anything that runs the model, close heavy applications, display on, sysctl `iogpu.wired_limit_mb=88064`.

Terminal 1 (the server; stop it with Ctrl-C after the replay finishes):

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && mkdir -p benchmarks/results/routing-trace-deepseek-leads && CACHALOT_ROUTING_TRACE=benchmarks/results/routing-trace-deepseek-leads/trace.npz CACHALOT_DECODE_MISS_BUDGET=off CACHALOT_PREDICT_AHEAD=3 CACHALOT_PREDICT_SUBMIT=0 ./serve.sh --expert-budget-gib 44 2>&1 | tee benchmarks/results/routing-trace-deepseek-leads/server.log
```

Terminal 2 (after the server prints that it is ready; the same eight requests as the old trace):

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && cp benchmarks/results/routing-trace-deepseek-long/source-requests.jsonl benchmarks/results/routing-trace-deepseek-leads/ && cp benchmarks/results/routing-trace-deepseek-long/replay_trace.py benchmarks/results/routing-trace-deepseek-leads/ && ~/venvs/deepseek-v41/bin/python benchmarks/results/routing-trace-deepseek-leads/replay_trace.py benchmarks/results/routing-trace-deepseek-leads/source-requests.jsonl 8 2>&1 | tee benchmarks/results/routing-trace-deepseek-leads/replay.log
```

When Terminal 2 finishes, press Ctrl-C in Terminal 1 (the trace is written on exit) and tell the agent. `CACHALOT_PREDICT_SUBMIT=0` is a diagnostic setting: this is not a speed run and the decode rate it prints says nothing about the shipped path. If the sysctl is not 88064 the 44 GiB budget may be refused; say so rather than lowering it.
