#!/bin/bash
# E3b (docs/E3B-REFIT-RECORD.md): one session, one sampler. Stage 1: the E1b synthetic steps with every Energy Model channel and the SMC keys logged
# (residual_test.py --full refit, ~8.5 min). Stage 2: the DeepSeek energy arms with the same logging (energy_arms.sh, E_FULL=1, ~9 min).
# Hamed starts `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 1300 -o benchmarks/results/energy/e3b-sampler.txt` first
# (never the `tasks` sampler) and closes Cachalot Lab. The two stages never overlap (the server's one-runtime guard).
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
D=benchmarks/results/energy
echo "stage 1 start $(date +%H:%M:%S)"
"$HOME/venvs/deepseek-v41/bin/python" benchmarks/residual_test.py --full refit > $D/e3b-stage1.log 2>&1 || { echo "STAGE 1 FAILED"; tail -5 $D/e3b-stage1.log; exit 1; }
echo "stage 1 done $(date +%H:%M:%S)"
sleep 20
E_FULL=1 ./benchmarks/energy_arms.sh > $D/e3b-stage2.log 2>&1 || { echo "STAGE 2 FAILED"; tail -5 $D/e3b-stage2.log; exit 1; }
echo "stage 2 done $(date +%H:%M:%S)"
