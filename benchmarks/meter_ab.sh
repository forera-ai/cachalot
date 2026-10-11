#!/bin/bash
# E3h (docs/E3H-METER-RERUN-RECORD.md): four DeepSeek server arms in the order off, on, on, off (ABBA), each: a fresh server, the warm-up requests, then 130 s of
# the repeated all-resident prompt at 160 tokens, with the logger beside and Hamed's root powermetrics sampling.
#   cd /Users/hamedprooshani/Projects/deepseek-v41-mac && sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 1300 -o benchmarks/results/energy/e3h-sampler.txt
#   cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./benchmarks/meter_ab.sh > benchmarks/results/energy/e3h-run.log 2>&1
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
for arm in 1-off 2-on 3-on 4-off; do
  if [ "${arm#*-}" = "on" ]; then export CACHALOT_ENERGY=1; else unset CACHALOT_ENERGY; fi
  echo "arm $arm start $(date +%H:%M:%S)"
  E_TAG=e3h-$arm E_FULL=1 E_MAX_TOKENS=160 E_A=0 E_B=130 E_C=-1 E_D=0 ./benchmarks/energy_arms.sh || { echo "ARM $arm FAILED"; exit 1; }
  echo "arm $arm done $(date +%H:%M:%S)"
  sleep 20
done
echo "ALL ARMS DONE $(date +%H:%M:%S)"
