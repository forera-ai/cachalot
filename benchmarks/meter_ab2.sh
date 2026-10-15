#!/bin/bash
# E3j (docs/E3H-METER-RERUN-RECORD.md, addendum): the corrected meter-cost design. Six DeepSeek server arms in the order off, on, on, off, off, on
# (ABBAAB, so a linear drift cancels), each a fresh server, the warm-up requests, then 60 s of the repeated all-resident prompt at 120 tokens
# (0.0 misses a token, unlike the void 160-token arms of E3h), with the logger beside and Hamed's root powermetrics sampling.
#   cd /Users/hamedprooshani/Projects/deepseek-v41-mac && sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 1500 -o benchmarks/results/energy/e3j-sampler.txt
#   cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./benchmarks/meter_ab2.sh > benchmarks/results/energy/e3j-run.log 2>&1
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
n=0
for arm in off on on off off on; do
  n=$((n+1))
  if [ "$arm" = "on" ]; then export CACHALOT_ENERGY=1; else unset CACHALOT_ENERGY; fi
  echo "arm $n-$arm start $(date +%H:%M:%S)"
  E_TAG=e3j-$n-$arm E_FULL=1 E_MAX_TOKENS=120 E_A=0 E_B=60 E_C=-1 E_D=0 ./benchmarks/energy_arms.sh || { echo "ARM $n-$arm FAILED"; exit 1; }
  echo "arm $n-$arm done $(date +%H:%M:%S)"
  sleep 20
done
echo "ALL ARMS DONE $(date +%H:%M:%S)"
