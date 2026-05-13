#!/bin/bash
# Exp25 bank-size sweep: N ∈ {32, 64, 200, 400} at best α=0.005.
# Skip N=100 (already in run_mps_exp25_alpha) and N=807 (run separately due to size).
set -e
cd /Users/gabiri/projects/RCV-HC
PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
for N in 32 64 200 400; do
  OUT=experiments/atb_validation_v1/exp13_anb_readdressability/run_mps_exp25_N${N}
  echo "===== Bank N=$N ====="
  rm -rf "$OUT"
  $PY -m experiments.atb_validation_v1.exp13_anb_readdressability.run_exp25 \
    --phase alpha --n $N --seeds 0,1,2 --alphas 0.005 --bank-size $N \
    --out "$OUT" 2>&1 | tail -3
done
echo "===== ALL DONE ====="
