#!/bin/bash
# Exp26a — V-site sweep at fixed K=relation_last, α∈{0.005,0.010,0.020}
# Variants: subject_last (baseline), object_last, period (end-token-like), relation_last (control: K==V).
set -e
cd /Users/gabiri/projects/RCV-HC
PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
for SV in object_last period relation_last subject_last; do
  OUT=experiments/atb_validation_v1/exp13_anb_readdressability/run_mps_exp26_V${SV}
  echo "===== V-site = $SV ====="
  rm -rf "$OUT"
  $PY -m experiments.atb_validation_v1.exp13_anb_readdressability.run_exp25 \
    --phase alpha --n 100 --seeds 0,1,2 --alphas 0.005,0.010,0.020 --bank-size 100 \
    --site-k relation_last --site-v $SV \
    --out "$OUT" 2>&1 | tail -3
done
echo "===== ALL DONE ====="
