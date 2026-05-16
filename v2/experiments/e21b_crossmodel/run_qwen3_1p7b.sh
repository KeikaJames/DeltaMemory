#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
python3 -u v2/experiments/e21_counterfactual_injection/run.py \
  --model Qwen/Qwen3-1.7B --bank_layer 14 \
  > v2/experiments/e21b_crossmodel/qwen3_1p7b.log 2>&1
cp v2/experiments/e21_counterfactual_injection/results.json \
   v2/experiments/e21b_crossmodel/qwen3_1p7b_results.json
