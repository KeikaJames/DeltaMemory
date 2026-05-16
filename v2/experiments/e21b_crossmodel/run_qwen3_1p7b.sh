#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
python3 -u v2/experiments/e21b_crossmodel/run.py \
  --model Qwen/Qwen3-1.7B --bank_layer 18 --steps 500 --lr 1e-2 \
  > v2/experiments/e21b_crossmodel/qwen3_1p7b.log 2>&1
