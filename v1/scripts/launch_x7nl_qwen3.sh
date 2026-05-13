#!/usr/bin/env bash
set -euo pipefail
cd /home/gabira/projects/RCV-HC
source .venv-gb10/bin/activate
mkdir -p runs/X7NL_full_v1_qwen3_27B
nohup python experiments/X7_nonlinear/run.py \
    --model /home/gabira/Desktop/workspace/models/whitelist/Qwen3.6-27B \
    --device cuda --dtype bf16 \
    --out runs/X7NL_full_v1_qwen3_27B \
    --sub all --seeds 0 1 2 \
    > runs/X7NL_full_v1_qwen3_27B/run.log 2>&1 &
echo "PID: $!"
