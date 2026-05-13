#!/usr/bin/env bash
set -euo pipefail
SNAPSHOT="/home/gabira/.cache/huggingface/hub/models--kernels-community--gpt-oss-triton-kernels/snapshots/642d2a71c6a9b32fc18acb8ec53505fb324bc394"
export LOCAL_KERNELS="kernels-community/gpt-oss-triton-kernels=${SNAPSHOT}"
cd /home/gabira/projects/RCV-HC
source .venv-gb10/bin/activate
mkdir -p runs/X7NL_full_v1_gpt_oss_120b
nohup python experiments/X7_nonlinear/run.py \
    --model /home/gabira/Desktop/workspace/models/whitelist/gpt-oss-120b \
    --device cuda --dtype bf16 \
    --out runs/X7NL_full_v1_gpt_oss_120b \
    --sub all --seeds 0 1 2 \
    > runs/X7NL_full_v1_gpt_oss_120b/run.log 2>&1 &
echo "PID: $!"
