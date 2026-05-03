#!/usr/bin/env bash
# Phase mHC1.6 — GB10 全量 finetune dispatch
# 在 GB10 上跑 hc + mhc 两个臂的 Wikitext-2 finetune，
# 让 projector 学到非 identity 的 mixing matrix，
# 之后才能用 scripts/run_mHC2_perturbation_sweep.py 做谱护盾的真实测试。
#
# 用法（在 GB10 上）：
#   cd ~/projects/RCV-HC
#   git pull
#   bash scripts/run_mHC1_6_finetune_gb10.sh
#
# 预计：GPT-2 small × 20k step × bs=8 × seq=1024，单 GPU 约 1-2 小时。
set -euo pipefail

PY="${PY:-.venv-gb10/bin/python3}"
OUT="${OUT:-reports/cleanroom/mHC1_6_finetune}"

"$PY" scripts/finetune_mhc_wikitext2.py \
    --base-model gpt2 \
    --device cuda \
    --dtype bfloat16 \
    --archs mhc hc \
    --max-steps 20000 \
    --batch-size 8 \
    --segment-length 1024 \
    --lr 1e-3 \
    --log-every 100 \
    --out-dir "$OUT"

echo "[done] checkpoints in $OUT/{mhc,hc}/state_dict.pt"
