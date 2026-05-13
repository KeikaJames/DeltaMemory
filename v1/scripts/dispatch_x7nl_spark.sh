#!/usr/bin/env bash
# Dispatch X.7-NL non-linear study on spark1 GB10 across whitelist archs.
# Runs sequentially: gemma-4-31B-it → Qwen3.6-27B → (optional) gpt-oss-120b.
# Each run: A (bank-scaling, 18 cells) + B (alpha-sweep, 123 cells) +
# C (multi-turn, 150 cells) = 291 cells per arch.
set -euo pipefail

REPO=/home/gabira/projects/RCV-HC
MODELS=/home/gabira/Desktop/workspace/models/whitelist
RUNS=$REPO/runs

cd $REPO
source .venv-gb10/bin/activate

dispatch() {
    local arch_dir=$1
    local out_tag=$2
    local model_path=$MODELS/$arch_dir
    if [ ! -d "$model_path" ]; then
        echo "[skip] $arch_dir not on spark yet"
        return 1
    fi
    local out=$RUNS/X7NL_full_v1_${out_tag}
    mkdir -p "$out"
    local log=$out/run.log
    echo "=== X.7-NL on $arch_dir → $out at $(date) ==="
    python experiments/X7_nonlinear/run.py \
        --model "$model_path" \
        --device cuda --dtype bf16 \
        --out "$out" \
        --sub all \
        --seeds 0 1 2 \
        2>&1 | tee "$log"
    python experiments/X7_nonlinear/aggregate.py --run-dir "$out" 2>&1 | tee -a "$log"
    echo "=== $arch_dir DONE at $(date) ==="
}

dispatch gemma-4-31B-it gemma4_31B || true
dispatch Qwen3.6-27B   qwen3_6_27B || true
# gpt-oss-120b: only if MXFP4 native runs out-of-the-box on transformers 5.7
dispatch gpt-oss-120b  gpt_oss_120b || true

echo "=== ALL X.7-NL DISPATCHES COMPLETE at $(date) ==="
