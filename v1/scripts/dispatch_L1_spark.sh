#!/usr/bin/env bash
# Sequential L.1 marathon dispatcher across whitelist archs on spark1.
# Schedule: gemma-4-31B-it → Qwen3.6-27B → gpt-oss-120b
# Per arch: 2 methods (lopi_default, caa) × 3 seeds × turns=2000.
# Run AFTER X.7-NL has completed (GPU contention).
set -euo pipefail
cd /home/gabira/projects/RCV-HC
source .venv-gb10/bin/activate

WHITELIST_DIR=/home/gabira/Desktop/workspace/models/whitelist
OUT_BASE=/home/gabira/projects/RCV-HC/runs

declare -A MODELS=(
    [gemma4_31B]="${WHITELIST_DIR}/gemma-4-31B-it"
    [qwen36_27B]="${WHITELIST_DIR}/Qwen3.6-27B"
    [gpt_oss_120B]="${WHITELIST_DIR}/gpt-oss-120b"
)

EXP=experiments/L_marathon
TURNS=${TURNS:-2000}
SEEDS=${SEEDS:-"0 1 2"}
METHODS=${METHODS:-"lopi_default caa"}

for tag in gemma4_31B qwen36_27B gpt_oss_120B; do
    model_path="${MODELS[$tag]}"
    if [[ ! -d "$model_path" ]]; then
        echo "[L1] skip $tag (not found)"
        continue
    fi
    for method in $METHODS; do
        for seed in $SEEDS; do
            out="${OUT_BASE}/L1_full_v1_${tag}/${method}/seed${seed}"
            mkdir -p "$out"
            echo "=== L.1 $tag $method seed=$seed turns=$TURNS → $out at $(date) ==="
            python "${EXP}/run.py" \
                --model "$model_path" \
                --method "$method" \
                --device cuda --dtype bf16 \
                --seed "$seed" --turns "$TURNS" \
                --inject-facts "${EXP}/facts_3.jsonl" \
                --probe-set "${EXP}/probes_8.jsonl" \
                --filler "${EXP}/filler.txt" \
                --out "$out" --resume
        done
    done
done
echo "[L1] done at $(date)"
