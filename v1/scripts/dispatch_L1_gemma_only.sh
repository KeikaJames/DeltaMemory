#!/usr/bin/env bash
# B5 — L.1 marathon dispatcher for gemma-4-31B only (Track B).
# Override: JUST gemma-4-31B-it, method=lopi_default, 3 seeds, 2000 turns.
# Run when Track A is between models to avoid GPU contention.
#
# Usage on spark1:
#   ssh spark1
#   cd /home/gabira/projects/RCV-HC && source .venv-gb10/bin/activate
#   bash scripts/dispatch_L1_gemma_only.sh
#
# Check GPU free before launching:
#   nvidia-smi --query-gpu=memory.used --format=csv,noheader
set -euo pipefail
cd /home/gabira/projects/RCV-HC
source .venv-gb10/bin/activate

WHITELIST_DIR=/home/gabira/Desktop/workspace/models/whitelist
OUT_BASE=/home/gabira/projects/RCV-HC/runs
MODEL_PATH="${WHITELIST_DIR}/gemma-4-31B-it"
EXP=experiments/L_marathon

TURNS=${TURNS:-2000}
SEEDS=${SEEDS:-"0 1 2"}
METHODS=${METHODS:-"lopi_default"}

echo "[L1-gemma] Checking GPU memory at $(date)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[L1-gemma] ERROR: gemma-4-31B-it not found at $MODEL_PATH"
    exit 1
fi

for method in $METHODS; do
    for seed in $SEEDS; do
        out="${OUT_BASE}/L1_full_v1_gemma4_31B/${method}/seed${seed}"
        mkdir -p "$out"
        echo "=== L.1 gemma4_31B $method seed=$seed turns=$TURNS → $out at $(date) ==="
        python "${EXP}/run.py" \
            --model "$MODEL_PATH" \
            --method "$method" \
            --device cuda --dtype bf16 \
            --seed "$seed" --turns "$TURNS" \
            --inject-facts "${EXP}/facts_3.jsonl" \
            --probe-set "${EXP}/probes_8.jsonl" \
            --filler "${EXP}/filler.txt" \
            --out "$out" --resume
    done
done
echo "[L1-gemma] done at $(date)"
