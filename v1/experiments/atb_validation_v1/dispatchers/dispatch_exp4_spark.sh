#!/usr/bin/env bash
# Exp 4 — CounterFact-1k main result on Gemma-4-31B-it (paper headline).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/_common.sh"

MODEL=${MODEL:-$GEMMA_31B}
SEEDS=${SEEDS:-"0,1,2"}

OUT="${OUT_BASE}/exp4_cf1k_main/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "[exp4] model=$MODEL seeds=$SEEDS → $OUT"
python experiments/atb_validation_v1/exp4_cf1k_main/run.py \
    --model "$MODEL" --dtype "$DTYPE" --device "$DEVICE" \
    --seeds "$SEEDS" --out "$OUT"
