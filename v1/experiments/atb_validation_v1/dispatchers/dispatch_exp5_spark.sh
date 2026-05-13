#!/usr/bin/env bash
# Exp 5 — α dense sweep on Gemma-4-31B-it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/_common.sh"

MODEL=${MODEL:-$GEMMA_31B}
SEEDS=${SEEDS:-"0,1,2"}
BANK_SIZE=${BANK_SIZE:-200}

OUT="${OUT_BASE}/exp5_alpha_sweep/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "[exp5] model=$MODEL bank=$BANK_SIZE seeds=$SEEDS → $OUT"
python experiments/atb_validation_v1/exp5_alpha_sweep/run.py \
    --model "$MODEL" --dtype "$DTYPE" --device "$DEVICE" \
    --seeds "$SEEDS" --bank-size "$BANK_SIZE" --out "$OUT"
