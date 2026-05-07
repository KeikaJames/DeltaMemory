#!/usr/bin/env bash
# Exp 2 — pre-RoPE vs post-RoPE position invariance.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/_common.sh"

MODEL=${MODEL:-$GEMMA_31B}
SEEDS=${SEEDS:-"0,1,2"}
N_FACTS=${N_FACTS:-50}

OUT="${OUT_BASE}/exp2_position_invariance/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "[exp2] model=$MODEL seeds=$SEEDS n_facts=$N_FACTS → $OUT"
python experiments/atb_validation_v1/exp2_position_invariance/run.py \
    --model "$MODEL" --dtype "$DTYPE" --device "$DEVICE" \
    --seeds "$SEEDS" --n-facts "$N_FACTS" --out "$OUT"
