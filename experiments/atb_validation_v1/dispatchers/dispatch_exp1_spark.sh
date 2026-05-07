#!/usr/bin/env bash
# Exp 1 — Core ablation on Gemma-4-31B-it (paper-primary).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/_common.sh"

MODEL=${MODEL:-$GEMMA_31B}
SEEDS=${SEEDS:-"0,1,2"}
N_PROMPTS=${N_PROMPTS:-200}

OUT="${OUT_BASE}/exp1_core_ablation/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "[exp1] model=$MODEL seeds=$SEEDS n=$N_PROMPTS → $OUT"
python experiments/atb_validation_v1/exp1_core_ablation/run.py \
    --model "$MODEL" --dtype "$DTYPE" --device "$DEVICE" \
    --seeds "$SEEDS" --n-prompts "$N_PROMPTS" --out "$OUT"
