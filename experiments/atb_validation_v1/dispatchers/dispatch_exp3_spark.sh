#!/usr/bin/env bash
# Exp 3 — α=0 bit-equality across 3 model classes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/_common.sh"

# Prefer Qwen3.6-4B; fall back to Qwen3-4B if not on disk.
QWEN_PATH="$QWEN36_4B"
[[ -d "$QWEN_PATH" ]] || QWEN_PATH="$QWEN3_4B"

declare -a MODELS=(
    "gemma_E2B|$GEMMA_E2B"
    "qwen_4B|$QWEN_PATH"
    "glm_9B|$GLM4_9B"
)

STAMP=$(date +%Y%m%d_%H%M%S)
for entry in "${MODELS[@]}"; do
    tag="${entry%%|*}"
    model="${entry##*|}"
    if [[ ! -d "$model" ]]; then
        echo "[exp3] skip $tag (not on disk: $model)"
        continue
    fi
    OUT="${OUT_BASE}/exp3_bit_equal/${tag}_${STAMP}"
    mkdir -p "$OUT"
    echo "[exp3] $tag $model → $OUT"
    python experiments/atb_validation_v1/exp3_bit_equal/run.py \
        --model "$model" --dtype "$DTYPE" --device "$DEVICE" \
        --n-prompts 100 --n-facts 8 --out "$OUT"
done
