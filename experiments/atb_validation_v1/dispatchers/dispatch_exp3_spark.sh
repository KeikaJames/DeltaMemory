#!/usr/bin/env bash
# Exp 3 — α=0 bit-equality across 3 model classes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/_common.sh"

# 3 distinct architectures from the on-disk whitelist:
#   Gemma-4-31B-it, Qwen3-4B-Instruct-2507, Llama-3.1-8B-Instruct.
# Bit-equality is architecture-agnostic — any 3 distinct families suffice.
declare -a MODELS=(
    "gemma_31B|$GEMMA_31B"
    "qwen_4B|$QWEN3_4B"
    "llama_8B|$LLAMA_8B"
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
