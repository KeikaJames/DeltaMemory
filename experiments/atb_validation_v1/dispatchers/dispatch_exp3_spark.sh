#!/usr/bin/env bash
# Exp 3 — α=0 bit-equality across 3 model classes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/_common.sh"

# Three largest distinct-architecture models that fit bf16 on GB10 (128GB unified):
#   Gemma-4-31B-it (~62GB), Qwen3.6-27B (~54GB), Llama-3.1-8B-Instruct (~16GB).
# gpt-oss-120b is excluded — bf16 ≈ 240GB exceeds GB10; quantising would break
# bit-equality semantics. Bit-equality is architecture-agnostic; 3 distinct
# families are sufficient evidence.
declare -a MODELS=(
    "gemma_31B|$GEMMA_31B"
    "qwen36_27B|$QWEN36_27B"
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
