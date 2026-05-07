#!/usr/bin/env bash
# Dispatcher template for ATB validation v1 on spark1 (GB10 / CUDA).
# Sourced by the per-experiment dispatchers; not meant to run directly.
set -euo pipefail
cd "$(dirname "$0")/../../.."        # → repo root
source .venv-gb10/bin/activate

export WHITELIST_DIR=${WHITELIST_DIR:-/home/gabira/Desktop/workspace/models/whitelist}
export GEMMA_31B="${WHITELIST_DIR}/gemma-4-31B-it"
export QWEN36_4B="${WHITELIST_DIR}/Qwen3.6-4B"
export QWEN3_4B="${WHITELIST_DIR}/Qwen3-4B"
export GLM4_9B="${WHITELIST_DIR}/GLM-4-9B"
export GEMMA_E2B="${WHITELIST_DIR}/gemma-4-E2B"

export OUT_BASE=${OUT_BASE:-experiments/atb_validation_v1}
export DTYPE=${DTYPE:-bf16}
export DEVICE=${DEVICE:-cuda}
