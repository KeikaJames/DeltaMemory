#!/usr/bin/env bash
# Mneme v3.1 one-line reproduction script.
#
# Reproduces the core counter-prior intervention result on Gemma-4-E2B:
# writes a false fact ("Python was created by Ada Lovelace") to the bank
# and measures the log-probability lift on the wrong target token.
#
# Prerequisites:
#   - torch with MPS (Apple Silicon) or CUDA (NVIDIA GB10/GPU)
#   - HF_HUB_OFFLINE=1 or internet access for model download
#   - google/gemma-4-E2B in HF cache (~9.6 GB)
#
# Usage:
#   bash repro_v31.sh                         # auto-detect device
#   bash repro_v31.sh --device cuda           # force CUDA
#   bash repro_v31.sh --model Qwen/Qwen3-4B-Instruct-2507  # cross-arch
#
# Expected output (Gemma-4-E2B, MPS bf16, identity-init K-projector):
#   Ada Lovelace target logprob: B0 ~ -12 nats, v3 ~ -9 nats, lift ~ +2.8 nats
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-E2B}"
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-/tmp/deltamemory/transcripts/v31_repro}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

echo "=== Mneme v3.1 reproduction ==="
echo "Model:  $MODEL"
echo "Device: $DEVICE"
echo

if [ "$DEVICE" = "auto" ]; then
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        DEVICE="cuda"
    elif python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        DEVICE="mps"
    else
        DEVICE="cpu"
    fi
fi
echo "Using device: $DEVICE"

$PYTHON scripts/run_intervention_demo.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --dtype bfloat16 \
    --alpha 1.0 \
    --false-facts \
    --capture-policy period \
    --label "v31_repro" \
    --out-dir "$OUT_DIR"

echo
echo "=== Done ==="
echo "Transcript: $OUT_DIR/demo.md"
echo "Raw data:   $OUT_DIR/demo.json"
