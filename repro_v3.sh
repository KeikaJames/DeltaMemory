#!/usr/bin/env bash
# Reproduce DeltaMemory v3 (Stage 14) end-to-end on Gemma-4-E2B.
#
# This script regenerates every artifact behind the Phase G report:
#   1. holdout splits (deterministic, sha-pinned in eval/splits/manifest.json)
#   2. Stage 14A InfoNCE K-projector training
#   3. Stage 14 dev sweep (frozen v3 + baselines)
#   4. Stage 14 dev re-eval with the trained projector
#   5. Phase G held-out test eval (one-shot, frozen config)
#
# Honest test result (Gemma-4-E2B / MPS bf16):
#   * v3 frozen recall@1 = 0.2778, B0 no_memory = 0.3590  (-8.1pp, p=0.0074)
#   * v3 beats v2 by +27.8pp (projector does something), but does NOT beat
#     no-memory on the held-out test split. See
#     reports/cleanroom/stage14_test_gemma4_e2b/REPORT.md for the full
#     statistical analysis and methodology amendment.
#
# Requirements:
#   * Apple Silicon Mac with MPS, OR a CUDA box
#   * .venv-mac with torch>=2.4, transformers, scipy, accelerate
#   * google/gemma-4-E2B already pulled into HF cache
#
# Override the model with $DM_MODEL and the device with $DM_DEVICE.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

PY="${DM_PYTHON:-${REPO_ROOT}/.venv-mac/bin/python}"
MODEL="${DM_MODEL:-google/gemma-4-E2B}"
DEVICE="${DM_DEVICE:-mps}"

echo "============================================================"
echo "DeltaMemory v3 reproduction"
echo "  python : ${PY}"
echo "  model  : ${MODEL}"
echo "  device : ${DEVICE}"
echo "============================================================"

echo
echo "[1/5] Verifying preregistered splits…"
"${PY}" eval/holdout_split.py --check

echo
echo "[2/5] Training Stage 14A InfoNCE K-projector…"
"${PY}" scripts/train_k_projector.py \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --epochs 12 \
    --out reports/cleanroom/stage14_kproj

echo
echo "[3/5] Running Stage 14 dev sweep (baselines + v3 candidates)…"
"${PY}" scripts/run_stage14_dev_sweep.py \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --out reports/cleanroom/stage14_dev

echo
echo "[4/5] Re-evaluating dev with trained K-projector…"
"${PY}" scripts/run_stage14_dev_with_kproj.py \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --projector reports/cleanroom/stage14_kproj/k_projector.pt \
    --out reports/cleanroom/stage14_dev_kproj

echo
echo "[5/5] Phase G held-out test eval (one-shot, FROZEN v3)…"
echo "      WARNING: per docs/preregistration.md, this consumes the test split."
"${PY}" scripts/run_stage14_test_eval.py \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --projector reports/cleanroom/stage14_kproj/k_projector.pt \
    --out reports/cleanroom/stage14_test_gemma4_e2b

echo
echo "============================================================"
echo "Done. Reports:"
echo "  reports/cleanroom/stage14_kproj/summary.json"
echo "  reports/cleanroom/stage14_dev/REPORT.md"
echo "  reports/cleanroom/stage14_dev_kproj/REPORT.md"
echo "  reports/cleanroom/stage14_test_gemma4_e2b/REPORT.md"
echo "============================================================"
