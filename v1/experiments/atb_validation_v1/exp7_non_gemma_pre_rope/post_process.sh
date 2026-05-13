#!/usr/bin/env bash
# post_process.sh — Rsync Exp7 results from spark1, run analysis, commit.
# Usage:
#   bash experiments/atb_validation_v1/exp7_non_gemma_pre_rope/post_process.sh [spark1_host]
# Defaults spark1_host=spark1

set -euo pipefail

SPARK="${1:-spark1}"
EXP_DIR="experiments/atb_validation_v1/exp7_non_gemma_pre_rope"
ANALYSIS_DIR="${EXP_DIR}/analysis"
REMOTE_DIR="~/projects/RCV-HC/${EXP_DIR}/"

echo "=== [1/4] Rsyncing results from ${SPARK}:${REMOTE_DIR} ==="
rsync -av --exclude='__pycache__' \
    "${SPARK}:${REMOTE_DIR}" "${EXP_DIR}/"

echo ""
echo "=== [2/4] Running analysis ==="
python3 "${EXP_DIR}/analyze.py" \
    --exp-dir "${EXP_DIR}" \
    --out "${ANALYSIS_DIR}"

echo ""
echo "=== [3/4] Updating finalize.py summary ==="
python3 experiments/atb_validation_v1/finalize.py \
    --exp-dir experiments/atb_validation_v1 \
    --out experiments/atb_validation_v1/final_report

echo ""
echo "=== [4/4] Git commit & push ==="
git add \
    "${EXP_DIR}/" \
    experiments/atb_validation_v1/final_report/ \
    experiments/atb_validation_v1/SUMMARY.csv 2>/dev/null || true

git commit -m "add exp7 non-gemma pre-rope negative controls

Model: Qwen3-4B-Instruct-2507
bank_key_mode: pre_rope, bank_size=200, seeds=[0,1,2]
Variants: correct_bank, shuffled_bank, random_kv, correct_K_random_V, random_K_correct_V

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

git push origin main
echo ""
echo "Done."
