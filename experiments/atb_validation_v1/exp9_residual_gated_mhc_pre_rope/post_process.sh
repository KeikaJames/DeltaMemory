#!/usr/bin/env bash
# post_process.sh — collect Exp9 results from spark1, analyze, commit.
# Usage: bash post_process.sh [spark_host]
set -euo pipefail

SPARK="${1:-spark1}"
LOCAL_REPO="$(cd "$(dirname "$0")/../../../" && pwd)"
EXP_REL="experiments/atb_validation_v1/exp9_residual_gated_mhc_pre_rope"
EXP_LOCAL="${LOCAL_REPO}/${EXP_REL}"
ANALYZE="${EXP_LOCAL}/analyze.py"
ANALYSIS_OUT="${EXP_LOCAL}/analysis"

echo "=== Exp9 Post-Process ==="
echo "Local repo: ${LOCAL_REPO}"
echo "Spark host: ${SPARK}"

# 1. rsync full exp9 directory from spark1.
echo ""
echo "--- Rsyncing from ${SPARK} ---"
rsync -av --progress \
    "${SPARK}:~/projects/RCV-HC/${EXP_REL}/" \
    "${EXP_LOCAL}/"

# 2. Run analysis.
echo ""
echo "--- Running analyze.py ---"
cd "${LOCAL_REPO}"
python3 "${ANALYZE}" \
    --exp-dir "${EXP_LOCAL}" \
    --out "${ANALYSIS_OUT}"

# 3. Print verdict.
VERDICT_FILE="${ANALYSIS_OUT}/analysis.json"
if [ -f "${VERDICT_FILE}" ]; then
    echo ""
    echo "--- Phase B Verdict ---"
    python3 -c "
import json, sys
data = json.load(open('${VERDICT_FILE}'))
v = data.get('verdict') or {}
print('verdict:', v.get('verdict', 'N/A'))
print('gap:', v.get('gap', 'N/A'))
print('strict_ci:', v.get('strict_ci_dominance'))
print('v_dominates:', v.get('pattern_v_dominates'))
bc = data.get('best_config') or {}
print('best mode:', bc.get('mode'))
print('best beta:', bc.get('beta'))
"
fi

# 4. Commit.
echo ""
echo "--- Git commit ---"
cd "${LOCAL_REPO}"
git add "${EXP_REL}/"
git commit -m "add exp9 residual-gated mhc results

Exp9: Residual-Gated mHC AttnNativeBank
Modes: merged_beta_mhc, sep_beta_mhc
Phases: A1 (beta grid) → A2 (full controls smoke) → B (807 prompts) → C (alpha stress)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push origin main

echo ""
echo "=== Done. ==="
