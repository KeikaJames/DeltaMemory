#!/bin/bash
# Post-processing pipeline for Exp 6b after spark1 run completes.
# Run from repo root: bash experiments/atb_validation_v1/exp6b_post_rope_negative_controls/post_process.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP6B_DIR="${REPO_ROOT}/experiments/atb_validation_v1/exp6b_post_rope_negative_controls"
ARCHIVE_DIR="${REPO_ROOT}/experiments/atb_validation_v1/results_archive/exp6b_post_rope_negative_controls"
FINAL_REPORT="${REPO_ROOT}/experiments/atb_validation_v1/final_report"

echo "=== Step 1: rsync exp6b results from spark1 ==="
rsync -av --progress \
    "spark1:~/projects/RCV-HC/experiments/atb_validation_v1/exp6b_post_rope_negative_controls/" \
    "${EXP6B_DIR}/"

echo "=== Step 2: standalone exp6b analysis ==="
python3 "${EXP6B_DIR}/analyze.py" \
    --exp-dir "${EXP6B_DIR}" \
    --out "${EXP6B_DIR}/analysis"

echo "=== Step 3: also archive to results_archive ==="
mkdir -p "${ARCHIVE_DIR}"
rsync -av "${EXP6B_DIR}/" "${ARCHIVE_DIR}/"

echo "=== Step 4: re-run finalize.py for updated final_report ==="
python3 "${REPO_ROOT}/experiments/atb_validation_v1/finalize.py" \
    --exp-root "${REPO_ROOT}/experiments/atb_validation_v1" \
    --out "${FINAL_REPORT}"

echo "=== Step 5: git commit ==="
cd "${REPO_ROOT}"
git add \
    "experiments/atb_validation_v1/exp6b_post_rope_negative_controls/analyze.py" \
    "experiments/atb_validation_v1/exp6b_post_rope_negative_controls/analysis/" \
    "experiments/atb_validation_v1/finalize.py" \
    "experiments/atb_validation_v1/final_report/" \
    "docs/atb_validation_v1.md" 2>/dev/null || true

# Stage any new run_* dirs too (but not large jsonl — they are gitignored)
git add "experiments/atb_validation_v1/exp6b_post_rope_negative_controls/run_*/manifest.yaml" \
        "experiments/atb_validation_v1/exp6b_post_rope_negative_controls/run_*/summary.json" \
        "experiments/atb_validation_v1/exp6b_post_rope_negative_controls/run_*/summary.csv" \
        2>/dev/null || true

git diff --cached --quiet && echo "[no staged changes]" && exit 0

git commit -m "exp6b: post-RoPE negative controls results + analysis

- Rerun of Exp 6 with bank_key_mode=post_rope
- 5 variants: correct_bank / shuffled / random_kv / correct_K_random_V / random_K_correct_V
- Model: Gemma-4-31B-it, CounterFact-1k (807 eligible), 3 seeds
- Analysis: analyze.py + finalize.py updated with exp6b section
- Key verdict: V6b_post_rope_correct_dominates

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

echo "=== Done ==="
