#!/bin/bash
# Post-processing pipeline for Exp 6b after spark1 run completes.
# Run from repo root: bash experiments/atb_validation_v1/exp6b_post_rope_negative_controls/post_process.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP6B_DIR="${REPO_ROOT}/experiments/atb_validation_v1/exp6b_post_rope_negative_controls"
FINAL_REPORT="${REPO_ROOT}/experiments/atb_validation_v1/final_report"
SPARK1_EXP_ROOT="~/projects/RCV-HC/experiments/atb_validation_v1"

echo "=== Step 1: rsync exp6b run dir from spark1 (manifest, summary, tables; skip large jsonl) ==="
rsync -av --progress \
    --exclude="results.jsonl" --exclude="run.log" --exclude="plots/" \
    "spark1:${SPARK1_EXP_ROOT}/exp6b_post_rope_negative_controls/" \
    "${EXP6B_DIR}/"

echo "=== Step 2: scp finalize.py to spark1 and run there (needs all results.jsonl) ==="
scp "${REPO_ROOT}/experiments/atb_validation_v1/finalize.py" \
    "spark1:~/projects/RCV-HC/experiments/atb_validation_v1/finalize.py"

ssh spark1 "cd ~/projects/RCV-HC && \
    ~/projects/RCV-HC/.venv-gb10/bin/python3 \
    experiments/atb_validation_v1/finalize.py \
    --exp-root experiments/atb_validation_v1 \
    --out experiments/atb_validation_v1/final_report 2>&1"

echo "=== Step 3: rsync final_report from spark1 (exclude plots/) ==="
rsync -av --progress --exclude="plots/" \
    "spark1:${SPARK1_EXP_ROOT}/final_report/" \
    "${FINAL_REPORT}/"

echo "=== Step 4: run standalone exp6b analysis locally (uses rsynced summary.csv) ==="
python3 "${EXP6B_DIR}/analyze.py" \
    --exp-dir "${EXP6B_DIR}" \
    --out "${EXP6B_DIR}/analysis" 2>/dev/null || echo "[warn] local analyze.py needs results.jsonl — skipping plots"

echo "=== Step 5: git add + commit ==="
cd "${REPO_ROOT}"
git add \
    "experiments/atb_validation_v1/exp6b_post_rope_negative_controls/" \
    "experiments/atb_validation_v1/final_report/" \
    "docs/atb_validation_v1.md" 2>/dev/null || true

git diff --cached --quiet && echo "[no staged changes]" && exit 0

git commit -m "exp6b: post-RoPE negative controls — final results + report

- Rerun of Exp 6 with bank_key_mode=post_rope (only effective mode on Gemma-4-31B)
- 5 variants: correct_bank / shuffled / random_kv / correct_K_random_V / random_K_correct_V
- Model: Gemma-4-31B-it, CounterFact-1k (807 eligible), seeds 0/1/2
- Key verdict: V6b_post_rope_correct_dominates
- Updated final_report/verdicts.json, README.md, paper_tables/

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

echo "=== Step 6: push ==="
git push origin main

echo "=== Done ==="
