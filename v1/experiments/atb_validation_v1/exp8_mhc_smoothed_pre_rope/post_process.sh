#!/usr/bin/env bash
# post_process.sh — run after rsyncing Exp8 results from spark1
# Usage: bash experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/post_process.sh <run_dir>
# Example:
#   rsync -av spark1:~/projects/RCV-HC/experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/ \
#       experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/
#   bash experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/post_process.sh \
#       experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/run_YYYYMMDD_HHMMSS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON="${ROOT}/.venv-mac/bin/python3"

if [[ $# -lt 1 ]]; then
    # Auto-detect latest run
    RUN_DIR=$(ls -dt "$SCRIPT_DIR"/run_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "Usage: $0 <run_dir>" >&2
        exit 1
    fi
    echo "[post_process] Auto-selected run_dir: $RUN_DIR"
else
    RUN_DIR="$1"
fi

OUT_DIR="$SCRIPT_DIR/analysis"
mkdir -p "$OUT_DIR"

echo "[post_process] Analyzing $RUN_DIR → $OUT_DIR"

"$PYTHON" "$SCRIPT_DIR/analyze.py" \
    --run-dir "$RUN_DIR" \
    --out "$OUT_DIR"

echo "[post_process] Done. Results in $OUT_DIR"
echo "[post_process] To commit:"
echo "  git add experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/"
echo "  git commit -m 'add exp8 mhc-smoothed pre-rope results'"
echo "  git push"
