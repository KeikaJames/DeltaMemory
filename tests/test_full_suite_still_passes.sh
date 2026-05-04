#!/usr/bin/env bash
# tests/test_full_suite_still_passes.sh
# Run the full DeltaMemory test suite (excluding real-model conservation tests)
# and assert exit 0.
#
# Usage:
#   bash tests/test_full_suite_still_passes.sh
#
# Exit codes:
#   0 — all tests passed
#   1 — one or more tests failed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate macOS venv if present
if [ -f ".venv-mac/bin/activate" ]; then
    # shellcheck source=/dev/null
    source ".venv-mac/bin/activate"
fi

echo "=== DeltaMemory full test suite ==="
echo "Working directory: $(pwd)"
echo "Python: $(python --version 2>&1)"
echo ""

python -m pytest tests/ \
    --ignore=tests/conservation_real_models.py \
    -q \
    "$@"

echo ""
echo "=== All tests passed ==="
