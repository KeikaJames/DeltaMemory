#!/usr/bin/env bash
# DeltaMemory / Project Mneme — Phase v0.4 reproduction script.
#
# Re-executes the methodology spine of the v0.4 release:
#   1. Real datasets sanity   — LAMA T-REx 500, ConceptNet 500, CAA pairs
#   2. W.4  CAA paired smoke   — gpt2-medium 5-prompt + alpha=0 bit-equality
#   3. W.6  counter-prior smoke — gpt2-medium pareto.json verdict
#   4. Aggregate + verdict     — runs aggregate.py for both, prints summary
#
# Out-of-scope by design (require 128 GB or external dataset auth):
#   W.5 MoE (Qwen3-MoE-A3B), W.7 long-ctx 16k+, W.8 multi-fact > 32,
#   W.10 (depends on W.6 verdict), W.13 synthetic full grid,
#   W.14 ROME/MEMIT/GRACE benchmark suite. Each has its own runner.
#
# Prerequisites:
#   - .venv-mac on Apple Silicon (torch 2.11 + MPS) or .venv-cuda equivalent
#   - HF cache populated for: gpt2-medium (mandatory), Qwen2.5-0.5B (W.6
#     uses gpt2-medium only for smoke; full grid needs the others)
#   - google/* repos accepted on the running HF account (gated=manual)
#
# Usage:
#   bash repro_v04.sh                  # full v0.4 smoke (~10 min on M2 Pro)
#   bash repro_v04.sh --quick          # W.4 alpha=0 witness only (~2 min)
#   bash repro_v04.sh --include-full   # additionally launches W.4 full grid

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

QUICK=false
FULL=false
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=true ;;
        --include-full) FULL=true ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [ -d ".venv-mac" ]; then
    # shellcheck disable=SC1091
    source .venv-mac/bin/activate
elif [ -d ".venv-cuda" ]; then
    # shellcheck disable=SC1091
    source .venv-cuda/bin/activate
else
    echo "WARNING: no .venv-mac or .venv-cuda found; using system python3" >&2
fi

PY="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"

if [ "$DEVICE" = "auto" ]; then
    if $PY -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        DEVICE="mps"
    elif $PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi

echo "=================================================================="
echo "  Mneme v0.4 reproduction"
echo "  Device: $DEVICE   Quick: $QUICK   Include full grid: $FULL"
echo "=================================================================="

# ---------------------------------------------------------------------- 1
echo
echo "--- Step 1: dataset sanity ---"
for f in experiments/datasets/lama_trex_500.jsonl \
         experiments/datasets/conceptnet_500.jsonl \
         experiments/datasets/caa_calibration_pairs.jsonl \
         experiments/datasets/gold_30prompts.jsonl \
         experiments/datasets/counterfact_60.jsonl; do
    if [ ! -f "$f" ]; then
        echo "MISSING: $f" >&2
        exit 1
    fi
    n=$(wc -l < "$f")
    echo "  $f   $n rows"
done

# ---------------------------------------------------------------------- 2
echo
echo "--- Step 2: W.4 CAA smoke (alpha=0 bit-equality witness) ---"
$PY experiments/W4_caa_baseline/run.py --smoke \
    --out experiments/W4_caa_baseline/cells_smoke_repro.jsonl \
    --device "$DEVICE" --dtype bfloat16
$PY -c "
import json, sys
rows = [json.loads(l) for l in open('experiments/W4_caa_baseline/cells_smoke_repro.jsonl')]
zeros = [r for r in rows if abs(r.get('alpha', -1)) < 1e-9]
bad = [r for r in zeros if abs(r.get('drift', 0.0)) >= 1e-4]
print(f'  alpha=0 cells: {len(zeros)}, redline violations: {len(bad)}')
if bad:
    for r in bad[:3]:
        print('  VIOLATION:', r)
    sys.exit(3)
"

if [ "$QUICK" = true ]; then
    echo
    echo "Quick mode: stopping after W.4 alpha=0 witness."
    exit 0
fi

# ---------------------------------------------------------------------- 3
echo
echo "--- Step 3: W.6 counter-prior smoke ---"
$PY experiments/W6_counter_prior/run.py --smoke \
    --out experiments/W6_counter_prior/cells_smoke_repro.jsonl \
    --device "$DEVICE" --dtype bfloat16
$PY experiments/W6_counter_prior/aggregate.py \
    --cells experiments/W6_counter_prior/cells_smoke_repro.jsonl \
    --out experiments/W6_counter_prior/pareto_repro.json
$PY -c "
import json
p = json.load(open('experiments/W6_counter_prior/pareto_repro.json'))
print(f'  W.6 smoke pareto entries: {len(p)}')
"

# ---------------------------------------------------------------------- 4
echo
echo "--- Step 4: regression suite ---"
$PY -m pytest -q --ignore=tests/conservation_real_models.py \
                 --ignore=tests/test_moe_adapter.py 2>&1 | tail -3

# ---------------------------------------------------------------------- 5
if [ "$FULL" = true ]; then
    echo
    echo "--- Step 5: W.4 full grid (background; ~6-15 min on MPS, ~2 h on CUDA-A100) ---"
    nohup $PY experiments/W4_caa_baseline/run.py \
        --out experiments/W4_caa_baseline/cells.jsonl \
        --device "$DEVICE" --dtype bfloat16 \
        > experiments/W4_caa_baseline/logs/full_grid.log 2>&1 &
    echo "  PID: $!  log: experiments/W4_caa_baseline/logs/full_grid.log"
    echo "  When done: python experiments/W4_caa_baseline/aggregate.py"
fi

echo
echo "=================================================================="
echo "  v0.4 smoke complete. Inspect:"
echo "    experiments/W4_caa_baseline/cells_smoke_repro.jsonl"
echo "    experiments/W6_counter_prior/pareto_repro.json"
echo "  For each phase's full contract, see PREREG.md in:"
echo "    experiments/W{1..14}_*/PREREG.md"
echo "=================================================================="
