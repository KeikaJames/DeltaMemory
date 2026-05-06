#!/usr/bin/env bash
# A.2 ablation per-method dispatch.
#
# Per the A.2 finding (experiments/A_ablation/FINDING_arm_method_mismatch.md),
# different arms patch different injector hot paths. Running the full 8-arm
# matrix under one --method default makes 6/8 arms no-op. This script runs
# each arm against its DESIGNED test-vehicle method, so every arm actually
# bites.
#
# Mapping (per FINDING):
#   control + A5         -> --method caa            (CAA hot path)
#   A4                   -> --method scar           (SCAR M_perp ablation)
#   A3 + A6              -> --method lopi_default   (LOPI gate / profiler)
#   A1 + A2              -> --method attn_native    (bank-attn read path)
#   A7                   -> NEEDS REDESIGN (currently numerically equivalent
#                          to control on CAA path; see FINDING).
#
# Usage (on GB10):
#   bash scripts/run_A_per_method.sh \
#     --models "Qwen/Qwen3-4B-Instruct-2507" \
#     --out-root runs/A_per_method_v1 \
#     --n-prompts 30 --seeds 0 1 2

set -euo pipefail

PY="${PY:-.venv-gb10/bin/python3}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
N_PROMPTS="${N_PROMPTS:-30}"
SEEDS_DEFAULT="0 1 2"
SEEDS="${SEEDS:-$SEEDS_DEFAULT}"
ALPHAS_DEFAULT="0.0 1.0"
ALPHAS="${ALPHAS:-$ALPHAS_DEFAULT}"
MODELS_DEFAULT="Qwen/Qwen3-4B-Instruct-2507"
MODELS="${MODELS:-$MODELS_DEFAULT}"
OUT_ROOT="${OUT_ROOT:-runs/A_per_method_v1}"

# CLI override
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --alphas) ALPHAS="$2"; shift 2 ;;
        --n-prompts) N_PROMPTS="$2"; shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_ROOT"

run_subset() {
    local subset_name="$1" method="$2" arms="$3"
    local out="$OUT_ROOT/$subset_name"
    echo "==> [A_per_method] subset=$subset_name method=$method arms=[$arms]"
    "$PY" experiments/A_ablation/run.py \
        --out "$out" \
        --device "$DEVICE" --dtype "$DTYPE" \
        --models $MODELS \
        --arms $arms \
        --alphas $ALPHAS \
        --seeds $SEEDS \
        --n-prompts "$N_PROMPTS" \
        --method "$method"
}

# A5 + control on CAA
run_subset caa_arm     caa           "control A5"
# A3 + A6 on LOPI-default
run_subset lopi_arm    lopi_default  "control A3 A6"

# A4 on SCAR — only if SCAR method is registered in run.py.
if grep -q '"scar"' experiments/A_ablation/run.py; then
    run_subset scar_arm scar "control A4"
else
    echo "==> [A_per_method] skipping scar_arm: --method scar not registered in run.py"
fi

# A1 + A2 on attn_native bank — only if registered.
if grep -q '"attn_native"' experiments/A_ablation/run.py; then
    run_subset bank_arm attn_native "control A1 A2"
else
    echo "==> [A_per_method] skipping bank_arm: --method attn_native not registered in run.py"
fi

echo "==> [A_per_method] done; subsets at $OUT_ROOT/{caa_arm,lopi_arm,...}/cells.jsonl"
