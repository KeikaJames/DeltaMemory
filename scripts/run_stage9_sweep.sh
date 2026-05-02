#!/usr/bin/env bash
# Stage 9 sweep driver. Run on GB10 (CUDA, bf16).
#
# Phase 9A — encoder sweep at N=4096, seed=0:
#   mean_pool (baseline / sanity), attn_pool, multilayer, prompt_hidden, residual_mlp
# Phase 9A-bis — best two encoders × seed {1,2} × N=4096 for reproducibility.
# Phase 9B — full LAMA-TREx (~183 facts) × prompt_hidden × seed {0,1,2}.
# Phase 9C — baselines on LAMA-TREx full: vector_rag, ike, sft_lora.
#
# Budget: ~12 main runs. Wall ~3-4h on GB10.

set -euo pipefail
cd "$(dirname "$0")/.."

source .venv-gb10/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

ROOT=reports/experiments
mkdir -p "$ROOT"

N=4096
STEPS=2000
KEY_DIM=256
TEMP=0.07

echo "=== Phase 9A: encoder sweep, N=$N, seed=0 ==="
for ENC in mean_pool attn_pool multilayer prompt_hidden residual_mlp; do
  OUT="$ROOT/stage9A_${ENC}_n${N}_seed0"
  if [[ -f "$OUT/delta_experiment_summary.json" ]]; then
    echo "SKIP $OUT (exists)"
    continue
  fi
  echo ">>> running $ENC"
  python3 scripts/run_stage8.py \
    --device cuda --dtype bfloat16 \
    --n-facts $N --steps $STEPS --seed 0 \
    --key-dim $KEY_DIM --retrieval-temperature $TEMP \
    --encoder "$ENC" \
    --report-dir "$OUT" 2>&1 | tee "$OUT.log" >/dev/null || echo "  WARN: $ENC failed"
done

echo "=== Phase 9A-bis: best two encoders × seed {1,2} ==="
for ENC in prompt_hidden multilayer; do
  for SEED in 1 2; do
    OUT="$ROOT/stage9A_${ENC}_n${N}_seed${SEED}"
    if [[ -f "$OUT/delta_experiment_summary.json" ]]; then
      echo "SKIP $OUT (exists)"
      continue
    fi
    echo ">>> $ENC seed=$SEED"
    python3 scripts/run_stage8.py \
      --device cuda --dtype bfloat16 \
      --n-facts $N --steps $STEPS --seed $SEED \
      --key-dim $KEY_DIM --retrieval-temperature $TEMP \
      --encoder "$ENC" \
      --report-dir "$OUT" 2>&1 | tee "$OUT.log" >/dev/null || echo "  WARN: $ENC seed=$SEED failed"
  done
done

echo "=== Phase 9B: full LAMA-TREx with best encoder ==="
LAMA_FACTS=183
LAMA_STEPS=1500
for SEED in 0 1 2; do
  OUT="$ROOT/stage9B_trex_prompt_hidden_seed${SEED}"
  if [[ -f "$OUT/delta_experiment_summary.json" ]]; then
    echo "SKIP $OUT (exists)"
    continue
  fi
  python3 scripts/run_stage8.py \
    --device cuda --dtype bfloat16 \
    --dataset lama_curated --lama-jsonl scripts/data/lama_trex_full.jsonl \
    --n-facts $LAMA_FACTS --steps $LAMA_STEPS --seed $SEED \
    --encoder prompt_hidden \
    --report-dir "$OUT" 2>&1 | tee "$OUT.log" >/dev/null || echo "  WARN seed=$SEED"
done

echo "=== Phase 9C: opponent baselines on LAMA-TREx full ==="
for METHOD in vector_rag ike sft_lora; do
  OUT="$ROOT/stage9C_${METHOD}_seed0"
  if [[ -f "$OUT/delta_experiment_summary.json" ]]; then
    echo "SKIP $OUT (exists)"
    continue
  fi
  python3 scripts/run_stage9_baselines.py \
    --device cuda --dtype bfloat16 \
    --method "$METHOD" --seed 0 \
    --lama-jsonl scripts/data/lama_trex_full.jsonl \
    --report-dir "$OUT" 2>&1 | tee "$OUT.log" >/dev/null || echo "  WARN $METHOD"
done

echo "=== Stage 9 sweep complete ==="
ls -la $ROOT/stage9*/delta_experiment_summary.json 2>/dev/null
