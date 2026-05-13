#!/usr/bin/env bash
# Stage 10 resume — finish 10F (LORO) + 10C (baselines). 10E skipped
# due to N=16384 OOM at decoy x100/x1000; scale claim covered by
# Stage 9A (N=4096) + Stage 10ABD (N=183, decoy x1000 = 1.000).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-gb10/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
ROOT=reports/experiments

run() {
  local out=$1; shift
  if [[ -f "$out/delta_experiment_summary.json" ]]; then echo "SKIP $out"; return 0; fi
  echo ">>> $out"
  mkdir -p "$out"
  "$@" --report-dir "$out" 2>&1 | tee "$out.log" >/dev/null || echo "  WARN: $out"
}

LAMA_STEPS=1500

echo "=== 10F : LORO ==="
for REL in P36 P19 P101 P641 P39 P937; do
  OUT="$ROOT/stage10F_loro_${REL}_seed0"
  run "$OUT" python3 scripts/run_stage8.py \
    --device cuda --dtype bfloat16 \
    --dataset lama_curated --lama-jsonl "scripts/data/loro_splits/loro_${REL}_train.jsonl" \
    --steps $LAMA_STEPS --seed 0 \
    --encoder prompt_hidden \
    --stage10-loro-add-jsonl "scripts/data/loro_splits/loro_${REL}_holdout.jsonl"
done

echo "=== 10C : SFT-LoRA × seed × rank ==="
for SEED in 0 1 2; do
  for RANK in 4 16 64; do
    OUT="$ROOT/stage10C_sft_lora_r${RANK}_seed${SEED}"
    run "$OUT" python3 scripts/run_stage9_baselines.py \
      --device cuda --dtype bfloat16 \
      --method sft_lora --seed $SEED --steps $LAMA_STEPS --rank $RANK \
      --lama-jsonl scripts/data/lama_trex_full.jsonl
  done
done

echo "=== 10C-bis : RAG / IKE × 3 seeds ==="
for METHOD in vector_rag ike; do
  for SEED in 0 1 2; do
    OUT="$ROOT/stage10C_${METHOD}_seed${SEED}"
    run "$OUT" python3 scripts/run_stage9_baselines.py \
      --device cuda --dtype bfloat16 \
      --method "$METHOD" --seed $SEED \
      --lama-jsonl scripts/data/lama_trex_full.jsonl
  done
done

echo "=== Stage 10 resume complete ==="
ls -la "$ROOT"/stage10*/delta_experiment_summary.json 2>/dev/null | wc -l
