#!/usr/bin/env bash
# Stage 10 — Adversarial Validation sweep. Run on GB10 (CUDA, bf16).
#
# Sub-stages launched here:
#   10A  — Paraphrase robustness  (--stage10-paraphrase-jsonl)
#   10B  — Distractor stress curve (--stage10-decoy-multipliers)
#   10D  — Null/value ablation     (--stage10-value-ablation)
#   10E  — Scale to N=16384
#   10F  — Leave-one-relation-out (LORO)
#   10C  — Equal-budget SFT-LoRA × seed × rank sweep
#
# 10A/10B/10D are bundled into a single re-run of Stage 9B per seed
# (the bank state is already correct for those tests).

set -euo pipefail
cd "$(dirname "$0")/.."

source .venv-gb10/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

ROOT=reports/experiments
mkdir -p "$ROOT"

LAMA_FACTS=183
LAMA_STEPS=1500
PARA=scripts/data/lama_trex_paraphrase.jsonl
DECOYS="1,10,100,1000"

run() {
  local out=$1; shift
  if [[ -f "$out/delta_experiment_summary.json" ]]; then
    echo "SKIP $out"
    return 0
  fi
  echo ">>> $out"
  mkdir -p "$out"
  "$@" --report-dir "$out" 2>&1 | tee "$out.log" >/dev/null || echo "  WARN: $out"
}

echo "=== 10A+10B+10D : Stage 9B re-run with stress test flags ==="
for ENC in prompt_hidden multilayer; do
  for SEED in 0 1 2; do
    OUT="$ROOT/stage10ABD_${ENC}_seed${SEED}"
    run "$OUT" python3 scripts/run_stage8.py \
      --device cuda --dtype bfloat16 \
      --dataset lama_curated --lama-jsonl scripts/data/lama_trex_full.jsonl \
      --n-facts $LAMA_FACTS --steps $LAMA_STEPS --seed $SEED \
      --encoder "$ENC" \
      --stage10-paraphrase-jsonl "$PARA" \
      --stage10-decoy-multipliers "$DECOYS" \
      --stage10-value-ablation
  done
done

echo "=== 10E : Scale to N=16384 ==="
for ENC in prompt_hidden multilayer; do
  for SEED in 0 1 2; do
    OUT="$ROOT/stage10E_${ENC}_n16384_seed${SEED}"
    run "$OUT" python3 scripts/run_stage8.py \
      --device cuda --dtype bfloat16 \
      --n-facts 16384 --steps 4000 --seed $SEED \
      --encoder "$ENC" \
      --stage10-decoy-multipliers "1,10,100" \
      --stage10-value-ablation
  done
done

echo "=== 10F : Leave-one-relation-out ==="
for REL in P36 P19 P101 P641 P39 P937; do
  OUT="$ROOT/stage10F_loro_${REL}_seed0"
  TRAIN="scripts/data/loro_splits/loro_${REL}_train.jsonl"
  HOLD="scripts/data/loro_splits/loro_${REL}_holdout.jsonl"
  run "$OUT" python3 scripts/run_stage8.py \
    --device cuda --dtype bfloat16 \
    --dataset lama_curated --lama-jsonl "$TRAIN" \
    --steps $LAMA_STEPS --seed 0 \
    --encoder prompt_hidden \
    --stage10-loro-add-jsonl "$HOLD"
done

echo "=== 10C : Equal-budget SFT-LoRA × seed × rank ==="
for SEED in 0 1 2; do
  for RANK in 4 16 64; do
    OUT="$ROOT/stage10C_sft_lora_r${RANK}_seed${SEED}"
    run "$OUT" python3 scripts/run_stage9_baselines.py \
      --device cuda --dtype bfloat16 \
      --method sft_lora --seed $SEED --steps $LAMA_STEPS --rank $RANK \
      --lama-jsonl scripts/data/lama_trex_full.jsonl
  done
done

echo "=== 10C-bis : vector_rag and IKE × 3 seeds (input-embed encoder; F4 bound noted in report) ==="
for METHOD in vector_rag ike; do
  for SEED in 0 1 2; do
    OUT="$ROOT/stage10C_${METHOD}_seed${SEED}"
    run "$OUT" python3 scripts/run_stage9_baselines.py \
      --device cuda --dtype bfloat16 \
      --method "$METHOD" --seed $SEED \
      --lama-jsonl scripts/data/lama_trex_full.jsonl
  done
done

echo "=== Stage 10 sweep complete ==="
ls -la "$ROOT"/stage10*/delta_experiment_summary.json 2>/dev/null | wc -l
