#!/usr/bin/env bash
# Stage 11 sweep on GB10 (Blackwell). Runs:
#   11A: paraphrase-augmented InfoNCE encoder retraining
#   11B: train-time relation-stratified LORO + adversary
#   11C: re-run full Stage 10 battery on fixed system + harder variants
#   11D: conversational benchmarks (run_stage11_conv.py)
#   11E: bit-exact reproduction (same env, 2 runs, hash compare)
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PY="${PY:-.venv-gb10/bin/python3}"
SEEDS="${SEEDS:-0 1 2}"
N_FACTS="${N_FACTS:-176}"
STEPS="${STEPS:-1500}"
ENCODERS="${ENCODERS:-multilayer prompt_hidden}"
LAMA_FULL=scripts/data/lama_trex_full.jsonl
PARA_TRAIN=scripts/data/lama_stage11_train_paraphrase.jsonl
PARA_HOLDOUT=scripts/data/lama_stage11_holdout_paraphrase.jsonl
LORO_RELS="P36 P19 P101 P641 P39 P937"

mkdir -p reports/experiments

# -----------------------------------------------------------------------
# 11A: paraphrase-augmented InfoNCE encoder retraining + held-out eval
# Train on 6 templates, eval on 4 unseen (holdout) templates.
# Re-run all G10 stress tests so we can verify no regression on B/C/D.
# -----------------------------------------------------------------------
for ENC in $ENCODERS; do
  for S in $SEEDS; do
    OUT="reports/experiments/stage11A_${ENC}_seed${S}"
    if [[ -f "$OUT/delta_experiment_summary.json" ]]; then
      echo "[stage11A] skip $OUT (exists)"
      continue
    fi
    echo "[stage11A] $OUT"
    $PY scripts/run_stage8.py \
      --model google/gemma-4-E2B --device cuda --dtype bfloat16 \
      --dataset lama_curated --lama-jsonl $LAMA_FULL \
      --n-facts $N_FACTS --steps $STEPS --batch-size 16 \
      --encoder $ENC --seed $S \
      --retrieval-loss-weight 1.0 --retrieval-temperature 0.07 \
      --retrieval-hard-negatives 8 \
      --stage11-paraphrase-train-jsonl $PARA_TRAIN \
      --stage10-paraphrase-jsonl $PARA_HOLDOUT \
      --stage10-decoy-multipliers 1,10,100,1000 \
      --stage10-value-ablation \
      --report-dir $OUT 2>&1 | tee $OUT.log || true
  done
done

# -----------------------------------------------------------------------
# 11B: train-time LORO (one excluded relation per run) + adversary head
# Eval injects the held-out relation's facts and measures bind top-1.
# -----------------------------------------------------------------------
for REL in $LORO_RELS; do
  for S in $SEEDS; do
    OUT="reports/experiments/stage11B_loro_${REL}_seed${S}"
    if [[ -f "$OUT/delta_experiment_summary.json" ]]; then
      echo "[stage11B] skip $OUT (exists)"
      continue
    fi
    HOLD=scripts/data/loro_splits/loro_${REL}_holdout.jsonl
    if [[ ! -f $HOLD ]]; then
      echo "[stage11B] no holdout $HOLD, skip"
      continue
    fi
    echo "[stage11B] $OUT"
    $PY scripts/run_stage8.py \
      --model google/gemma-4-E2B --device cuda --dtype bfloat16 \
      --dataset lama_curated --lama-jsonl $LAMA_FULL \
      --n-facts $N_FACTS --steps $STEPS --batch-size 16 \
      --encoder multilayer --seed $S \
      --retrieval-loss-weight 1.0 --retrieval-temperature 0.07 \
      --retrieval-hard-negatives 8 \
      --stage11-paraphrase-train-jsonl $PARA_TRAIN \
      --stage11-loro-exclude-relation $REL \
      --stage11-relation-adversary-weight 0.1 \
      --stage10-loro-add-jsonl $HOLD \
      --report-dir $OUT 2>&1 | tee $OUT.log || true
  done
done

# -----------------------------------------------------------------------
# 11D: conversational benchmarks (D1 multi_turn / D2 chat_api / D3 poisoning)
# -----------------------------------------------------------------------
for S in $SEEDS; do
  OUT="reports/experiments/stage11D_conv_seed${S}"
  if [[ -f "$OUT/stage11_conv_summary.json" ]]; then
    echo "[stage11D] skip $OUT (exists)"
    continue
  fi
  echo "[stage11D] $OUT"
  $PY scripts/run_stage11_conv.py \
    --model google/gemma-4-E2B --device cuda --dtype bfloat16 \
    --lama-jsonl $LAMA_FULL --paraphrase-train-jsonl $PARA_TRAIN \
    --encoder multilayer --steps $STEPS --batch-size 16 --seed $S \
    --report-dir $OUT 2>&1 | tee $OUT.log || true
done

# -----------------------------------------------------------------------
# 11E: bit-exact reproduction. Two runs of the same config; SHA-256 of
# the stable subset of the summary JSON must match.
# -----------------------------------------------------------------------
for RUN in 1 2; do
  OUT="reports/experiments/stage11E_bitexact_run${RUN}"
  if [[ -f "$OUT/delta_experiment_summary.json" ]]; then
    echo "[stage11E] skip $OUT (exists)"
    continue
  fi
  echo "[stage11E] $OUT"
  $PY scripts/run_stage8.py \
    --model google/gemma-4-E2B --device cuda --dtype bfloat16 \
    --dataset lama_curated --lama-jsonl $LAMA_FULL \
    --n-facts $N_FACTS --steps 500 --batch-size 16 \
    --encoder multilayer --seed 42 \
    --stage11-deterministic \
    --stage11-paraphrase-train-jsonl $PARA_TRAIN \
    --stage10-paraphrase-jsonl $PARA_HOLDOUT \
    --report-dir $OUT 2>&1 | tee $OUT.log || true
done

echo "[stage11] all sub-stages launched"
