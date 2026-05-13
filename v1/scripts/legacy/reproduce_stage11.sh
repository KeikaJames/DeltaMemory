#!/usr/bin/env bash
# Stage 11 reproduction harness. Runs the bit-exact deterministic config
# twice and compares SHA-256 of stable summary subsets.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv-gb10/bin/python3}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for RUN in 1 2; do
  OUT="reports/experiments/stage11E_repro_run${RUN}"
  if [[ -f "$OUT/delta_experiment_summary.json" ]]; then
    echo "[repro] skip $OUT (exists)"
    continue
  fi
  $PY scripts/run_stage8.py \
    --model google/gemma-4-E2B --device cuda --dtype bfloat16 \
    --dataset lama_curated --lama-jsonl scripts/data/lama_trex_full.jsonl \
    --n-facts 176 --steps 500 --batch-size 16 \
    --encoder multilayer --seed 42 \
    --stage11-deterministic \
    --stage11-paraphrase-train-jsonl scripts/data/lama_stage11_train_paraphrase.jsonl \
    --report-dir $OUT
done

$PY - <<'PY'
import json, hashlib, sys
from pathlib import Path
def h(p):
    s = json.loads(p.read_text())
    m = s.get("metrics", {})
    keys = {
        "retr@1": m.get("address_retrieval_recall_at_1"),
        "swap":   m.get("swap_paired"),
        "bank_retr_top1": m.get("bank_inject_retrieved", {}).get("top1"),
        "no_mem_top1":    m.get("no_memory", {}).get("top1"),
    }
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()
h1 = h(Path("reports/experiments/stage11E_repro_run1/delta_experiment_summary.json"))
h2 = h(Path("reports/experiments/stage11E_repro_run2/delta_experiment_summary.json"))
print(f"run1: {h1}")
print(f"run2: {h2}")
print("MATCH" if h1 == h2 else "MISMATCH")
sys.exit(0 if h1 == h2 else 1)
PY
