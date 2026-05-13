# Exp31 — Learned K-adapter (InfoNCE)

## Status
**Phase Φ1 — data splits built.** Splits live in `data/splits/`.

## Hypothesis under test
**H_A**: the Exp23–27 ANB falsification (cosine-routed native attention bank
collapses for N ≫ 100 across three architectures) is caused by insufficient
discriminability of the raw pre-RoPE K space. A small learned projection
W_K trained by InfoNCE (anchor: canonical write prompt's K; positive: K of
2 paraphrases of the same fact; negatives: in-batch + hard negatives) should
recover Gate B (retrieval_accuracy) ≥ 4× chance at N=200 and survive at
N∈{400, 800}.

## Splits (deterministic, seed=0)

| Split   | Facts | Paraphrases / fact | Use                                         |
|---------|------:|-------------------:|---------------------------------------------|
| train   |   700 |                  2 | InfoNCE training of W_K                     |
| val     |   150 |                  2 | early-stop on Gate B; α grid selection      |
| test    |   150 |                  2 | final 5-gate evaluation                     |
| distractors | 10000 |              n/a | bank padding for N > 150                    |

Source: `experiments/datasets/counterfact_1k.jsonl` (split by P-relation
round-robin) + `experiments/X1_bank_scaling/distractors.jsonl`. SHA-256s
recorded in `data/splits/manifest.json`.

## Evaluation gates (paired bootstrap B=2000)
- **A**: margin(topk1, learned) − margin(minus_correct) > 0
- **B**: retrieval_accuracy(learned) ≥ 4× chance
- **C**: margin(topk1, learned) − margin(meanV) > 0
- **D**: margin(topk1, learned) − margin(shuffled_factids) > 0
- **E** *(new)*: margin(topk1, learned) − margin(shuffled_adapter) > 0
  — fact-content cannot leak through W_K itself; W_K must learn routing,
  not memorize answers.

## Architecture variants
- **Main**: low-rank `Linear(d_h, 64) → Linear(64, d_h)`, identity-init
  outer composition.
- **Ablation**: full-rank `Linear(d_h, d_h)`, identity-init.

## Models
- Qwen3-4B-Instruct-2507 (primary)
- google/gemma-4-E2B (cross-arch)
- mistralai/Mistral-7B-Instruct-v0.3 (cross-arch)

## Layout
```
exp31_learned_k_adapter/
  build_splits.py          # this Φ1 — produces data/splits/*
  data/
    splits/                # tracked (700/150/150 split + manifest)
    paraphrase_cache/      # gitignored (regenerable)
  train_k_adapter.py       # Φ2+ — InfoNCE training
  eval_k_adapter.py        # Φ2+ — five-gate eval
  run_mps_exp31_qwen_smoke/   # Φ2 N=200 smoke artifacts
  run_mps_exp31_qwen_full/    # Φ3 full N×3-seed grid
  run_mps_exp31_gemma_full/   # Φ4
  run_mps_exp31_mistral_full/ # Φ4
  EXP31_VERDICT.md         # Φ8 final verdict
```
