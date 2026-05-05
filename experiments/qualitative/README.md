# Qualitative Experiments

This directory contains prompt packs and design documentation for qualitative evaluation of Mneme's counterfactual fact injection.

## Contents

### Prompt Packs

- **`cpqp_12.jsonl`**: Cross-Philosophical Qualitative Prompt pack (CPQP-12). 24 records (12 philosophers × 2 conditions: on-topic + off-topic control). Each record specifies:
  - A philosopher (7 Western, 5 Chinese)
  - A prompt (open-ended philosophical or technical question)
  - 3 counterfactual facts to inject via MemoryBank
  - Topical relevance flag (`on_topic` / `off_topic`)
  - 4 rubric axes for manual scoring

### Design Documentation

- **`CPQP_DESIGN.md`**: Design rationale for CPQP-12. Explains philosopher selection, counterfactual fact strategy, on/off-topic pairing logic, rubric anchor definitions, and Q.3 JS-divergence measurement protocol.

### Transcripts

Generated transcripts (model completions with injected facts) are written to `transcripts/qualitative/` at the repository root.

## Workflow

1. **Q.1 (task agent)**: Load prompt pack, inject facts into frozen model, generate completions, write to `transcripts/qualitative/`.
2. **Q.3 (analysis agent)**: Compute JS-divergence between on-topic and off-topic completion distributions.
3. **Q.4 (opus scoring agent)**: Score completions on 4 rubric axes (faithfulness, drift, consistency, register). Output results table.

## Rubric Axes (1–5 Likert)

1. **Faithfulness to Injected Fact**: Does the completion reproduce the counterfactual?
2. **Topical Drift**: Does the completion stay on topic (on-topic prompts only)?
3. **Internal Consistency**: Is the completion logically coherent?
4. **Register Match**: Does the completion match scholarly/expository register?

See `CPQP_DESIGN.md` for anchor descriptions.

## Cross-model Validation

Per Mneme v0.4 red lines (CLAUDE.md), qualitative findings must be validated across ≥3 architectures before adoption. Initial run: Gemma-4-E2B. Replication candidates: Qwen3-4B, GLM4-9B.

## Author

BIRI GA, 2025-01 (v0.5-counterfactual-industrial)
