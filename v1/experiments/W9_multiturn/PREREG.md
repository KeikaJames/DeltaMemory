# W.9 Pre-Registration: Multi-Turn Override

**Status**: locked.
**Authored**: 2026-05-04.
**Depends on**: W.4 verdict supplies the winning injection method.
**Hardware target**: 128 GB unified memory.

---

## 1. Question

In a multi-turn dialogue, the bank-injected fact must (a) remain accessible
across turns and (b) not corrupt turns where it is not relevant. Does
`M_winner` survive turn boundaries and selective-relevance gating?

## 2. Hypotheses

For each dialogue with `T` turns:

**H9a (cross-turn persistence)**:
  When the bank-injected fact is queried at turn `T-1` (last turn),
  `nll_target(M_winner) - nll_target(M_none) < 0` with paired Wilcoxon
  p < 0.01. The fact is still retrievable T-2 turns after it was written.

**H9b (relevance gating)**:
  At off-topic turns (turns where no bank fact is relevant per gold
  annotation), `|drift(M_winner) - drift(M_none)| < 0.2 nats` per model.
  The injection should not corrupt unrelated turns.

**H9c (red-line)**:
  At `alpha=0`, drift is bit-equal at every turn. Inheritance.

## 3. Grid

`5 models x 7 alphas x 3 seeds x 20 dialogues = 2,100 dialogue-cells`.

Each dialogue produces multiple per-turn rows; expanded count depends on
average dialogue length (8-12 turns observed in
`multiturn_dialogues_20.jsonl`). Expected total ~21,000 turn-rows.

- Models, methods (`none` + W.4 winner), alphas, seeds: same as W.6.
- Dialogues: 20 from `multiturn_dialogues_20.jsonl`. Bank fact for each
  dialogue is the assistant turn 1 statement (e.g. "Paris is the capital
  of France, known for the Eiffel Tower.").
- Relevance annotation: each turn carries a gold flag `relevant_to_bank`
  computed by exact-match of subject string. Annotation is deterministic
  and committed alongside the dataset; if subjects do not appear in any
  later turn, the dialogue is `H9b-unscorable` and dropped from H9b only.

## 4. Probes

- `nll_target` per turn (assistant continuation, last 8 tokens mean).
- `drift_per_turn` vs `M_none`.
- `relevant_to_bank` (bool).

## 5. Statistics

H9a paired Wilcoxon across 20 dialogues at turn `T-1`, Holm across 5 x 7
= 35 cells.
H9b per-model max-drift across off-topic turns; threshold strict per
model (no aggregation).

## 6. Red-lines and aborts

1. `alpha=0` `|drift| >= 1e-4` at any turn -> flagged.
2. `H9b-unscorable` rate > 30% across dialogues -> annotation pipeline is
   broken; abort and re-author dataset before continuing.
3. Tokenizer chat-template mismatch between baseline and injection runs
   -> run aborts.

## 7. Deliverables

- `cells.jsonl`, `cells_smoke.jsonl`, `summary.json`,
  `per_turn_curves.json`, `REPORT.md`, `env.json`. Smoke = gpt2-medium x
  3 dialogues x 1 seed x 2 alphas.

## 8. Out of scope

- Long-context (W.7).
- Multi-fact interference (W.8).
- RLHF or instruct-tuned dialogue heuristics (causal LM behaviour only).
- Tool-call / function-call dialogue formats.
