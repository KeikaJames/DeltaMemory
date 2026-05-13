# W.8 Pre-Registration: Multi-Fact Interference

**Status**: locked.
**Authored**: 2026-05-04.
**Depends on**: W.4 verdict supplies the winning injection method.
**Hardware target**: 128 GB unified memory.

---

## 1. Question

When the bank holds K facts simultaneously, does retrieval of fact `i`
remain selective, or does the bank collapse into an averaged blur?
Capacity without selectivity is not memory.

## 2. Hypotheses

Let `K ∈ {1, 8, 32, 128}` be bank capacity. For each `(K, prompt_i)` where
`prompt_i` queries the i-th fact:

**H8a (selectivity)**:
  `nll_target_i(M; K) - nll_target_i(M; K=1) < 0.5 * log(K)` per model.
  i.e. selectivity decays slower than uniform mixture.

**H8b (cross-talk floor)**:
  `mean_{j != i} nll_target_j(M; K, prompt_i) > nll_target_i(M; K, prompt_i)`
  with paired Wilcoxon p < 0.01. Wrong-fact NLL must exceed correct-fact
  NLL.

**H8c (red-line)**:
  At `alpha=0`, drift is bit-equal regardless of `K`. Inheritance.

## 3. Grid

`5 models x 4 capacities x 7 alphas x 3 seeds x 32 prompts = 13,440 cells`.

- Models, methods (`none` + W.4 winner), alphas, seeds: same as W.6.
- Prompts and bank facts: drawn from `multifact_pack_8.jsonl`,
  `multifact_pack_32.jsonl`, `multifact_pack_128.jsonl`. The K=1 condition
  uses the i-th fact alone.
- Cross-talk evaluation queries every prompt against every fact in the
  bank (recorded as `target_fact_id`).

## 4. Probes

- `nll_target_i_correct`, `nll_target_j_other_mean`.
- `bank_attn_entropy`: Shannon entropy of attention weights over the K
  bank slots, mean across query positions.
- `top_attn_match_frac`: fraction of query tokens where the highest-
  attended bank slot matches the queried fact id.

## 5. Statistics

H8a per-model OLS of `nll_target_i` on `log(K)`. H8b paired Wilcoxon
across (i != j) pairings, Holm across 5 x 4 = 20 cells.

## 6. Red-lines and aborts

1. `alpha=0` `|drift| >= 1e-4` -> flagged.
2. K=128 OOM on a model -> drop model.
3. `bank_attn_entropy` saturates at `log(K)` for >= 95% of cells -> bank
   has collapsed; abort the run, file as W.8 fundamental limitation.

## 7. Deliverables

- `cells.jsonl`, `cells_smoke.jsonl`, `summary.json`, `selectivity.json`,
  `REPORT.md`, `env.json`. Smoke = gpt2-medium x K=8 x 1 seed x 5 prompts
  x 2 alphas.

## 8. Out of scope

- Long-context (W.7).
- Multi-turn (W.9).
- Bank update / forgetting dynamics (R-7-style; tracked separately).
