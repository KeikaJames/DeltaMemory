# W.7 Pre-Registration: Long-Context Degradation

**Status**: locked.
**Authored**: 2026-05-04.
**Depends on**: W.4 verdict supplies the winning injection method.
**Hardware target**: 128 GB unified memory (long-context inflates KV cache).

---

## 1. Question

Does external memory injection preserve drift discipline as prompt length
grows? A method that ships clean at 64 tokens but disintegrates at 1024 is
not a memory; it is a short-prompt artifact.

## 2. Hypotheses

For each context length `L ∈ {64, 128, 256, 512, 1024, 2048}`:

**H7a (drift bound)**:
  `median_p drift(M_winner; L)` grows sublinearly in `log(L)`. Operationally:
  `drift(L=2048) - drift(L=64) < 1.5 * drift(L=64)` per model.

**H7b (rank preservation)**:
  Top-1 next-token rank under `M_winner` matches the top-1 rank under
  `M_none` for >= 70% of held-out-suffix tokens, at every `L`.

**H7c (red-line)**:
  At `alpha=0`, drift is bit-equal (`< 1e-4`) at every `L`. Inheritance
  from W.4 / W.6.

## 3. Grid

`5 models x 6 lengths x 7 alphas x 3 seeds x 30 prompts = 18,900 cells`.

- Models, methods (`none` + W.4 winner), alphas, seeds: same as W.6.
- Prompts: 30 from `gold_30prompts.jsonl` extended via deterministic
  wikitext-2 prefix concatenation to each target length. Prefix sentence
  windows are pre-shuffled with seed=0 and cached at
  `experiments/W7_longctx/prefix_cache.jsonl`.

## 4. Token-rank probe

At each cell we record:
- `nll_target` (last 8 tokens of the gold suffix, mean).
- `top1_match_frac`: fraction of suffix tokens where `argmax logits(M_winner)`
  equals `argmax logits(M_none)`.
- `kv_cache_mb`: peak KV-cache memory for the cell (audit).

## 5. Statistics

H7a per-model OLS regression of `drift(L)` on `log(L)` with bootstrap CI.
H7b per-`(model, alpha)` paired Wilcoxon vs `M_none`. Holm-Bonferroni
across 5 x 6 = 30 comparisons per hypothesis, threshold 0.01.

## 6. Red-lines and aborts

1. `alpha=0` cell with `|drift(M) - drift(M_none)| >= 1e-4` -> flagged.
2. KV cache OOM on a model -> drop that model, record substitution.
3. Tokenization mismatch between baseline and injection runs -> run aborts.

## 7. Deliverables

- `cells.jsonl`, `cells_smoke.jsonl`, `summary.json`, `length_curve.json`,
  `REPORT.md`, `env.json`. Smoke = gpt2-medium x {64, 256, 1024} x 1 seed
  x 5 prompts x 2 alphas.

## 8. Out of scope

- Multi-fact interference (W.8).
- Multi-turn conversational drift (W.9).
- Encoder-decoder architectures (causal LMs only).
