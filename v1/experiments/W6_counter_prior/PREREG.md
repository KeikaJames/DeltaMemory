# W.6 Pre-Registration: Counter-Prior Pareto

**Status**: locked.
**Authored**: 2026-05-04 (post W.3 decision gate, post W.4 PREREG, post W.5 PREREG).
**Depends on**: W.4 verdict (CAA paired comparison) must land before this run starts; the
winning method from W.4 (`caa` if `H1: caa < none` is supported under Holm; otherwise
`v_scale` + `lopi_default` ablation per W.3) supplies the injection arm here.
**Hardware target**: 128 GB unified memory or single CUDA GPU with ≥48 GB. Does not run
on the 64 GB development machine.

---

## 1. Question

W.4 establishes whether memory injection moves NLL on neutral-anchor prompts in the
**right direction**. W.6 asks the harder question:

> Can a frozen LLM be steered to override its own factual prior toward a counterfactual
> answer, **without** unacceptable collateral damage to unrelated next-token
> distributions?

This is the operational definition of *useful* external memory. A method that lowers
target-NLL only by widening base-token KL is doing rote distortion, not memory.

## 2. Hypotheses

Let `M(method, alpha)` denote the system under test. Define for each prompt
`p = (subject, relation, target_true, target_new)`:

- `nll_new(p; M)`  — negative log-likelihood of `target_new` continuation under `M`.
- `nll_true(p; M)` — negative log-likelihood of `target_true` continuation under `M`.
- `kl_unrel(p; M)` — token-mean KL divergence between `M` and the unmodified base on a
   held-out **unrelated** continuation drawn from the same model family's pretraining
   distribution proxy (wikitext-2 sentence sample anchored to a non-overlapping
   subject).

**H6a (steering)**:
  `median_p [nll_new(p; M_winner) - nll_new(p; M_none)] < 0`
  with paired Wilcoxon p < 0.01 after Holm correction.

**H6b (override)**:
  `median_p [nll_new(p; M_winner) - nll_true(p; M_winner)] < 0`
  i.e. after injection, the counterfactual answer beats the model's own prior on the
  same prompt. Wilcoxon p < 0.01.

**H6c (Pareto frontier non-trivial)**:
  At the alpha that minimizes `nll_new`, `kl_unrel < 0.5 nats`. Strict per-model.

**H6d (red-line)**:
  At alpha=0, `max |nll_new(M) - nll_new(M_none)| < 1e-4`.
  Any violation aborts the cell and flags the run.

## 3. Grid

Total cells: **5 models × 2 methods × 7 alphas × 3 seeds × 60 prompts = 12,600**.

- **Models**: same as W.4 — `gpt2-medium`, `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`,
  `google/gemma-3-270m`, `google/gemma-3-1b-it`. Substitution policy inherits W.2/W.4.
- **Methods**: `none` (no injection, baseline) + `M_winner` (decided by W.4 verdict).
- **Alphas**: `{0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0}`.
- **Seeds**: `{0, 1, 2}`.
- **Prompts**: 60-row counterfact pack expanded from `counterfact_60.jsonl` (sha
  `c3e1ac77…`); each row supplies subject, prompt template, target_true, target_new.

## 4. Injection content

For each prompt the injection bank receives **one** synthetic Fact line:
`Fact: {subject} {relation_phrase} {target_new}.` — relation_phrase is rendered from
the LAMA T-REx template (`experiments/datasets/lama_trex_500.jsonl`) using
`predicate_id == relation`. If no template exists for that relation, the prompt is
dropped from the run and recorded as `relation_template_missing=true`. Drop rate is
expected ≤ 5%; if it exceeds 10% the run aborts and the gap is filled by hand-authored
templates, recorded as a deviation.

## 5. Unrelated-KL probe

For each `(model, seed)` we precompute 60 unrelated continuations: from
wikitext-2-raw-v1 we sample 60 distinct 16-token windows whose first noun does **not**
match the prompt's subject (case-insensitive substring check; if all 60 collide we
resample with seed+1000). Token-mean KL is computed only on the final 8 tokens of each
window so that prefix prompt artifacts do not dominate. KL is symmetric Jensen-Shannon
not raw KL, to bound the metric in `[0, log 2]`.

## 6. Statistics

- **H6a, H6b**: paired Wilcoxon, two-sided, `zero_method="wilcox"`, paired by
  `(seed, prompt_id)`. Holm-Bonferroni across `(model, alpha)` cells = 5 × 7 = 35
  comparisons per hypothesis, threshold 0.01.
- **H6c**: per-model report Pareto frontier (alpha, median nll_new, median kl_unrel);
  flag the alpha that minimizes nll_new and report kl_unrel at that alpha. Strict
  threshold; no aggregation across models.
- **Effect size**: median paired diff with 95% bootstrap CI (B=1000, seed=0).

## 7. Red-lines and abort conditions

1. Any alpha=0 cell with `|nll_new(M) - nll_new(M_none)| ≥ 1e-4` → cell flagged
   `redline_violation=true`; per-model fail of H6d.
2. relation-template miss > 10% → run aborts before stats.
3. CUDA / MPS OOM on any model → that model is dropped, recorded as substitution.
4. If W.4 verdict is FAIL on H1 (i.e. `caa` does not beat `none`), W.6 substitutes
   `M_winner = lopi_default` per W.3 fallback list. This branch is recorded in
   `env.json.method_winner_source = "w4_h1_failed"`.

## 8. Deliverables

- `cells.jsonl`     — full grid (12,600 rows; one row per cell).
- `cells_smoke.jsonl` — pre-flight on gpt2-medium, 1 seed, 5 prompts, 2 alphas.
- `pareto.json`     — H6c frontier per model.
- `summary.json`    — H6a/H6b Wilcoxon + Holm verdicts.
- `REPORT.md`       — narrative, no emoji, no colloquial Chinese.
- `env.json`        — env hash, commit, method_winner_source.

## 9. Out of scope

- Multi-fact interference (W.8).
- Long-context degradation (W.7).
- Multi-turn override (W.9).
- Anything that requires re-fitting the bank between prompts.

---

End of pre-registration. After this point, no parameter in §3, §6, §7 may change without
recording the change in REPORT §Deviations and bumping `prereg_version` in `env.json`.
