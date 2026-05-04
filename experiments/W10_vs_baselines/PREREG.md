# W.10 Pre-Registration: vs Prompt-Insertion and RAG Baselines

**Status**: locked.
**Authored**: 2026-05-04.
**Depends on**: W.4 verdict supplies `M_winner` (the DeltaMemory arm under
test); W.6 results (counter-prior Pareto) supply `DM_best_alpha` (the
alpha at which `M_winner` minimises `nll_new` while `kl_unrel < 0.5`).
**Hardware target**: 64 GB unified memory (no MoE here; Qwen3-MoE-A3B is
out of scope for W.10).

W.10 is the answer to "why did you not just put the fact in the prompt?"
This is the most common reviewer objection for any external-memory paper.
The question must be answered with the same data, the same prompts, the
same seeds, and the same paired-test machinery as W.6, otherwise the
comparison is not honest.

---

## 1. Question

For the same counterfactual prompt, the same target_new, the same model,
and the same seed, does `DM_best_alpha` beat the three obvious cheaper
alternatives?

The three baselines are:
- **B0** `no_memory` — the unmodified frozen LLM. Lower bound; serves as
  the bit-equality witness at alpha=0 inheritance.
- **B1** `prompt_insertion` — the bank Fact line is prepended to the
  prompt as plain text. This is the strongest cheap baseline; if
  `M_winner` cannot beat B1 the method is not interesting.
- **B2** `rag_bm25` — the LAMA T-REx 500 corpus is treated as a retrieval
  index; for each test prompt, BM25 returns the top-3 facts; those are
  prepended. This baseline tests "is your method better than off-the-shelf
  retrieval?".

## 2. Hypotheses

For each `(model, prompt)` pair, compute `nll_new` under each of
{`DM_best_alpha`, B0, B1, B2}.

**H10a (beat B1, the dominant test)**:
  `median_p [nll_new(DM_best_alpha) - nll_new(B1)] < 0` with paired
  Wilcoxon p < 0.01 after Holm correction. PASS condition: at least 3 of
  the 5 dense models satisfy this. Failure across 3 or more models means
  prompt insertion is at least as effective as the proposed memory; the
  paper has no story.

**H10b (beat B2)**:
  Same comparison vs B2. PASS condition: 4 of 5 models. RAG BM25 is a
  weaker baseline than B1 because it can retrieve the wrong fact; we
  expect a clearer separation here.

**H10c (separation from B0)**:
  `median_p [nll_new(DM_best_alpha) - nll_new(B0)] < median_p [nll_new(B1) - nll_new(B0)]`.
  i.e. the lift over no-memory is at least as large as the lift produced
  by trivial prompt insertion. Paired Wilcoxon, Holm-Bonferroni across 5
  models, threshold 0.01.

**H10d (red-line)**:
  At `alpha=0` for `M_winner`, `|nll_new(DM_best_alpha) - nll_new(B0)| < 1e-4`.
  Inherited from W.4.

## 3. Grid

`5 models x 4 methods x 3 seeds x 60 prompts = 3,600 cells`. Methods are
{`DM_best_alpha`, B0, B1, B2}. No alpha sweep — `DM_best_alpha` is fixed
to the per-model alpha selected by W.6's H6c.

- Models: `gpt2-medium`, `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`,
  `google/gemma-3-270m`, `google/gemma-3-1b-it`.
- Prompts: 60 from `counterfact_60.jsonl` (sha `c3e1ac77`).
- Seeds: `{0, 1, 2}`.
- `M_winner` and `DM_best_alpha` are read at runtime from
  `experiments/W6_counter_prior/REPORT.md` (sections "method_winner" and
  "per_model_argmin_alpha"). If those sections do not exist (W.6 has not
  produced a verdict yet), the run aborts — W.10 is not allowed to ship
  before W.6.

## 4. Baseline implementation contract

### B1 prompt_insertion

For each prompt `p` with subject `S`, relation `R`, target_new `T_new`:
```
prefixed_prompt = f"Fact: {S} {relation_phrase(R)} {T_new}.\n{p['prompt'].format(S)}"
```
Same `relation_phrase` lookup as W.6. Tokenisation uses the model's own
tokenizer; no chat template.

### B2 rag_bm25

A single BM25 index is built once per run from `lama_trex_500.jsonl`,
treating each row's `template`-rendered fact line as a document. For
each prompt's `(subject, relation)`, the query is
`f"{subject} {relation_phrase(relation)}"` and the top-3 hits are
joined into a single prefix. The index is computed deterministically
(`rank_bm25` k1=1.5 b=0.75); index sha is recorded in `env.json`.

### Tokenisation parity

All four methods use identical tokenisation: tokenize once per prompt
including the prefix, then compute `nll_new` on the suffix tokens whose
indices are determined post-prefix. This guarantees the only thing
varying between methods is what is in the prefix and how the bank is
filled, not how the suffix is delimited.

## 5. Statistics

H10a, H10b, H10c: paired Wilcoxon two-sided, `zero_method="wilcox"`,
paired by `(seed, prompt_id)`. Holm-Bonferroni across 5 models per
hypothesis, threshold 0.01.

Effect size: median paired diff with 95 percent bootstrap CI, B=1000,
seed=0.

H10c uses a paired bootstrap test on the difference of paired-medians;
the null is "DM lift equals B1 lift".

## 6. Red-lines and aborts

1. `alpha=0` `|nll_new(DM_best_alpha) - nll_new(B0)| >= 1e-4` -> flagged.
2. BM25 index sha mismatch between two cells of the same model -> abort.
3. Tokenisation parity failure (different suffix token counts between
   methods on the same prompt) -> abort the cell, log to stderr.
4. Drop `relation_template_missing` rows (inherited from W.6) before
   stats.

## 7. Deliverables

- `cells.jsonl`            — 3,600 rows.
- `cells_smoke.jsonl`      — gpt2-medium x 4 methods x 1 seed x 5 prompts.
- `summary.json`           — H10a/b/c verdicts, per-model.
- `bm25_index.txt`         — sha + first 5 hits for audit.
- `REPORT.md`              — narrative.
- `env.json`               — env hash, prereg_version, M_winner read from
  W.6, DM_best_alpha per model.

## 8. Out of scope

- ROME / MEMIT / GRACE (W.14, weight-editing or codebook baselines).
- Long context (W.7).
- Multi-turn (W.9).
- DM ablation cross-product (W.12).

---

End of pre-registration. After this point §3 grid, §5 statistics, and §6
red-lines may not change. New baselines (e.g. R-ROME) require a new
phase, not an amendment to W.10.
