# Exp13 — ANB QK Addressability (PREREG)

> **Status**: pre-registered before any data run.
> **Branch**: `feat/exp13-anb-readdressability`
> **Locked**: Phase 0 commit (see git log immediately prior to first `run_*` output).

## Hypothesis

**H13 (addressability)**: at read time, the per-layer query `q_l = W_Q^l h_l` produced
on a knowledge prompt naturally aligns with the per-layer key
`M_K[l, ·, fact_correct]` captured at write time from the same fact, more than
with `M_K[l, ·, fact_distractor]` from unrelated facts in the bank, **without
any value injection**. Concretely, the head-mean inner product
`s(l) = ⟨q_l, M_K[l, fact_i]⟩`, reduced as `max_l s(l)`, ranks `fact_correct`
above random distractors and above shuffled-layer controls.

Failure of H13 implies the bank-concat softmax pathway used by Exp8–10 has
no per-fact addressability signal to amplify; injection causality (Exp14+)
is then immaterial.

## Independent variables

| Axis | Levels (smoke) | Notes |
|---|---|---|
| Capture site | `period`, `subject_last`, `object_last` | period is canonical; sites widened in Exp17 |
| Read query | `query` (formatted prompt), `paraphrase` (templated), `subject_only` | `paraphrase` falls back to `query` if CF row has none |
| Bank variant | `correct_in_bank`, `shuffle_layer`, `shuffle_V`, `random_K_only` | `correct_in_bank` is the addressability test; others are negative controls |
| Seed | {0, 1, 2} | 3 independent row orderings |
| Query scoring mode | `pre_rope` (primary), `post_rope` (sanity) | bank K is stored pre-RoPE by default |

Fixed: model `Qwen/Qwen3-4B-Instruct-2507`, dtype bf16, device `mps`,
`attn_implementation=eager`, `use_cache=False`, `mhc_shield=False`,
`lopi_enabled=False`, alpha 0 (no V injection — pure QK).

## Sample

Smoke: full-eligible CounterFact subset after
`filter_cf_for_tokenizer + build_write_prompt` on Qwen3-4B. Prior runs yield
~21 eligible rows; smoke uses ALL eligible rows × seeds {0,1,2}. Rows whose
extended capture spans can't be resolved are logged and excluded only for the
site that failed (other sites still measured).

## Metrics (per row × seed × capture × read-query × bank-variant)

Primary (computed offline from recorded Q and bank M_K):
- `correct_rank` — 0-indexed rank of correct fact under `max_layer` reduction.
- `recall@1`, `recall@5` — indicator.
- `correct_score`, `best_other_score`, `score_gap = correct - best_other`.
- `top_layer` — layer at which correct fact achieves max score.
- `top_index` — bank slot achieving global max.

Secondary:
- `hard_negative_win_rate` over `same_subject_wrong_object`,
  `same_relation_wrong_subject`, `same_object_wrong_subject` indices.
- `score_gap_post_rope` (sanity, separate scoring pass).

## Aggregation & inference

- Bootstrap CI: paired-by-row, B=2000, 95%, seed-fixed (0xCAFE).
- Primary contrast: `mean(score_gap)` under `correct_in_bank` vs each control,
  with paired bootstrap CI on the difference.
- Mean `recall@1`, `recall@5` reported per (capture, query) cell.

## Verdict ladder (encoded in `analyze.py`)

Same 6-way ladder as the master plan §13.

For Exp13 alone:
- **ADDRESSABILITY_STRONG**: ∃ (capture, query) such that
  `mean recall@5 ≥ 0.50` and `score_gap CI_lo > 0` versus all three controls
  (`shuffle_layer`, `shuffle_V`, `random_K_only`).
- **ADDRESSABILITY_DIRECTIONAL**: `mean recall@5 ≥ 0.30` and `score_gap > 0`
  on the point estimate vs ALL controls (CI not required to exclude zero).
- **FAIL (Exp13)**: `max over (capture, query) of mean recall@5 < 0.15`
  AND `score_gap CI_lo ≤ 0` versus controls.

## Stop-rule

If Exp13 = FAIL → skip Exp14/Exp15/Exp18 on this branch and jump directly
to Exp17 capture-site sweep (the master plan §17 fallback). If Exp17 also
returns no site with addressability, the entire ANB-natural-addressing
hypothesis is closed and the branch lands a `VERDICT.md` with FAIL_DEEP.

## Falsifiability

`correct_in_bank` recall@5 must beat `shuffle_layer` and `shuffle_V` and
`random_K_only` simultaneously to claim ADDRESSABILITY. If `shuffle_V`
matches `correct_in_bank` we still have an ADDRESSABILITY claim (V doesn't
matter for QK), but `shuffle_layer` matching means layer-identity carries
no information — that is a `FAIL` for the layer-routed retrieval model.
`random_K_only` matching means K isn't even distinguishable from RMS-matched
noise, i.e. the bank is geometrically meaningless.

## Implementation freeze

Code paths used (locked at Phase 0):
- `deltamemory.memory.attn_native_bank.AttnNativePatcher` (with new
  `record_queries` knob — additive; bit-equal when off).
- `deltamemory.memory.anb_diagnostics.{record_read_queries, score_query_against_bank, rank_correct, hard_negative_win_rate}`.
- `deltamemory.memory.anb_addressed.{subbank_shuffle_layer, subbank_shuffle_V}`.
- `experiments.atb_validation_v1._lib.{load_model, load_counterfact, filter_cf_for_tokenizer}`.
- `experiments.atb_validation_v1._lib.hard_negatives.build_hard_negatives`.

No new model loading paths, no SCAR, no mHC, no LOPI.

## Output

```
run_mps_smoke/
  manifest.json
  rows.jsonl               # one row per (seed, row_id, capture, query, variant, scoring_mode)
  summary.json             # aggregated by (capture, query, variant)
  analysis.json            # bootstrap diffs + verdict
  VERDICT.txt              # one-word verdict
```
