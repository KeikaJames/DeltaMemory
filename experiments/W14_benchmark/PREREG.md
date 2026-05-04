# W.14 Pre-Registration: Standard Benchmark Suite vs ROME / MEMIT / GRACE

**Status**: locked.
**Authored**: 2026-05-04.
**Depends on**: W.4 verdict (`M_winner`); W.6 verdict (`DM_best_alpha`);
  W.10 (B1 prompt-insertion numbers under identical tokenisation as a
  paired anchor).
**Hardware target**: 128 GB unified memory recommended (ROME/MEMIT
  weight edits require materialising the gradient of the projected MLP
  layer; this is heavy on Qwen2.5-1.5B and gemma-3-1b-it). 64 GB host
  may complete the gpt2-medium and Qwen2.5-0.5B sweeps only.

W.14 is the **public-benchmark answer**. Reviewers will not accept
self-authored counterfact prompts as the primary evaluation. W.14 runs
the proposed `M_winner @ DM_best_alpha` configuration against three
fixed-corpus benchmarks and against three published method baselines, in
the same harness, on the same models.

The published-baseline numbers from the original ROME/MEMIT/GRACE papers
are explicitly **not used** as quoted comparisons. Every baseline is
re-executed in this harness to produce paired observations. If
re-execution diverges from the published number by more than 5 absolute
percentage points on Efficacy Score, the deviation is documented in the
report — but the **paired comparison stands** because both methods see
the same prompts under the same harness.

---

## 1. Datasets (public splits, no resampling)

| dataset    | source                                         | items used | sha sentinel                |
| ---------- | ---------------------------------------------- | ---------- | --------------------------- |
| CounterFact| Meng+ 2022 (ROME, public release v1.0)         | first 2000 | `experiments/datasets/counterfact_2k.jsonl.sha` |
| ZsRE       | Levy+ 2017 (zsRE editor split, MEMIT release)  | first 2000 | `experiments/datasets/zsre_2k.jsonl.sha`        |
| MQuAKE-CF  | Zhong+ 2023 (MQuAKE v1.0, single-hop subset)   | 1000       | `experiments/datasets/mquake_1k.jsonl.sha`      |

Splits are downloaded once via the dataset's official release URL and
SHA-pinned. No resampling, no filter beyond "tokenises under all 5
target models without OOV in subject string".

## 2. Methods

| arm     | description                                          | edit type     | code path                                  |
| ------- | ---------------------------------------------------- | ------------- | ------------------------------------------ |
| **DM**  | M_winner @ DM_best_alpha (per-model from W.6)        | activation    | `deltamemory/memory/{caa_injector,lopi}.py`|
| **B0**  | no_memory frozen LLM                                 | none          | `deltamemory.engine.prototype` baseline    |
| **B1**  | prompt-insertion (paired, from W.10 contract)        | prompt prefix | `experiments/W10_vs_baselines/run.py`      |
| **B3**  | ROME (Meng+ 2022)                                    | weight        | vendored `third_party/rome/`               |
| **B4**  | MEMIT (Meng+ 2022, batch=100)                        | weight        | vendored `third_party/memit/`              |
| **B5**  | GRACE (Hartvigsen+ 2023)                             | codebook      | vendored `third_party/grace/`              |

ROME/MEMIT/GRACE are vendored at locked commits recorded in
`third_party/COMMITS.md`. Any vendored patch (e.g. for
transformers compatibility) is documented diff-style in
`third_party/PATCHES.md` and the patched code shipped under our LICENSE
attribution. No re-implementation; original weights/checkpoints used
where available.

## 3. Metrics (CounterFact protocol)

For every (model, dataset, item, method) cell:
- **Efficacy Score (ES)**: P(target_new) > P(target_true) on the edit
  prompt itself. Binary; reported as accuracy across items.
- **Paraphrase Score (PS)**: ES averaged over 10 paraphrases (provided
  by CounterFact / generated for ZsRE/MQuAKE per dataset's released
  paraphrase splits).
- **Neighborhood Score (NS)**: ES on neighborhood prompts unrelated to
  the edit; **higher = better specificity**, so success here means
  ES_neighborhood is **unchanged from B0**.
- **Generation Score (GS)**: top-1 generation under greedy decoding
  contains target_new substring.

Paired observation unit: `(model, dataset, item)`. ROME/MEMIT/GRACE all
have well-defined sequential edit semantics; for each item we re-load
the base model state before edit (or for MEMIT, batch 100 then evaluate
on each batched item).

## 4. Grid

`5 models * 3 datasets * 6 methods * 1 seed = 90 (model, dataset,
method) combinations`. Items per combo are dataset-fixed (2000 / 2000 /
1000). Total per-item evaluations: `5 * (2000+2000+1000) * 6 = 150,000`.

Single seed, because the methods are deterministic given the dataset
order and the harness fixes that order.

## 5. Hypotheses

**H14a (DM not catastrophic on specificity)**:
  Across (model, dataset) pairs, `NS(DM) - NS(B0) > -0.05` on at least
  13 of the 15 pairs. ROME/MEMIT historically lose 10–25 percentage
  points on NS at scale; activation-injection methods should not.

**H14b (DM beats prompt-insertion on at least one bench)**:
  On at least one of {CounterFact, ZsRE, MQuAKE}, DM ES > B1 ES paired
  test p<0.01 across items, on >=3 of 5 models.

**H14c (DM does not lose to prompt-insertion on >=2 benchmarks)**:
  On at least 2 of 3 benchmarks, paired test of `DM - B1` is not
  significantly negative at p<0.05 on >=4 of 5 models.

**H14d (NS preserved relative to weight-editing)**:
  Mean `NS(DM) - NS(B3 ROME)` > 0.05 averaged across (model, dataset).
  This is the headline claim "we do not damage unrelated knowledge".

**H14e (efficacy not catastrophic)**:
  `ES(DM) > 0.50` on at least 2 of 3 benchmarks for at least 3 of 5
  models. This is a sanity floor, not an outperformance claim.

## 6. Statistics

H14b/c: paired McNemar's test on binary item-level ES, Holm-Bonferroni
across the 3-bench x 5-model = 15 family. Threshold 0.01 for H14b, 0.05
for H14c.

H14a/d: per-pair difference in proportion with Wilson 95% CI;
non-overlap with 0 at the threshold counts as a pair-level pass.

H14e: per-cell Wilson CI on ES.

## 7. Red-lines and aborts

1. `alpha=0` bit-equality on DM arm: `|nll_new(DM,alpha=0) - nll_new(B0)|
   < 1e-4` on at least one held-out item per (model, dataset).
2. Vendored ROME/MEMIT/GRACE harness self-test: re-execute the
   ROME author's own CounterFact-2000 sample for gpt2-xl and confirm
   `ES > 0.95`. If our harness drops below 0.85, abort and flag harness
   regression. (gpt2-xl is included as a harness-witness model only and
   is not part of the 5-model paired evaluation.)
3. Memory cap: 64 GB host aborts on Qwen2.5-1.5B / gemma-3-1b-it for B3/B4;
   defer those cells to 128 GB run.
4. Sequential-edit drift: between item N and item N+1, B3 ROME's edited
   model state must be reset. Failure to reset is detected by hashing
   the gpt2-medium first-block weights at start of each item and
   asserting equality. Hash mismatch -> abort.

## 8. Deliverables

- `cells.jsonl`             — per (model, dataset, method, item) row.
- `summary.json`            — H14a/b/c/d/e verdicts + 5x3x6 ES/PS/NS/GS table.
- `figures/`                — leaderboard bar (3 figures, one per dataset).
- `REPORT.md`               — narrative + leaderboard table.
- `env.json`                — env hash, vendored commit SHAs, dataset SHAs.
- `third_party/COMMITS.md`  — frozen ROME/MEMIT/GRACE commit IDs.

## 9. Out of scope

- LLaMA-3.2 / Qwen3 (not yet supported by ROME/MEMIT vendor
  implementations; pending W.14.2 follow-up).
- Sequential-edit fragility (separate W.14.3 phase).
- Long-context benchmark integration (NocrutCM / RULER) — W.7 follow-up.

---

End of pre-registration. The vendored commit SHAs in
`third_party/COMMITS.md` and the dataset SHAs in
`experiments/datasets/{counterfact_2k,zsre_2k,mquake_1k}.jsonl.sha` are
pinned at the moment of W.14 launch; later upstream changes to
ROME/MEMIT/GRACE require a new W.14.X phase, not an in-place amendment.
