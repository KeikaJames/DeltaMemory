# W.13 Pre-Registration: Synthetic Ground-Truth Stress

**Status**: locked.
**Authored**: 2026-05-04.
**Depends on**: W.4 verdict (`M_winner`); W.6 verdict (`DM_best_alpha`).
  W.13 may run before either lands using `M_winner = "caa"` and
  `DM_best_alpha = 1.0` as **provisional defaults** for a smoke pass; the
  full grid blocks on the real verdicts.
**Hardware target**: 64 GB unified memory; entire grid in scope of MPS
  bfloat16. No MoE.

W.13 is the experiment that answers the question every reviewer asks:
"how do you know the lift you measured on natural-language counterfacts
is not just noise from the LLM's prior?"  W.13 strips the LLM prior to
zero by routing through random-string keys, so any recall is causally
attributable to the bank.

---

## 1. Three task families

### Task A — Synthetic Dictionary Recall (the sanity floor)

A bank is populated with `N` (key, value) pairs where each key is a
fresh 8-character base32 string (`[a-z2-7]^8`, 5e12 entries) and each
value is an independent 8-character base32 string. Both are tokenised
with the model's own tokenizer; pairs whose key or value tokenises to
fewer than 4 tokens are rejected and re-sampled until N pairs remain
(the multi-token requirement makes the task non-trivial).

Write phase: for each pair `(k_i, v_i)`, call
`write_fact(bank, address=k_i, prompt=f"{k_i} -> {v_i}", fact_id=i)`.

Read phase: for each pair, present `f"{k_i} -> "` and ask the model to
generate the next `len(tokenize(v_i))` tokens.

The metric is `recall@1` — the fraction of pairs for which the model's
top-1 sequence equals `tokenize(v_i)` exactly. The natural baseline is
catastrophic: with random base32 keys/values, prior-only `recall@1` is
bounded by `(1/|vocab|)^len_v <= 1e-20` for any reasonable vocab. So any
non-trivial `recall@1` is **100% attributable to the bank**.

Levels: `N in {10, 100, 500, 1000}`. The break point where `recall@1`
collapses is the bank's **effective capacity** for that model.

### Task B — Synthetic Counter-Prior

Generate 100 prompts of the form
`"The capital of {fake_country} is {fake_city}."` where `fake_country`
and `fake_city` are 6-character pseudo-words drawn from a phonotactically
plausible C(VC){2,3} grammar but tested against the model's own
embedding to ensure they are out-of-distribution (cosine to nearest 1k
real-token embedding centroid > 0.7, model-conditional). Strict
randomness check: each pseudo-word is rejected if it matches any
substring of the model's tokenizer vocabulary as a whole token.

Write the bank with the (fake_country, fake_city) mapping using the same
machinery as W.6. Then evaluate `nll_new` on the next-token under the
factual completion. Compare against:
- B0 no-memory (frozen LLM)
- B1 prompt-insertion ("Fact: ...\n" prefix)

Hypothesis H13B: `M_winner @ DM_best_alpha` produces lower `nll_new` than
B1 on at least 4/5 dense models, paired Wilcoxon p<0.01 after Holm.
Because the prior is engineered to be uniform, the bank should dominate
prompt insertion here more clearly than on natural counterfacts (W.6).

### Task C — Needle-in-Haystack with Bank

A 16,384-token context is constructed by concatenating LAMA-style
distractor sentences. A **single needle** sentence
`"The {fake_attribute} of {fake_subject} is {fake_value}."` is inserted
at relative position `pos in {0.05, 0.25, 0.50, 0.75, 0.95}` within the
context. The bank receives the same fact via `write_fact`. The eval
prompt then asks `"What is the {fake_attribute} of {fake_subject}?"` and
records `recall@1`.

Three arms are compared at each position:
- **prompt-only**: needle in context, bank empty.
- **bank-only**: bank has needle, context is distractor-only (needle
  removed).
- **both**: needle in context AND in bank.

Hypothesis H13C: bank-only `recall@1` is **flat** with respect to
position (within +/- 0.05 over all 5 positions); prompt-only follows the
classic U-shape (high at 0.05 and 0.95, low in the middle). The flatness
of bank-only is the empirical signature that justifies "memory" as a
distinct primitive from "context".

## 2. Grid

| dim | Task A | Task B | Task C |
| --- | --- | --- | --- |
| models | 5 dense | 5 dense | 5 dense |
| methods | {none, M_winner@DM_best_alpha} | {none, B1, M_winner} | {prompt-only, bank-only, both} |
| levels | N in {10,100,500,1000} | 100 prompts | 5 positions |
| seeds | 3 | 3 | 3 |
| cells | 5*2*4*3 = 120 | 5*3*100*3 = 4500 | 5*3*5*3 = 225 |

Total: **4,845 cells**. Estimated wall-clock on MPS: 90–120 min.

## 3. Hypotheses

- **H13A1 (capacity floor)**: at `N=100`, `M_winner` `recall@1 >= 0.90`
  on at least 3/5 models.
- **H13A2 (capacity break)**: at `N=1000`, `recall@1` decay rate
  (slope of log(1-recall) vs log N) is monotone non-increasing on every
  model — i.e. capacity degrades smoothly, not catastrophically.
- **H13B (counter-prior dominance)**: `M_winner` beats B1 with paired
  Wilcoxon p<0.01 on >=4/5 models.
- **H13C (position invariance)**: bank-only `recall@1` standard
  deviation across positions is < 0.05 on every model.

A **single-arm fail** on H13A1 (every model below 0.90 at N=100) means
the retrieval path is not behaving as the equations claim and the v0.4
release is blocked pending a bug-hunt phase.

## 4. Red-lines and aborts

- alpha=0 bit-equality on Task A `none` arm: max `|drift|` < 1e-4.
- Tokeniser determinism: re-tokenising any generated value must match.
- Random-key collision: each batch's `set(keys)` must equal `len(keys)`;
  enforced in `run.py` line ~120.
- Vocab-substring leak in Task B: pseudo-word that appears in tokenizer
  vocab as a single token is rejected before write.
- Memory cap: process RSS > 48 GB at any cell start triggers an abort
  for that model (drop to next).

## 5. Statistics

- H13A1 / H13C: per-model thresholds, no aggregate test.
- H13A2: monotone-decreasing test on the four-point recall curve via
  Spearman rank correlation across log(N), reject at p<0.05.
- H13B: paired Wilcoxon, Holm-Bonferroni across 5 models, threshold 0.01.

Bootstrap 95% CIs on `recall@1` use 1000 stratified resamples per cell.

## 6. Deliverables

- `cells.jsonl`     — 4,845 rows.
- `cells_smoke.jsonl` — gpt2-medium, A only, N in {10, 100}, 1 seed.
- `summary.json`    — H13A1/A2/B/C verdicts + per-model recall curves.
- `figures/`        — recall vs N (1 figure), needle position curve
  (1 figure), counter-prior nll histogram (1 figure).
- `REPORT.md`       — narrative.
- `env.json`        — env hash, M_winner provenance, alpha provenance.

## 7. Out of scope

- Compare against ROME/MEMIT/GRACE (W.14).
- Multi-fact bank stress (W.8 covers natural-language; W.13 only single
  pair per cell for Task B/C).
- Long context > 16,384 (W.7).

---

End of pre-registration. Hypotheses, grid, and statistics are frozen.
Random seeds for key/value generation are fixed at run-launch and
recorded in `env.json` so that re-execution is bit-reproducible.
