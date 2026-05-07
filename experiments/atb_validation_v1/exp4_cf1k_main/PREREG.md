# Exp 4 — CounterFact-1k Main Result + Protocol (PREREG)

## Hypothesis
On the full filtered CounterFact-1k (~807 prompts after W.6 filter), enabling
AttnNativeBank with α=1 produces a recall@1 lift over the unbanked baseline
that is large in effect size (Δrecall@1 > 0.30) and statistically significant
(McNemar p < 1e-3). With α=0 the model collapses back to baseline up to
bit-level (cross-checked against Exp 3).

## Setup
- Dataset: `experiments/datasets/counterfact_1k.jsonl` (sha1 logged in manifest).
- Filter (W.6 protocol): `paraphrase_prompts` non-empty AND `target_new` /
  `target_true` tokenize to distinct first-3-alpha-token heads.
  Logged: `sampled_size = 1000`, `final_n` (~807).
- Model: pinned per dispatcher (default Gemma-4-31B-it).
- dtype = bf16, attention_impl = eager.
- Seeds: 0, 1, 2.
- write_template: `Fact: {subject} {phrase} {target_new}.`
- read_template: `prompt.format(subject)` (CounterFact's `prompt` field).
- Drift: 100 fixed neutral Wikitext-2 prompts (cached in `_lib/neutral_100.json`).
- enabled_modules: `[AttnNativeBank]`.
- disabled_modules: `[SCAR, CAA, LOPI-skip-list]`. **No `lopi_default` label.**

## Rows
| variant                     | method | α |
|-----------------------------|--------|---|
| none_alpha0                 | none   | 0 |
| AttnNativeBank_alpha0       | anb    | 0 |
| AttnNativeBank_alpha1       | anb    | 1 |

## Metrics
recall@1, mean_margin, median_margin, JS/KL drift, target_new_logprob,
target_true_logprob.

## Statistics
- Paired bootstrap 95% CI (10 000 resamples) on Δrecall@1 and Δmargin
  (ANB α=1 − none).
- McNemar χ² on recall@1 swing (paired by prompt id, pooled across seeds).

## Acceptance gates
- ANB α=1 vs none: Δrecall@1 CI strictly > 0; McNemar p < 1e-3.
- ANB α=0 vs none: |Δrecall@1| within sampling noise; cross-checked against
  Exp 3 bit-equality.

## Stop conditions
- If McNemar p ≥ 1e-3 or CI on Δrecall@1 contains 0, the main result is
  not significant and the paper's headline claim must be revised.
