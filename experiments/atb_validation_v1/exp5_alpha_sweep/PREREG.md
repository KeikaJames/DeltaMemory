# Exp 5 — α dense sweep (PREREG)

## Hypothesis
There exists a smooth, monotone-ish response of margin / recall / drift to α
on a 200-fact bank. Earlier internal claims of an "α≈0.25 cliff" were
empirically debunked (`fix/safe-alpha-empirical`); this experiment makes a
**non-claim**: we report the curve as observed, with explicit verdicts on
whether any discontinuity is present.

## Setup
- Model: Gemma-4-31B-it (paper-pinned).
- Bank size: 200 facts; bank composed of one target row + 199 distractors
  drawn from the W.6-filtered CounterFact-1k.
- Seeds: 0, 1, 2 (controls bank order / RNG; dataset is deterministic).
- α grid: 0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75,
  1.00, 1.50, 2.00.

## Metrics (per cell)
alpha, recall@1, mean_margin, median_margin, JS/KL drift,
bank_attention_mass, max_bank_prob, target_rank, residual_delta_norm,
o_bank_norm, o_seq_norm, obank_oseq_ratio.

## Plots
- α vs mean_margin
- α vs JS drift
- α vs bank_attention_mass
- α vs obank_oseq_ratio

## Acceptance gates
- Curves are well-defined and monotone within bootstrap noise OR an explicit
  numeric verdict on any α at which margin / drift shows a > 2σ break.
- Do **not** claim a "cliff" if the data does not show one. The paper section
  must report whatever the curve actually is.

## Stop conditions
- N/A — this is a descriptive experiment.
