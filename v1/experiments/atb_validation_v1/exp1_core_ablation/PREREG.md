# Exp 1 — Core Ablation (PREREG)

## Hypothesis
The AttnNativeBank (ATB) core mechanism — bank read at the attention layer —
is the source of the recall@1 / margin lift. We further hypothesise:
1. `pre_rope_bank_only > no_bank` (CI > 0): the core read works without V-scale.
2. `pre_rope_vscale ≥ pre_rope_bank_only`: V-scale is a stabiliser, not the
   source of the effect.
3. `full_attnnativebank ≈ pre_rope_vscale`: production default does not depend
   on SCAR / LOPI-skip-list (both explicitly disabled).
4. `post_rope_bank` underperforms `pre_rope_bank` at the matched V-scale.

## Variants
| name | method | bank_key_mode | value_scale_mode | SCAR |
|---|---|---|---|---|
| no_bank              | none | —         | —             | off |
| post_rope_bank       | anb  | post_rope | auto_rms_cap  | off |
| pre_rope_bank_only   | anb  | pre_rope  | none          | off |
| pre_rope_vscale      | anb  | pre_rope  | auto_rms_cap  | off |
| full_attnnativebank  | anb  | pre_rope  | auto_rms_cap  | off |

α = 1.0 for all anb variants. Same prompts, same seeds across variants.

## Metrics
recall@1, mean_margin, median_margin, JS/KL drift, bank_attention_mass,
max_bank_prob, target_rank.

margin = log p(target_new) − log p(target_true) (multi-token = sum of token
logprobs).

## Acceptance gates
- pre_rope_bank_only vs no_bank: paired bootstrap 95% CI on Δrecall and Δmargin
  must be strictly > 0.
- pre_rope_vscale ≥ pre_rope_bank_only on mean_margin (point estimate).
- full_attnnativebank ≥ pre_rope_vscale (point estimate; CI may overlap).

## Stop conditions
- If pre_rope_bank_only's CI on Δmargin contains 0, the paper's core claim
  (bank read alone is sufficient) is unsupported and must be revised.
