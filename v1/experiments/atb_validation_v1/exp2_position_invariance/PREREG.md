# Exp 2 — pre-RoPE vs post-RoPE position invariance (PREREG)

## Hypothesis
ATB's pre-RoPE bank keys are position-invariant: read accuracy / margin is
flat across `position_delta ∈ {0, 128, 512, 1024}`. Post-RoPE keys, by
contrast, decay monotonically because the read query carries a different
positional rotation than the bank key.

## Setup
- Fact = single CounterFact-1k row (50 fact-prompt pairs sampled, 3 filler seeds).
- Read prompt = `[FILLER × position_delta] + canonical_query`.
- Filler = neutral Wikitext-2 sentences, scrubbed of subject / target_true /
  target_new tokens, truncated/padded to exactly `position_delta` tokens.
- α = 1.0; V-scale = auto_rms_cap; SCAR off.

## Variants
| name             | bank_key_mode |
|------------------|---------------|
| pre_rope_bank    | pre_rope      |
| post_rope_bank   | post_rope     |

## Metrics
position_delta, recall@1, mean_margin, target_rank, bank_attention_mass.

## Acceptance gates
- pre_rope_bank: linear regression of mean_margin on position_delta — slope
  CI contains 0; recall@1 CI on each delta overlaps the delta=0 cell.
- post_rope_bank: at least one of {128, 512, 1024} has mean_margin CI strictly
  below the delta=0 mean_margin CI.

## Stop conditions
- If pre_rope_bank also degrades, the position-invariance claim is unsupported.
