# Exp 6b — post-RoPE Negative Controls (PREREG)

**Status**: locked  
**Replaces / complements**: Exp 6 (which used `pre_rope` — a near-no-op on Gemma-4-31B)

## Motivation

Exp 6 was designed to test whether correct K/V binding is necessary for margin lift.
However, all Exp 6 variants used `bank_key_mode="pre_rope"`, which is effectively a
no-op on Gemma-4-31B-it: Gemma-4 applies native per-head RMS normalisation on V, making
`auto_rms_cap` produce scale≈1.0 and leaving pre_rope injection at baseline-level margin
(−0.632, indistinguishable from no_bank −0.653).

Exp 6b reruns the identical 5 variants under `bank_key_mode="post_rope"`, the only mode
confirmed to produce positive mean margin on Gemma-4-31B-it (+0.088, Exp 1).
This is the correct regime to evaluate whether the lift is due to *correct* K/V binding.

## Hypothesis

**H_6b.0 (red-line)**: at α=0 (not run here, inherited from Exp 3), all variants produce
bit-identical logits. Assumed satisfied; do not re-run.

**H_6b.1 (correct binding dominates)**:
`correct_bank` mean_margin > shuffled_bank, random_kv, correct_K_random_V,
and random_K_correct_V — all measured in the post_rope regime.

**H_6b.2 (K-addressing necessary)**:
`random_K_correct_V` shows significantly lower bank_attention_mass than `correct_K_random_V`,
confirming K controls addressing.

**H_6b.3 (V-content necessary)**:
`correct_K_random_V` may show non-trivial bank_attention_mass (addresses found) but
near-correct_bank or worse margin (V content wrong → injection is incoherent).

## Variants (all α=1.0, bank_key_mode=post_rope, value_scale_mode=auto_rms_cap, SCAR off)

| Variant | K | V | bank_perturbation |
|---------|---|---|-------------------|
| correct_bank | correct | correct | none |
| shuffled_bank | shuffled rows | shuffled rows | shuffled |
| random_kv | random (RMS-matched) | random (RMS-matched) | random_kv |
| correct_K_random_V | correct | random (RMS-matched) | random_V_only |
| random_K_correct_V | random (RMS-matched) | correct | random_K_only |

## Random K/V protocol
- Same `shape`, `dtype`, `device` as the real bank.
- Per layer, scaled to match the real bank's per-layer RMS of K and V respectively.
- Seeded: `torch.Generator('cpu').manual_seed(0xC0FFEE)`.
- Seed is stable across variants sharing the same fact; each seed gets its own generator.

## Fixed settings

| Item | Value |
|------|-------|
| model | google/gemma-4-31b-it |
| dataset | counterfact_1k.jsonl, W.6 filter |
| expected eligible | ~807 |
| seeds | 0, 1, 2 |
| α | 1.0 |
| dtype | bfloat16 |
| attention_impl | eager |
| write_template | `Fact: {subject} {phrase} {target_new}.` |
| read_template | `prompt.format(subject)` |

## Metrics (per variant, aggregated over seeds × eligible)

- `n` — number of (seed, prompt) cells
- `recall_at_1` — fraction where target_new is rank-0 next token
- `mean_margin` — mean of log p(target_new) − log p(target_true)
- `median_margin`
- `95% CI` — bootstrap percentile over prompts
- `JS drift` — symmetric JS vs unrelated neutral prompts
- `KL drift`
- `bank_attention_mass` (if available)
- `max_bank_prob` (if available)
- `target_rank`

## Acceptance gates

1. `correct_bank` mean_margin is the **highest** among all 5 variants.
2. `correct_K_random_V` bank_attention_mass ≥ 0.5× correct_bank's (addresses found),
   but recall@1 ≤ correct_bank's (V content wrong).
3. `random_K_correct_V` bank_attention_mass < 0.5× correct_bank's (cannot address).
4. `random_kv` mean_margin should not match `correct_bank` (if it does, mechanism fails).

## Stop conditions

- If correct_bank mean_margin is **not** the highest, the lift in post_rope mode is not
  from correct K/V binding. This does not invalidate the margin improvement (Exp 1/4),
  but it complicates the mechanistic claim.
