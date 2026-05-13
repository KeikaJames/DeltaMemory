# Exp 7 — Non-Gemma Pre-RoPE Negative Controls (PREREG)

**Status**: locked  
**Model**: Qwen3-4B-Instruct-2507  
**Replaces / complements**: Exp 6 / Exp 6b (both Gemma-4-31B, both invalidated for K-addressing test)

## Motivation

Exp 6 used `bank_key_mode="pre_rope"` on Gemma-4-31B-it and Exp 6b used `post_rope`
on the same model.  Both failed to prove correct K/V binding dominates for different reasons:

| Exp | Failure mode |
|-----|-------------|
| Exp 6 (pre_rope, Gemma-4-31B) | Gemma-4 native V-norm → `auto_rms_cap` produces scale≈1.0 → pre_rope injection is a near-no-op |
| Exp 6b (post_rope, Gemma-4-31B) | post-RoPE K is position-specific → correct K doesn't generalise write→read positions |
| Exp 6 / 6b (both) | `bank_size=1` → shuffled perturbation is always a no-op (`n < 2` guard) |

Exp 7 fixes all three issues:

1. **Model without native V-norm**: Qwen3-4B-Instruct-2507
   - `model_type=qwen3`, standard GQA (32 attn heads, 8 KV heads, 128 head_dim)
   - No `v_norm` or `qk_norm` in config → `pre_rope` injection operates at full scale
2. **`bank_key_mode="pre_rope"`**: position-invariant K-addressing (the ATB design goal)
3. **`bank_size=200`**: 1 target + 199 distractors → shuffled = genuine 200-row permutation

## Hypotheses

**H7.1 (correct binding dominates)**:
`correct_bank` mean_margin > shuffled_bank, random_kv, correct_K_random_V,
random_K_correct_V — all measured in the pre_rope regime on Qwen3-4B.

**H7.2 (target_rank)**:
`correct_bank` mean target_rank ≤ (better or comparable to) all controls.

**H7.3 (K addressing works pre_rope)**:
`random_K_correct_V` shows lower margin than `correct_bank`, confirming that
correct K is necessary for addressing in pre_rope mode (unlike post_rope mode where
V dominates because K is position-sensitive).

**H7.4 (shuffled degrades)**:
`shuffled_bank` mean_margin < `correct_bank`, confirming that fact-level K/V
binding matters (correct K → wrong V → lower margin).

## Success Criteria

All three must hold:

1. `correct_bank` has the **highest mean_margin** among all 5 variants.
2. `correct_bank` has **better or comparable mean target_rank** than controls (lower = better).
3. `shuffled_bank` and `random_K_correct_V` must **not** outperform `correct_bank`
   on either metric.

## Variants (all α=0.05, bank_key_mode=pre_rope, value_scale_mode=auto_rms_cap, SCAR off)

| Variant | K | V | bank_perturbation |
|---------|---|---|-------------------|
| correct_bank | correct (pre_rope) | correct | none |
| shuffled_bank | rows permuted | rows permuted | shuffled |
| random_kv | random (RMS-matched) | random (RMS-matched) | random_kv |
| correct_K_random_V | correct | random (RMS-matched) | random_V_only |
| random_K_correct_V | random (RMS-matched) | correct | random_K_only |

## Bank construction (bank_size=200)

For each (seed, prompt):
- Target fact (1): write with template `Fact: {subject} {phrase} {target_new}.`
- Distractors (199): sampled from other eligible facts using
  `random.Random(seed ^ (hash(prompt_id) & 0xFFFF_FFFF))`
- Base bank written ONCE; cloned and perturbed per variant.

## Random K/V protocol

- Same `shape`, `dtype`, `device` as the real bank.
- Per layer, scaled to match the real bank's per-layer RMS of K / V respectively.
- Seeded: `torch.Generator('cpu').manual_seed(0xC0FFEE ^ seed)`.

## Fixed settings

| Item | Value |
|------|-------|
| model | Qwen3-4B-Instruct-2507 |
| dataset | counterfact_1k.jsonl, W.6 filter |
| expected eligible | ~807 |
| seeds | 0, 1, 2 |
| α (primary) | 0.05 |
| α (secondary, optional) | 0.10 |
| bank_size | 200 |
| dtype | bfloat16 |
| attention_impl | eager |
| write_template | `Fact: {subject} {phrase} {target_new}.` |
| read_template | `prompt.format(subject)` |
| SCAR | disabled |
| LOPI | disabled |

## Metrics (per variant, aggregated over seeds × eligible)

- `n` — number of (seed, prompt) cells
- `recall_at_1` — fraction where target_new is rank-0 next token
- `mean_margin` — mean of log p(target_new) − log p(target_true)
- `median_margin`
- `95% CI` — bootstrap percentile over prompts
- `JS drift` — symmetric JS vs unrelated neutral prompts
- `KL drift`
- `bank_attention_mass` (if available from patcher observability hooks)
- `max_bank_prob` (if available)
- `target_rank`

## Expected cell count

807 eligible × 5 variants × 3 seeds = **12,105 cells**

## Interpretation guide

If H7.1 passes (correct_bank best margin):
→ Pre-RoPE K-based content-addressable retrieval is confirmed on non-V-norm models.
→ ATB mechanism works as designed.

If H7.1 fails but shuffled_bank ≈ correct_bank (shuffled dominates):
→ Possible leak: the shuffled bank at bank_size=200 may still accidentally route
   attention to the target V (shared subject-term K in distractors).
→ Next step: analyse bank_attention_mass for the target slot.

If H7.3 fails (random_K_correct_V ≈ correct_bank):
→ Pre-RoPE V injection is the main driver, not K addressing (like post_rope).
→ This would suggest even pre_rope V is dominating (α too high, or V too large).
→ Retry with lower α (0.02).
