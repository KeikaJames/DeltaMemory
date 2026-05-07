# Exp 6 — Negative controls (PREREG)

## Hypothesis
The recall@1 / margin lift comes from **correct K/V binding**, not from
"any bank perturbation that has the right shape". Specifically:
1. `correct_bank` outperforms all 4 controls on recall@1 (CI > 0).
2. `correct_K_random_V` exhibits non-trivial bank_attention_mass (the right
   addresses are still hit) but has near-floor recall (V is wrong).
3. `random_K_correct_V` has near-floor recall AND near-floor
   bank_attention_mass (the addresses cannot be reached).

## Variants (all α = 1.0, V-scale = auto_rms_cap, SCAR off)
| name                  | bank_perturbation |
|-----------------------|-------------------|
| correct_bank          | none              |
| shuffled_bank         | shuffled (V permuted across facts) |
| random_kv             | random_kv         |
| correct_K_random_V    | random_V_only     |
| random_K_correct_V    | random_K_only     |

## Random K/V protocol
- Same shape, dtype, device as the real bank.
- Per layer, scaled to match the real bank's per-layer RMS of K and V.
- Seeded with `torch.Generator('cpu').manual_seed(0xC0FFEE)`.

## Metrics
recall@1, mean_margin, median_margin, JS/KL drift, bank_attention_mass,
max_bank_prob, target_rank.

## Acceptance gates
- correct_bank's recall@1 CI strictly above each control's CI.
- correct_K_random_V's bank_attention_mass within (0.5×, 1.5×) of correct_bank
  AND its recall@1 CI strictly below correct_bank.
- random_K_correct_V's bank_attention_mass < 0.5× correct_bank's.

## Stop conditions
- If `random_kv` matches `correct_bank` on recall@1 → the effect is not from
  K/V binding; the paper's mechanism claim fails.
