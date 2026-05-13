# Exp8 Pre-Registration: mHC-Smoothed Pre-RoPE Negative Controls

**Registered:** 2025-05-09 (before any Phase A run)

---

## Motivation

Exp7 used raw ATB (mhc_shield=False) at α=0.05 on Qwen3-4B-Instruct-2507.
All 5 variants returned negative margins (model prefers original facts).
Diagnosis: `pattern_v_dominates=True` — injected V content overwhelms K routing
when α is large enough to carry signal regardless of K correctness.

Exp8 tests whether the mHC spectral shield (`shield_attention_weights`, kappa
bound on bank-column attention sums) suppresses V-path dominance and restores
the K–V binding contrast required for a valid ATB signal.

---

## Hypotheses

**H8.1 (primary):** Under mHC-on α=0.05, `correct_bank` achieves the highest
mean margin among all 5 variants.

**H8.2:** `random_kv` and `random_K_correct_V` margins *decrease* relative to
Exp7 raw baseline (or no longer exceed `correct_bank`), indicating mHC suppresses
V-path dominance.

**H8.3:** JS/KL drift for `correct_bank` does not increase relative to Exp7
(mHC does not increase semantic drift on the base distribution).

**H8.4 (strict):** `correct_bank` 95 % CI strictly dominates all control
variants at best kappa (CI lower bound > all-controls CI upper bound).

---

## Design

| Item | Value |
|------|-------|
| model | Qwen3-4B-Instruct-2507 |
| bank_key_mode | pre_rope |
| value_scale_mode | auto_rms_cap |
| bank_size | 200 |
| seeds | 0, 1, 2 |
| dtype / attn | bf16 / eager |
| dataset | CounterFact-1k 807-eligible subset |

### Phase A — kappa smoke (n=200 per kappa)

kappas tested: **1.0, 0.5, 0.25**
5 variants × 3 seeds × 200 prompts × 3 kappas = **9000 cells**

### Phase B — full run (n=807, best kappa from Phase A)

Best kappa is selected as: `argmax_κ (correct_bank_margin − max_control_margin)`.
This maximises the gap between `correct_bank` and the strongest control,
rather than correct_bank's absolute margin in isolation.

5 variants × 3 seeds × 807 = **12 105 cells**

### Phase C — high-alpha stress (n=200, best kappa, 3 variants)

alphas: 0.10, 0.20, 0.50, 1.00
3 variants × 3 seeds × 200 × 4 alphas = **7200 cells**

---

## Variants

| variant | K | V | mhc_shield |
|---------|---|---|------------|
| correct_bank | correct (pre_rope) | correct | **True** (kappa swept) |
| shuffled_bank | correct (pre_rope) | V rows permuted across facts | True |
| random_kv | random RMS-matched | random RMS-matched | True |
| correct_K_random_V | correct | random | True |
| random_K_correct_V | random | correct | True |

---

## Success Criteria

- **STRONG PASS:** H8.1 + H8.4 both met
- **DIRECTIONAL PASS:** H8.1 met, H8.4 not met
- **FAIL:** H8.1 not met at any tested kappa

If FAIL: Phase C tests whether mHC prevents high-alpha runaway; lower-alpha rescue is left for a separate experiment.

---

## What This Does Not Test

- Post-RoPE key mode (Exp6/6b scope)
- SCAR (disabled)
- LOPI (disabled)
- Gemma family (Exp7 scope)
