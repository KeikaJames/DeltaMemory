# Exp9 Pre-Registration: Residual-Gated mHC AttnNativeBank

**Status:** Pre-registered (implemented before results collected)
**Follows:** Exp8 mHC-Smoothed Pre-RoPE Negative Controls

---

## Title

Exp9: Residual-Gated mHC AttnNativeBank — Does Legacy Residual Smoothing Transfer?

---

## Goal

Determine whether adding a residual-style beta output gate to AttnNativeBank (ATB) restores
correct-bank selectivity that was absent in Exp7 (raw ATB) and Exp8 (mHC-only ATB).

---

## Why Exp8 Was Insufficient

Exp8 tested **mHC column capping** inside the raw **merged-softmax** ATB path. It showed
`pattern_v_dominates=True` even at kappa=0.25 — the V-path dominance persisted regardless of
how hard we cap bank column attention mass.

Exp8 did NOT test **output-level residual gating**. The legacy delta-QKV Mneme path that
succeeded used:

    x' = x + alpha_scale * sigmoid(gate) * Delta_x

where `alpha_scale * sigmoid(gate_bias) ≈ 0.05` provided strong output smoothing.

The current ATB merged-softmax path has no equivalent:

    O = O_seq_merged + O_bank_merged    (Exp7/Exp8, no beta gate)

Exp9 introduces:

    O = O_seq + beta * O_bank           (with or without mHC on bank weights)

Two structural variants are tested to isolate the cause:

1. **merged_beta_mhc**: keep merged softmax structure, apply beta and mHC to out_bank.
   Minimal delta from Exp8 — if this fixes gap, beta smoothing alone is sufficient.

2. **sep_beta_mhc**: decouple bank from sequence softmax, apply beta and mHC.
   If only this works, softmax dilution (not just beta) was the bottleneck.

---

## Mechanism

### merged_beta_mhc

    scores = cat([scores_seq, scores_bank], dim=-1)
    weights = softmax(scores)
    # mHC: cap bank columns in merged weights
    weights = shield_attention_weights(weights, bank_size=N, kappa=kappa)
    out_seq = weights[:, :, :, :T] @ V_seq
    out_bank = weights[:, :, :, T:] @ (alpha * M_V)
    O = out_seq + beta * out_bank

### sep_beta_mhc

    w_seq = softmax(scores_seq)
    w_bank = softmax(scores_bank)
    # mHC: cap bank columns in standalone bank weights
    w_bank = shield_bank_weights(w_bank, kappa=kappa)
    out_seq = w_seq @ V_seq
    out_bank = w_bank @ (alpha * M_V)
    O = out_seq + beta * out_bank

### Legacy reference

    effective beta ≈ alpha_scale * sigmoid(gate_bias) ≈ 0.2 * sigmoid(-1) ≈ 0.054
    → beta=0.05 is the most important test point.

---

## Fixed Settings

| Parameter | Value |
|-----------|-------|
| model | Qwen3-4B-Instruct-2507 |
| bank_key_mode | pre_rope |
| bank_size | 200 |
| value_scale_mode | auto_rms_cap |
| dtype | bf16 |
| attention_impl | eager |
| device | cuda |
| seeds | 0, 1, 2 |
| primary alpha | 0.05 |
| kappa | 0.25 (Exp8 Phase A best) |
| dataset | CounterFact-1k W.6 filter, 807 eligible |

---

## Variants

| Variant | M_K | M_V | Description |
|---------|-----|-----|-------------|
| correct_bank | correct | correct | Full correct K/V binding |
| shuffled_bank | correct | shuffled V rows | Correct K routing, wrong V content |
| random_kv | random | random | Both K and V random (RMS-matched) |
| correct_K_random_V | correct | random | Correct K routing, random V |
| random_K_correct_V | random | correct | Wrong K routing, correct V |

---

## Modes

| Mode | bank_separate_softmax | mhc_shield | bank_merge_beta |
|------|----------------------|------------|-----------------|
| merged_beta_mhc | False | True | beta |
| sep_beta_mhc | True | True | beta |

Beta grid: 0.05, 0.10, 0.20, 0.50, 1.00

---

## Hypotheses

**H9.1**: For merged_beta_mhc, correct_bank mean_margin will be higher than for at least one
control variant at the optimal beta (vs Exp8 kappa=0.25 where all controls beat correct_bank).

**H9.2**: The gap score (correct_bank - max_control) will be higher under sep_beta_mhc than
merged_beta_mhc at the same beta, because separate-softmax eliminates softmax dilution of
the bank signal.

**H9.3**: For beta near the legacy value (0.05), JS/KL drift will be lower than Exp8 mHC-only
(kappa=0.25 baseline) because the residual gate naturally limits injection magnitude.

**H9.4**: Under high-alpha stress (Phase C), beta+mHC will degrade more gracefully than
Exp8 mHC-only: correct_bank selectivity will not cliff as sharply.

---

## Phase Structure

### Phase A1 — Beta Grid Smoke

- n_prompts: 100
- seeds: 0, 1, 2
- variants: correct_bank, random_kv, random_K_correct_V (key controls)
- modes: merged_beta_mhc, sep_beta_mhc
- betas: 0.05, 0.10, 0.20, 0.50, 1.00
- cells: 100 × 3 × 5 × 2 × 3 = 9000

**A1 goal:** Find whether any beta reduces random_kv / random_K_correct_V margins.
Selection metric: gap_A1 = correct_bank_margin - max(random_kv, random_K_correct_V) margins.
Select top 2 (mode, beta) configurations.

### Phase A2 — Full Controls Smoke

- n_prompts: 100
- seeds: 0, 1, 2
- variants: all 5
- configs: top 2 from A1
- cells: 100 × 3 × 5 × 2 = 3000

**A2 goal:** Confirm no other control (shuffled, correct_K_random_V) becomes the new worst case.
Selection metric: gap_A2 = correct_bank_margin - max_all_controls.
Select 1 best (mode, beta) for Phase B.

### Phase B — Full Validation

- n_prompts: 807 (all eligible)
- seeds: 0, 1, 2
- variants: all 5
- config: best from A2
- cells: 807 × 3 × 5 = 12105

**B goal:** Formal statistical test with bootstrap 95% CI.

### Phase C — High-Alpha Stress

- n_prompts: 200
- seeds: 0, 1, 2
- alphas: 0.10, 0.20, 0.50, 1.00
- config: best from A2 (mode + beta fixed, kappa=0.25)
- variants: correct_bank, random_kv, random_K_correct_V
- cells: 200 × 3 × 4 × 3 = 7200

**C goal:** Determine if beta+mHC prevents high-alpha runaway seen in Exp7/Exp8.

---

## Primary Score

    gap = mean_margin(correct_bank) - max(
        mean_margin(shuffled_bank),
        mean_margin(random_kv),
        mean_margin(correct_K_random_V),
        mean_margin(random_K_correct_V),
    )

Exp8 best gap (kappa=0.25, no beta): **-0.159** — Exp9 must improve on this.

---

## Metrics Recorded

Primary:
- variant, mode, beta, mhc_shield, mhc_kappa, bank_separate_softmax, alpha, seed, prompt_id
- target_new_logprob, target_true_logprob, margin
- recall_at_1, target_rank
- js_drift, kl_drift
- bank_attention_mass, max_bank_prob

Secondary:
- o_bank_norm, o_seq_norm, obank_oseq_ratio (sep branch)
- pre_mhc_bank_col_sum_max, post_mhc_bank_col_sum_max

---

## Phase B Verdict Criteria

**PASS_STRONG**: gap > 0 AND correct_bank 95CI_low > max_control 95CI_high AND drift ≤ Exp8 drift

**PASS_DIRECTIONAL**: gap > 0 AND CI overlapping

**STABILIZER_ONLY**: gap ≤ 0 AND random controls margins drop relative to Exp8 AND drift drops

**FAIL**: gap ≤ 0 AND random controls remain highest AND drift unchanged

---

## Phase C Verdict Criteria

**HIGH_ALPHA_SMOOTH**: correct_bank margin degrades more slowly with alpha vs Exp8 mHC-only

**HIGH_ALPHA_FAIL**: beta+mHC still shows runaway or abrupt collapse

---

## Failure Interpretation

If FAIL:
- Beta gate does not transfer legacy residual smoothing through ATB injection
- Remaining bottleneck is row-level K retrieval, not injection envelope control
- Next step: K-projector / InfoNCE, bank_topk, target-slot diagnostics

If STABILIZER_ONLY:
- Beta gate reduces V-path dominance but K routing remains insufficient for correct-bank selectivity
- Next step: improve K retrieval mechanism

If PASS:
- Conclusion: V-scale + mHC + beta gate form the stable injection envelope analogous to legacy delta-QKV
- Paper claim: ATB requires an output-gated safe preset for large-bank non-Gemma factual memory

---

## Benchmark Comparison Table

| Experiment | beta | mHC | kappa | correct | max_ctrl | gap | verdict |
|-----------|------|-----|-------|---------|----------|-----|---------|
| Exp7 raw ATB | 1.0 | off | - | -0.537 | -0.249 | -0.288 | FAIL |
| Exp8 kappa=0.25 | 1.0 | on | 0.25 | TBD | TBD | -0.159 | FAIL |
| Exp9 best | grid | on | 0.25 | TBD | TBD | TBD | TBD |

---

## Files

- `PREREG.md` — this file
- `run.py` — phase dispatcher (A1 → A2 → B → C, fully sequential)
- `analyze.py` — post-hoc analysis and plot generation
- `post_process.sh` — rsync + analyze + commit helper

---

*Pre-registered before first result collected. Analysis code committed before data collected.*
