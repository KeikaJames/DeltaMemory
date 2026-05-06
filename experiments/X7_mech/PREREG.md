# Pre-registration: X7_mech — Mechanistic Deep Probe (Track B)

**Version**: X7MECH.v1  
**Registered**: 2025-01-01 (before any runs)  
**Experiment IDs**: B1, B2, B3  
**Builds on**: X7NL.v1 verdict (`runs/X7NL_full_v1_gemma4_31B/REPORT.md`)

---

## 0. Purpose

X.7-NL found two catastrophic non-linearities on gemma-4-31B-it:
1. **U-shape bank scaling**: recall collapses at |bank|=500, rescues at 5000.
2. **α=0.25 cliff**: catastrophic recall loss before recovery at α≥0.75.

This pre-registration specifies hypotheses and analyses to source-locate
those non-linearities to specific layers and attention mechanisms **before
any data is collected**.

---

## 1. B1 — Per-layer Attention Probe

### Hypotheses

**H_B1.1 (Phase-transition layer)**: There exists a single layer L* such
that the transition from "bank-mass increases with |bank|" to "bank-mass
decreases with |bank|" first occurs. This layer is the phase-transition point.

**H_B1.2 (Early-layer dominance)**: The bank-attention mass is concentrated
in early-to-mid layers (L* < num_layers/2).

**H_B1.3 (Quasi-top-k at L*)**: At |bank|=5000, bank-column entropy **falls**
relative to |bank|=500 at layer L* (consistent with quasi-top-k selection),
while remaining monotone in other layers.

**H_B1.4 (Residual contribution tracks log_margin)**: The residual contribution
of bank-induced delta (‖delta‖/‖residual‖) correlates positively with
log_margin across bank sizes and seeds.

### Operationalisation

- Bank sizes: {100, 500, 5000}
- Seeds: {0, 1, 2}
- Per-layer metrics: `bank_mass`, `bank_entropy`, `top1/3/10_concentration`,
  `residual_ratio`
- Phase-transition layer identified as: argmax over L of
  |bank_mass(L, 100) - bank_mass(L, 500)| + |bank_mass(L, 5000) - bank_mass(L, 500)|

### Primary test

Kruskal-Wallis across layers for (mass difference: 5000 vs 500) ≠ 0.
Phase-transition layer: layer with highest absolute difference.

---

## 2. B2 — Sparsity at |bank|=5000

### Hypotheses

**H_B2.1 (Quasi-top-k regime)**: At |bank|=5000, the top-10 bank columns
capture ≥80% of bank attention mass, while at |bank|=500 top-10 captures <80%.

**H_B2.2 (Discontinuous jump)**: The top-k fraction is NOT monotone in |bank|
over {100, 500, 5000} — specifically the top-1 fraction exhibits a
non-monotone pattern consistent with the U-shape.

**H_B2.3 (Statistical significance)**: Paired Wilcoxon (3 seeds) on
top-10 fraction: 5000 vs 500 is significant at p<0.05.

### Operationalisation

- Bank sizes: {100, 200, 500, 1000, 5000}
- k values: {1, 5, 10, 20}
- Seeds: {0, 1, 2}
- Per-seed top-k fraction at each (bank_size, k) combination

---

## 3. B3 — α-cliff Residual Analysis

### Hypotheses

**H_B3.1 (Layer-specific threshold)**: There exists a layer L_cliff such
that ‖Δresidual(α=0.25) - Δresidual(α=0)‖ is maximised — the cliff
originates at a specific layer, not uniformly across all layers.

**H_B3.2 (Threshold crossing at α=0.25)**: The per-layer residual norm
‖residual(α=0.25)‖ / ‖residual(α=0)‖ exceeds 1.5× the pre-cliff ratio
at layer L_cliff, specifically at α=0.25 (not 0.20 or 0.30).

**H_B3.3 (Recovery mechanism)**: At α≥0.75, ‖Δresidual‖ at L_cliff
returns to within 20% of the α=0 baseline, correlated with log_margin
recovery.

**H_B3.4 (Monotone recovery post-cliff)**: For α ∈ {0.5, 0.75, 1.0},
‖Δresidual‖ at L_cliff is monotone decreasing.

### Operationalisation

- α values: {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.0}
- Seeds: {0, 1, 2}
- Per-layer per-α residual norm; Δresidual vs α=0 baseline

### Statistical test

For each layer L, compute mean Δresidual across seeds for each α.
Identify L_cliff = argmax_L Δresidual(α=0.25, L) / Δresidual(α=0, L).
Test: is ratio at α=0.25 significantly larger than at α=0.20?
(paired t-test or Wilcoxon across seeds)

---

## 4. Analysis plan (post-data)

All analyses will be conducted in `experiments/X7_mech/aggregate.py`.

The following are pre-specified:
- No post-hoc layer selection — L* and L_cliff must be identified by the
  criterion stated above, not cherry-picked.
- All tests Bonferroni-corrected for num_layers.
- Results reported as effect sizes (Cohen's d) alongside p-values.

---

## 5. Safe-α prescription (B4)

Based on B3 findings, implement a data-driven safe-α scheduler that:
- Uses per-layer residual norms as a real-time cliff detector
- Recommends α=0 or α≥0.5 (skipping 0.05–0.30)
- Validates on held-out seeds (seed=3 if possible)

The prescription is implemented **after** B3 data collection, using only
the pre-registered cliff criterion.
