# A.3 Ablation Verdict

Paired Wilcoxon signed-rank test + bootstrap 95% CI on `nll_new` (ablation − control, paired by prompt_id × seed).

**Necessity claim**: SUPPORTED iff `p < 0.01` AND 95% CI excludes 0.

## Results

| arm | method | n_pairs | mean_delta | ci_low | ci_high | p_value | supported |
|-----|--------|--------:|-----------:|-------:|--------:|--------:|:---------:|
| A5 | caa | 20 | +1.1266 | +0.9256 | +1.3373 | 0.0000 | ✅ SUPPORTED |
| A3 | lopi_default | 20 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | ❌ NOT SUPPORTED |
| A6 | lopi_default | 20 | +0.0000 | +0.0000 | +0.0000 | 1.0000 | ❌ NOT SUPPORTED |

## Notes

- **A5** (CAA random steering): replaces target-mean vector with a seeded random unit vector.
- **A3** (LOPI η_σ=1): force eta_sigma=1 (disable σ-shrink) in `lopi_default`.
- **A6** (LOPI θ=0): force theta=0 in `lopi_default`.

## Interpretation

- **A5**: NECESSARY — removing this component degrades `nll_new` by +1.1266 nats on average (95% CI [+0.9256, +1.3373], p=0.0000).
- **A3**: NO-OP — ablation does not significantly affect `nll_new` (mean Δ=+0.0000, p=1.0000, CI includes 0).
- **A6**: NO-OP — ablation does not significantly affect `nll_new` (mean Δ=+0.0000, p=1.0000, CI includes 0).

## Data gaps

- `scar_arm` and `bank_arm` not available (methods not registered in dispatcher).
- A1, A2, A4 verdicts pending extension of run.py (see REPORT.md §Next actions).
