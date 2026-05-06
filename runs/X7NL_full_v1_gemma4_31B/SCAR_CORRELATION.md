# X.7-NL/D — SCAR Signal Correlation (gemma-4-31B-it)

## Method

Post-hoc correlation analysis on `runs/X7NL_full_v1_gemma4_31B/cells.jsonl`
(N=291). SCAR proxies (since explicit proj/ortho/alpha-drift were not
logged in this run):

- **proj**         ≈ `bank_col_mean` (mean attention mass on bank columns)
- **ortho**        ≈ `attn_entropy_bank_mean − attn_entropy_native_mean`
- **alpha_drift**  ≈ `residual_norm_mean − residual_norm_mean(α=0)` per (sub,seed,bank_size)

Outcome: `log_margin = score_new − score_canonical`.

## Overall correlations

| proxy | n | pearson | spearman |
| --- | ---: | ---: | ---: |
| proj | 288 | +0.291 | +0.747 |
| ortho | 291 | -0.513 | -0.805 |
| alpha_drift | 291 | +0.360 | +0.459 |

## Per-sub correlations

### sub = A

| proxy | n | pearson | spearman |
| --- | ---: | ---: | ---: |
| proj | 18 | -0.197 | -0.102 |
| ortho | 18 | +0.145 | +0.108 |
| alpha_drift | 18 | +nan | +nan |

### sub = B

| proxy | n | pearson | spearman |
| --- | ---: | ---: | ---: |
| proj | 120 | -0.590 | -0.555 |
| ortho | 123 | -0.187 | -0.535 |
| alpha_drift | 123 | +0.116 | -0.064 |

### sub = C

| proxy | n | pearson | spearman |
| --- | ---: | ---: | ---: |
| proj | 150 | +0.132 | +0.725 |
| ortho | 150 | -0.484 | -0.725 |
| alpha_drift | 150 | +0.659 | +0.739 |

## Caveats

- Proxies are *correlated with* but not identical to the prereg'd
  SCAR signals. A future X.7-NL re-run should record proj/ortho/alpha-drift
  directly via `deltamemory.memory.scar_injector` hooks.
- alpha_drift uses a within-cohort baseline; cohorts without an α=0
  reference (sub A bank-scaling fixes α=1.0) fall back to cohort mean.
- |Pearson| > 0.3 with N≥30 is treated as *suggestive*; |Pearson| > 0.5
  is *strong*. We do not infer causation.
