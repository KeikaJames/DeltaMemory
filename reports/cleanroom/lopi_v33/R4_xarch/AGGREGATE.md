# R-4 cross-arch LOPI sweep — paired aggregate

Each row = one (model, α) cell.  Columns = (shield, lopi) configurations.
Lift = mean log-prob lift on FALSE_FACTS (higher = bank pushes counter-prior);
drift = mean per-token NLL change on NEUTRAL_PROMPTS (lower magnitude = safer).

## Qwen/Qwen3-4B-Instruct-2507

| α | sh=OFF lopi=OFF lift / drift | sh=OFF lopi=ON lift / drift | sh=ON  lopi=OFF lift / drift | sh=ON  lopi=ON lift / drift |
|---:|---:|---:|---:|---:|
| 0.0 | +0.000 / +0.000 | +0.000 / +0.000 | +0.000 / +0.000 | +0.000 / +0.000 |
| 0.05 | -0.031 / +7.626 | -0.144 / +3.700 | +0.115 / +7.362 | +0.834 / +3.619 |
| 0.1 | -1.071 / +5.515 | -0.761 / +3.697 | -1.282 / +7.640 | -0.531 / +3.687 |
| 0.5 | -6.084 / +10.123 | -0.337 / +5.720 | +1.498 / +7.495 | -1.761 / +4.501 |
| 1.0 | -6.317 / +10.640 | -2.685 / +5.827 | -4.391 / +9.025 | -0.537 / +5.130 |
| 2.0 | -6.777 / +11.075 | -3.185 / +5.971 | -6.994 / +11.141 | -2.402 / +5.141 |
| 5.0 | -1.250 / +12.317 | -0.969 / +8.435 | -7.151 / +11.431 | -4.394 / +8.221 |

### Qwen/Qwen3-4B-Instruct-2507 — drift ratio LOPI/no-LOPI (paired by shield, α)

| α | shield=OFF | shield=ON |
|---:|---:|---:|
| 0.05 | +0.49 | +0.49 |
| 0.1 | +0.67 | +0.48 |
| 0.5 | +0.57 | +0.60 |
| 1.0 | +0.55 | +0.57 |
| 2.0 | +0.54 | +0.46 |
| 5.0 | +0.68 | +0.72 |

## THUDM/GLM-4-9B-0414

| α | sh=OFF lopi=OFF lift / drift | sh=OFF lopi=ON lift / drift | sh=ON  lopi=OFF lift / drift | sh=ON  lopi=ON lift / drift |
|---:|---:|---:|---:|---:|
| 0.0 | +0.000 / +0.000 | +0.000 / +0.000 | +0.000 / +0.000 | +0.000 / +0.000 |
| 0.05 | -0.918 / +2.248 | -1.873 / +2.276 | -3.323 / +2.493 | -1.442 / +2.445 |
| 0.5 | +1.343 / +1.361 | +0.114 / +1.503 | -1.068 / +2.334 | -1.881 / +2.162 |
| 1.0 | +2.252 / +1.620 | +2.319 / +1.146 | -0.940 / +2.160 | -1.906 / +2.086 |
| 2.0 | +2.276 / +2.321 | +3.030 / +1.040 | -0.321 / +1.783 | -0.108 / +1.794 |

### THUDM/GLM-4-9B-0414 — drift ratio LOPI/no-LOPI (paired by shield, α)

| α | shield=OFF | shield=ON |
|---:|---:|---:|
| 0.05 | +1.01 | +0.98 |
| 0.5 | +1.10 | +0.93 |
| 1.0 | +0.71 | +0.97 |
| 2.0 | +0.45 | +1.01 |

## google/gemma-4-E2B

| α | sh=OFF lopi=OFF lift / drift | sh=OFF lopi=ON lift / drift | sh=ON  lopi=OFF lift / drift | sh=ON  lopi=ON lift / drift |
|---:|---:|---:|---:|---:|
| 0.0 | +0.000 / +0.000 | +0.000 / +0.000 | +0.000 / +0.000 | +0.000 / +0.000 |
| 0.5 | +0.515 / -0.028 | +0.373 / +0.081 | +0.255 / +0.024 | +0.269 / +0.148 |
| 1.0 | +1.400 / -0.062 | +0.720 / -0.066 | +0.510 / +0.011 | +0.423 / +0.017 |
| 2.0 | +4.620 / -0.069 | +2.245 / -0.081 | +1.680 / -0.027 | +0.810 / -0.030 |
| 5.0 | +1.955 / +0.192 | +4.566 / -0.016 | +4.931 / -0.057 | +3.669 / +0.004 |
| 10.0 | +0.579 / +1.260 | +3.736 / +0.226 | +2.839 / +0.165 | +4.964 / +0.151 |

### google/gemma-4-E2B — drift ratio LOPI/no-LOPI (paired by shield, α)

| α | shield=OFF | shield=ON |
|---:|---:|---:|
| 0.5 | -2.93 | +6.20 |
| 1.0 | +1.07 | +1.60 |
| 2.0 | +1.17 | +1.14 |
| 5.0 | -0.08 | -0.07 |
| 10.0 | +0.18 | +0.92 |

## L2 verdict — per-model drift reduction (LOPI ON vs OFF, paired)

| model | n_pairs | n_LOPI_reduces_drift | mean_drift_LOPI_OFF | mean_drift_LOPI_ON | abs_reduction_pp | L2_strict (≤0.5 nats LOPI ON) |
|---|---:|---:|---:|---:|---:|:---:|
| Qwen/Qwen3-4B-Instruct-2507 | 12 | 12 | 9.283 | 5.304 | +3.979 | ❌ |
| THUDM/GLM-4-9B-0414 | 8 | 5 | 2.040 | 1.806 | +0.234 | ❌ |
| google/gemma-4-E2B | 10 | 4 | 0.189 | 0.082 | +0.107 | ✅ |
