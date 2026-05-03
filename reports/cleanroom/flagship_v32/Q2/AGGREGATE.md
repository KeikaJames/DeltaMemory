# Phase Q2 Aggregate — mHC Shield α-Safety Sweep

**Models**: 4  |  **Date**: 2026-05-04

## Qwen/Qwen3-4B-Instruct-2507

| α | Shield OFF lift | Shield OFF drift | Shield ON lift | Shield ON drift |
|---:|---:|---:|---:|---:|
| 0.05 | -0.493 | +7.597 | +0.220 | +7.432 |
| 0.10 | -1.090 | +5.673 | -0.970 | +7.705 |
| 0.50 | -5.873 | +10.032 | +1.785 | +7.629 |
| 1.00 | -6.986 | +10.590 | -4.873 | +9.145 |
| 2.00 | -6.666 | +11.064 | -7.377 | +11.193 |
| 5.00 | -1.339 | +12.278 | -7.031 | +11.474 |
| 10.00 | -1.047 | +13.470 | -3.734 | +11.731 |

## THUDM/GLM-4-9B-0414

| α | Shield OFF lift | Shield OFF drift | Shield ON lift | Shield ON drift |
|---:|---:|---:|---:|---:|
| 0.05 | -0.582 | +2.296 | -3.784 | +2.478 |
| 0.10 | -0.171 | +2.168 | -2.068 | +2.466 |
| 0.50 | +1.361 | +1.373 | -0.693 | +2.304 |
| 1.00 | +2.169 | +1.623 | -0.892 | +2.096 |
| 2.00 | +2.310 | +2.323 | -0.132 | +1.732 |
| 5.00 | +1.779 | +3.780 | +3.117 | +0.380 |
| 10.00 | +2.409 | +5.768 | +1.791 | +0.703 |

## deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

| α | Shield OFF lift | Shield OFF drift | Shield ON lift | Shield ON drift |
|---:|---:|---:|---:|---:|
| 0.05 | -0.143 | +1.489 | -0.610 | +0.298 |
| 0.10 | +2.126 | +3.979 | -0.533 | +0.588 |
| 0.50 | -0.725 | +5.576 | +2.063 | +1.565 |
| 1.00 | -0.941 | +6.590 | +1.243 | +0.977 |
| 2.00 | -2.753 | +8.840 | +0.177 | +3.777 |
| 5.00 | -3.305 | +11.358 | -1.595 | +7.770 |
| 10.00 | -4.123 | +12.086 | -3.697 | +8.770 |

## google/gemma-4-E2B

| α | Shield OFF lift | Shield OFF drift | Shield ON lift | Shield ON drift |
|---:|---:|---:|---:|---:|
| 0.05 | +0.201 | +0.329 | +0.108 | +0.268 |
| 0.10 | +0.127 | +0.239 | +0.113 | +0.351 |
| 0.50 | +0.510 | -0.032 | +0.263 | +0.025 |
| 1.00 | +1.305 | -0.062 | +0.556 | +0.022 |
| 2.00 | +4.576 | -0.069 | +1.746 | -0.030 |
| 5.00 | +1.916 | +0.205 | +4.952 | -0.051 |
| 10.00 | +0.375 | +1.260 | +2.787 | +0.166 |

## Hypothesis Verdicts

### H1: shield ON drift ≤ 0.5 nats

- **Qwen/Qwen3-4B-Instruct-2507**: ❌ FAIL  (0/7 α pass, max drift=11.731)
- **THUDM/GLM-4-9B-0414**: ❌ FAIL  (1/7 α pass, max drift=2.478)
- **deepseek-ai/DeepSeek-R1-Distill-Qwen-32B**: ❌ FAIL  (1/7 α pass, max drift=8.770)
- **google/gemma-4-E2B**: ✅ PASS  (7/7 α pass, max drift=0.351)

### H2: shield ON lift > 0

- **Qwen/Qwen3-4B-Instruct-2507**: ❌ FAIL  (2/7 α pass)
- **THUDM/GLM-4-9B-0414**: ❌ FAIL  (2/7 α pass)
- **deepseek-ai/DeepSeek-R1-Distill-Qwen-32B**: ❌ FAIL  (3/7 α pass)
- **google/gemma-4-E2B**: ✅ PASS  (7/7 α pass)

### Wilcoxon Signed-Rank: shield ON vs OFF drift

- alpha_0.05: p=0.4652  (no significant difference)
- alpha_0.1: p=0.7150  (no significant difference)
- alpha_0.5: p=0.4652  (no significant difference)
- alpha_1.0: p=0.4652  (no significant difference)
- alpha_10.0: p=0.0679  (no significant difference)
- alpha_2.0: p=0.4652  (no significant difference)
- alpha_5.0: p=0.0679  (no significant difference)

