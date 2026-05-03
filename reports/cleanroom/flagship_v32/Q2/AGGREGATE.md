# Phase Q2 Aggregate — mHC Shield α-Safety Sweep

**Models**: 4  |  **Date**: 2026-05-04

## Qwen/Qwen3-4B-Instruct-2507

| α | Shield OFF lift | Shield OFF drift | Shield ON lift | Shield ON drift |
|---:|---:|---:|---:|---:|
| 0.05 | +0.871 | +3.565 | +0.727 | +3.557 |
| 0.10 | +1.130 | +3.467 | +0.898 | +3.516 |
| 0.50 | -0.604 | +3.444 | +0.991 | +3.664 |
| 1.00 | -4.646 | +8.252 | +0.784 | +3.680 |
| 2.00 | -5.336 | +8.904 | -5.727 | +6.784 |
| 5.00 | -1.314 | +8.515 | -6.130 | +7.876 |
| 10.00 | -0.131 | +11.618 | -8.936 | +9.177 |

## THUDM/GLM-4-9B-0414

| α | Shield OFF lift | Shield OFF drift | Shield ON lift | Shield ON drift |
|---:|---:|---:|---:|---:|
| 0.05 | -1.893 | +2.179 | -2.613 | +2.357 |
| 0.10 | -1.538 | +2.138 | -3.253 | +2.517 |
| 0.50 | +0.234 | +1.653 | -0.650 | +2.341 |
| 1.00 | +1.501 | +1.554 | +0.910 | +2.165 |
| 2.00 | +2.127 | +1.579 | -0.342 | +1.982 |
| 5.00 | +2.038 | +2.468 | +1.991 | +0.915 |
| 10.00 | +2.871 | +3.931 | +2.581 | +0.270 |

## deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

| α | Shield OFF lift | Shield OFF drift | Shield ON lift | Shield ON drift |
|---:|---:|---:|---:|---:|
| 0.05 | -0.592 | +0.305 | -0.612 | +0.429 |
| 0.10 | -0.474 | +0.351 | -0.718 | +0.331 |
| 0.50 | +0.174 | +3.464 | -0.664 | +0.341 |
| 1.00 | +3.536 | +2.089 | +0.155 | +0.307 |
| 2.00 | -0.874 | +4.758 | +0.737 | +1.746 |
| 5.00 | -1.958 | +8.789 | +1.616 | +2.001 |
| 10.00 | -2.646 | +9.199 | +0.603 | +1.737 |

## google/gemma-4-E2B

| α | Shield OFF lift | Shield OFF drift | Shield ON lift | Shield ON drift |
|---:|---:|---:|---:|---:|
| 0.05 | +0.170 | +0.334 | +0.122 | +0.259 |
| 0.10 | +0.212 | +0.256 | +0.120 | +0.355 |
| 0.50 | +0.515 | -0.028 | +0.255 | +0.024 |
| 1.00 | +1.400 | -0.062 | +0.510 | +0.011 |
| 2.00 | +4.620 | -0.069 | +1.680 | -0.027 |
| 5.00 | +1.955 | +0.192 | +4.931 | -0.057 |
| 10.00 | +0.579 | +1.260 | +2.839 | +0.165 |

## Hypothesis Verdicts

### H1: shield ON drift ≤ 0.5 nats

- **Qwen/Qwen3-4B-Instruct-2507**: ❌ FAIL  (0/7 α pass, max drift=9.177)
- **THUDM/GLM-4-9B-0414**: ❌ FAIL  (1/7 α pass, max drift=2.517)
- **deepseek-ai/DeepSeek-R1-Distill-Qwen-32B**: ❌ FAIL  (4/7 α pass, max drift=2.001)
- **google/gemma-4-E2B**: ✅ PASS  (7/7 α pass, max drift=0.355)

### H2: shield ON lift > 0

- **Qwen/Qwen3-4B-Instruct-2507**: ❌ FAIL  (4/7 α pass)
- **THUDM/GLM-4-9B-0414**: ❌ FAIL  (3/7 α pass)
- **deepseek-ai/DeepSeek-R1-Distill-Qwen-32B**: ❌ FAIL  (4/7 α pass)
- **google/gemma-4-E2B**: ✅ PASS  (7/7 α pass)

### Wilcoxon Signed-Rank: shield ON vs OFF drift

- alpha_0.05: p=0.4652  (no significant difference)
- alpha_0.1: p=0.1441  (no significant difference)
- alpha_0.5: p=0.7150  (no significant difference)
- alpha_1.0: p=0.4652  (no significant difference)
- alpha_10.0: p=0.0679  (no significant difference)
- alpha_2.0: p=0.4652  (no significant difference)
- alpha_5.0: p=0.0679  (no significant difference)

