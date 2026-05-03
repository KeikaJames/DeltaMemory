# Phase Q2 Aggregate — mHC Shield α-Safety Sweep

**Models**: 1  |  **Date**: 2026-05-04

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

- **google/gemma-4-E2B**: ✅ PASS  (7/7 α pass, max drift=0.355)

### H2: shield ON lift > 0

- **google/gemma-4-E2B**: ✅ PASS  (7/7 α pass)

### Wilcoxon Signed-Rank: shield ON vs OFF drift


