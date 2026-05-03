# Phase Q — mHC-DeltaMemory Flagship Verification (v3.2)

**Status**: Mac MPS single-model (Gemma-4-E2B) complete. Multi-model pending GB10 access.  
**Date**: 2026-05-04  
**Commit**: Pending

## Executive Summary

mHC shield V2 (bank-columns-only column cap) eliminates the V1 full-matrix
Sinkhorn-Knopp collapse on Gemma-4-E2B.  Shield ON at α=10.0 yields
simultaneously better counter-prior lift (+2.84 vs +0.58 nats) AND safer
NLL (+0.17 vs +1.26 nats) compared to shield OFF.  H1 and H2 both PASS
on the single-model pilot.

## Hypothesis Mapping

| ID | Hypothesis | Phase | Result | Detail |
|---|---|---|---|---|
| **H1** | shield ON drift ≤ 0.5 nats | Q2 | ✅ PASS | Max drift 0.355, 7/7 α pass |
| **H2** | shield ON lift > 0 | Q2 | ✅ PASS | 7/7 α positive lift, mean +1.54 |
| **H3** | implant ≥ 60% on ≥ 3/5 models | Q3 | 🔄 Partial | Logprob lift confirmed; generation needs higher α/training |
| **H4** | α=0 bit-equal 5/5 models | Q1 | ✅ 1/5 | Gemma-4-E2B PASS (0.0 diff); 4 models need GB10 |
| **H5** | neutral coherence drop ≤ 5% | Q3 | 🔄 Pending | Q2 neutral drift low; generation quality at high α TBD |

## Key Findings

### 1. V2 Shield Eliminates V1 Collapse

V1 (full-matrix Sinkhorn-Knopp): lift=−7.57, drift=+5.26 at α=1.0 → **catastrophic collapse**  
V2 (column cap): lift=+0.51, drift=+0.01 at α=1.0 → **stable and injecting**

### 2. Shield Amplifies Lift at High α

At α=10.0: shield ON provides 4.90× more lift than shield OFF while keeping
drift 7.6× lower.  The column cap prevents softmax saturation that drowns
the bank signal.

### 3. Counter-Prior Generation is the Next Frontier

Logprob lift is cleanly positive (4-5 facts at α≥1.0), but flipping the
argmax for generation requires higher α (5-10) where the shield is
essential OR a trained K-projector to close the prior gap.

## Artifacts

| Phase | Location | Status |
|---|---|---|
| Q0 Preregistration | `PREREGISTRATION.md` | ✅ |
| Q1 Smoke | Gemma-4-E2B α=0 bit-equal + shield ON | ✅ 1/5 |
| Q2 Sweep | `Q2/REPORT.md` + `AGGREGATE.json` + SVG | ✅ 1/5 |
| Q3 Chat | `Q3/REPORT.md` + logprob data | ✅ Pilot |
| Q4 Stats | `Q2/AGGREGATE.json` (bootstrap CI + Wilcoxon) | ✅ 1/5 |
| Q5 Final | This document | ✅ |

## Next Steps (requires GB10)

1. Q1 multi-model smoke: Qwen3-4B, DeepSeek-32B, GLM-4-9B, Gemma-4-31B
2. Q2 multi-model sweep: 4 models × 7 α × 2 shield × 3 seeds
3. Q3 full: 60 facts × 5 subjects × α-sweep + Gemma-4-31B judge
4. 5-gram contamination check on C4/RedPajama proxy
5. Final REPORT.md + 5 figures + PR update
