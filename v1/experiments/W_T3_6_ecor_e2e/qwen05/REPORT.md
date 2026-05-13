# W-T3 round 2 — end-to-end ECOR vs additive

**Model**: `Qwen/Qwen2.5-0.5B-Instruct`

**Device/dtype**: `mps` / `bfloat16`

**α grid**: [0.0, 0.5, 1.0, 2.0, 4.0] | **Seeds**: [0, 1, 2] | **Prompts**: 8

**Baseline NLL** (no bank): +1.9035

**Total cells**: 60 | **Wall**: 24.0s

**Generated**: 2026-05-04T16:56:36Z


## Drift vs baseline (lower is better)

| α | add | add_ortho | ecor_blend50 | ecor_pure | best |
|--:|--:|--:|--:|--:|--:|
| 0.00 | +0.000 | +0.000 | +0.000 | +0.000 | **add** |
| 0.50 | +2.331 | +2.387 | +2.783 | +2.711 | **add** |
| 1.00 | +2.256 | +2.308 | +2.374 | +2.674 | **add** |
| 2.00 | +3.142 | +2.979 | +2.587 | +2.681 | **ecor_blend50** |
| 4.00 | +4.426 | +4.514 | +2.919 | +2.678 | **ecor_pure** |

## Cells

- `cells.jsonl` (60 cells)
