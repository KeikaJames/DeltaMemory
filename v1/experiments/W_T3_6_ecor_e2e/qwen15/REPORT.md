# W-T3 round 2 — end-to-end ECOR vs additive

**Model**: `Qwen/Qwen2.5-1.5B`

**Device/dtype**: `mps` / `bfloat16`

**α grid**: [0.0, 0.5, 1.0, 2.0, 4.0] | **Seeds**: [0, 1, 2] | **Prompts**: 8

**Baseline NLL** (no bank): +1.5669

**Total cells**: 60 | **Wall**: 30.1s

**Generated**: 2026-05-04T16:57:26Z


## Drift vs baseline (lower is better)

| α | add | add_ortho | ecor_blend50 | ecor_pure | best |
|--:|--:|--:|--:|--:|--:|
| 0.00 | +0.000 | +0.000 | +0.000 | +0.000 | **add** |
| 0.50 | +1.968 | +1.688 | +1.806 | +1.859 | **add_ortho** |
| 1.00 | +2.148 | +2.067 | +1.666 | +2.072 | **ecor_blend50** |
| 2.00 | +2.172 | +2.087 | +1.895 | +2.022 | **ecor_blend50** |
| 4.00 | +1.692 | +2.078 | +1.764 | +2.105 | **add** |

## Cells

- `cells.jsonl` (60 cells)
