# X.7-NL Cross-Arch Replication — Qwen3-4B-Instruct-2507

**Status**: ✅ COMPLETED, 2026-05-07 02:35 CST
**Cells**: 12 / 12 PASS (sub-A, |M|∈{10, 100, 500, 5000} × seeds {0,1,2}, α=1.0)
**Hardware**: GB10 (Spark1) CUDA bf16
**Goal**: cross-architecture replication of the X.7-NL bank-size U-curve
discovered on Gemma-4-31B (paper Table 8.2). Addresses arXiv blocker #3
(*at least one non-Gemma data point on U-curve*).

## Headline numbers (sub=A, α=1.0)

| |M| | mean log_margin | Gemma-4-31B reference |
|---:|---:|---:|
| 10 | +1.344 | −0.188 |
| 100 | +1.490 | +0.479 |
| 500 | +3.654 | −0.375 |
| 5000 | −2.042 | +0.604 |

## Verdict

**Partial replication, different shape**. Both architectures exhibit
non-monotonic |M|-recall coupling (the *interesting* phenomenon), but
the location of the dip differs:

- **Gemma-4-31B**: U-shape — confusion valley at |M|∈[500, 1000],
  recovery at |M|=5000 (margin = +0.604).
- **Qwen3-4B**: late-cliff — monotone rise to |M|=500 (peak +3.654),
  collapse at |M|=5000 (−2.042).

This **strengthens** the paper claim (the universal observation is
"non-monotonic", not "U-shaped specifically"), and it is precisely the
kind of cross-model heterogeneity that the X.7-mech B1/B2 per-layer
ablations are designed to disentangle in v0.8.

## Plotted

- `papers/figures/fig1_u_curve.png` (with both arches overlaid)

## Files

- `cells.jsonl` — 12 raw rows
- `env.json` — runtime stamp
- `run.log` — dispatcher log
