# Stage 13D — Per-query routing fix for P3 locality drift

**Status:** PASS
**Pass criterion:** drift ≤ 0.05 AND override ≥ 0.90

- Model: google/gemma-4-E2B (mps, bfloat16)
- N facts: 50, N controls: 50
- alpha: 50.0 (lm_head-row injection)

## Baseline (Stage 12 reproduction — no gating, broadcast mean(bank))

- locality_drift_rate: **0.220**
- override_rate_on_wrong: **1.000**
- base_top1 (no DM): 0.000
- DM_top1: 1.000

## Gated sweep

| tau | mode | drift | override | route_acc | cos_canon | cos_ctrl | α_eff_canon | α_eff_ctrl |
|-----|------|-------|----------|-----------|-----------|----------|-------------|------------|
| 0.30 | soft | 0.580 | 0.980 | 0.980 | 0.787 | 0.475 | 50.000 | 41.110 |
| 0.30 | hard | 0.640 | 0.980 | 0.980 | 0.787 | 0.475 | 50.000 | 42.000 |
| 0.50 | soft | 0.440 | 0.980 | 0.980 | 0.787 | 0.475 | 49.999 | 23.030 |
| 0.50 | hard | 0.420 | 0.980 | 0.980 | 0.787 | 0.475 | 50.000 | 23.000 |
| 0.60 | soft | 0.320 | 0.980 | 0.980 | 0.787 | 0.475 | 49.966 | 14.336 |
| 0.60 | hard | 0.280 | 0.980 | 0.980 | 0.787 | 0.475 | 50.000 | 14.000 |
| 0.70 | soft | 0.120 | 0.980 | 0.980 | 0.787 | 0.475 | 48.204 | 5.104 |
| 0.70 | hard | 0.080 | 0.980 | 0.980 | 0.787 | 0.475 | 50.000 | 4.000 |
| 0.72 | soft | 0.120 | 0.980 | 0.980 | 0.787 | 0.475 | 46.198 | 3.736 |
| 0.72 | hard | 0.060 | 0.980 | 0.980 | 0.787 | 0.475 | 50.000 | 3.000 |
| 0.75 | soft | 0.060 | 0.980 | 0.980 | 0.787 | 0.475 | 39.483 | 2.098 |
| 0.75 | hard | 0.040 | 0.980 | 0.980 | 0.787 | 0.475 | 50.000 | 2.000 |

## Best passing config

- tau=0.75, mode=hard
- drift=0.040, override=0.980

## Method

Bank keys k_i = mean-pool(input_embed(address_i)). Query keys
q = mean-pool(input_embed(prompt)). Per query, route to slot 
argmax_i cos(q, k_i); cos_max gates alpha. Bank vector for slot
i is lm_head.weight[value_token_id_i] so override is
deterministic (no Writer training needed) — this isolates the
routing-fix variable.