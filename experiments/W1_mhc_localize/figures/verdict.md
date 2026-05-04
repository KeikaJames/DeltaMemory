# W.1 Verdict
**Overall verdict: FAIL**  
PASS threshold: mean_drift(shield=on, V-scale=on, α≥1) ≤ 0.5 nats  
Models passing: 0/2

## Per-model results

| Model | Status | mean_drift (shield+V-scale, α≥1) |
|-------|--------|-----------------------------------|
| Qwen/Qwen2.5-0.5B | **FAIL** | 3.3134 |
| Qwen/Qwen2.5-1.5B | **FAIL** | 4.3912 |

## DH1: Shield truncates col-sums?

bank_col_sum_p99 shield=ON: 559.8123  
bank_col_sum_p99 shield=OFF: 561.9774  
→ DH1 **CONFIRMED**: shield reduces col-sum p99

