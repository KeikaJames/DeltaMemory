# W.1 Pre-Registration: mHC Shield Failure Localization

**Registered**: before any data collection  
**Paradigm**: 控制变量 + 消融 (controlled ablation sweep)  
**Plan ref**: v0.4 Phase W.1

---

## 1. Research Questions

| ID | Question |
|----|----------|
| Q1 | Does the mHC shield actually truncate bank column sums (col-sum P99 < κ)? |
| Q2 | Does shielding change attention entropy over bank slots (attn_entropy_bank)? |
| Q3 | What is the interaction between shield and V-scale on mean_drift? |
| Q4 | Does residual norm explode at large α, and does shield prevent it? |

## 2. Experimental Grid

| Variable | Levels |
|----------|--------|
| model | gpt2, gpt2-medium, Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, meta-llama/Llama-3.2-1B |
| alpha (α) | 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0 |
| shield | off, on |
| kappa (κ) | 1.0, 2.0, 4.0 (ignored when shield=off) |
| seed | 0 (single-seed pilot; multi-seed deferred to W-T1.1 as estimate >24h) |
| V-scale | off (value_scale_mode="none"), on (value_scale_mode="auto_rms_cap") |
| gold prompts | 30 from experiments/datasets/gold_30prompts.jsonl |

**Total cells (single-seed)**: 5 × 7 × 2 × 3 × 1 × 2 = 420  
(With 3 seeds would be 1260; shrinkage applied because 3-seed estimate ~28h > 24h threshold)

**Bank**: N=32 K/V entries, random tensors seeded via SHA-256("wikitext-bank-N32-seed{seed}"),
consistent across models by normalizing shapes to match each model's architecture.

## 3. Per-Cell Measurements

- `mean_drift` = mean(NLL_bank_α − NLL_nobank) over 30 prompts (nats; lower=better; <0=bank helps)
- `mean_lift` (α=0 sanity): must be ~0.0 (red-line: |mean_lift| < 0.01 nats)
- `bank_col_sum_p99`: 99th percentile of per-slot col-sums across layers×slots×batch
- `attn_entropy_bank`: mean Shannon entropy of bank attention weights across layers
- `m_perp_energy_ratio`: always 0 in W.1 (LOPI disabled)
- `residual_norm_p50`: median residual L2 norm across layers and tokens

## 4. Acceptance Criteria

### Per-model PASS condition
Model M **PASSES** if: `mean_drift(shield=on, V-scale=on, α≥1) ≤ 0.5 nats` (mean across all α∈{1,2,4,8,16}).

### Experiment-level Verdict
- **PASS**: ≥ 5/5 models PASS
- **PARTIAL**: ≥ 3/5 models PASS
- **FAIL**: < 3/5 models PASS

### Shield sanity (DH1)
`bank_col_sum_p99(shield=on) ≤ κ` for all cells with shield=on.

### α=0 red-line
`|mean_lift(α=0)| < 0.05 nats` for all cells (bank inactive, drift must be negligible).

## 5. What We Are NOT Testing in W.1

- LOPI (disabled throughout; m_perp_energy_ratio recorded but expected 0)
- MoE architectures (deferred to W.5)
- Long context (deferred to W.7)
- Human eval (deferred to W.15)

## 6. Pre-registered Hypotheses

| ID | Hypothesis | Expected direction |
|----|-----------|-------------------|
| DH1 | Shield truncates col-sums to ≤ κ | col_sum_p99 drops when shield=on |
| DH2 | Shield alone insufficient for no-v_norm models; V-scale required | drift reduction only when shield+V-scale combined |
| DH3 | N=32 bank makes shield active (col_sum > κ without shield at large α) | col_sum_noShield P99 > 1.0 at α≥2 |
