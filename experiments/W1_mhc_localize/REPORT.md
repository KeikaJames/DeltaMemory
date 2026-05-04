# W.1 — mHC Shield Failure Localization

**Branch:** `feat/v04-w1-mhc-localize`  
**Date:** 2025-07  
**Verdict:** **FAIL** (drift floor >> 0.5 nats acceptance threshold)

---

## 1. Experimental Setup

### 1.1 Controlled Variables (Ablation Grid)

| Factor | Levels |
|--------|--------|
| α (bank injection strength) | 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0 |
| mHC shield | on / off |
| κ (shield clipping threshold) | 1.0, 2.0, 4.0 |
| V-scale (`value_scale_mode`) | `auto_rms_cap` / `none` |
| Seeds | 0 (single seed) |

Unique cells per model: **56** (shield=off only uses κ=1; α=0 produces 0 drift regardless of other factors).

### 1.2 Models Evaluated

| Model | Status | Notes |
|-------|--------|-------|
| Qwen/Qwen2.5-0.5B | ✅ **Run** | LlamaAdapter via Qwen2Attention |
| Qwen/Qwen2.5-1.5B | ✅ **Run** | LlamaAdapter via Qwen2Attention |
| gpt2 | ⏭ **Skipped** | Uses `transformer.h[i].attn` — not compatible with AttnNativePatcher (requires `model.layers[i].self_attn`) |
| gpt2-medium | ⏭ **Skipped** | Same architecture as gpt2 |
| meta-llama/Llama-3.2-1B | ⏭ **Skipped** | Gated repository; 403 auth error even with HF_TOKEN |

### 1.3 Dataset and Metric

- **Dataset:** 30 WikiText-2 prompts (256 tokens each), device=MPS, dtype=bfloat16
- **Metric:** `mean_drift` = mean(NLL_with_bank − NLL_baseline) across prompts
- **Acceptance criterion (PASS):** `mean_drift(shield=on, V-scale=on, α≥1) ≤ 0.5 nats` per model

### 1.4 Infrastructure

- `run.py`: sweep runner with `patched_kappa()` context manager (monkey-patches
  `deltamemory.memory.mhc_shield.shield_attention_weights` per cell)
- `aggregate.py`: reads `cells.jsonl`, emits `summary.csv`, 4 SVG marginal plots, `verdict.md`
- No deltamemory source files were modified

---

## 2. Results

### 2.1 Overall Verdict

| Model | PASS? | mean_drift (shield+V-scale, α≥1) |
|-------|-------|----------------------------------|
| Qwen/Qwen2.5-0.5B | ❌ FAIL | **3.31 nats** |
| Qwen/Qwen2.5-1.5B | ❌ FAIL | **4.39 nats** |

**Threshold:** ≤ 0.5 nats. Both models miss by **6–9×**.

### 2.2 Drift Floor (Best Configuration)

The best achievable configuration is: shield=on, κ=1.0, V-scale=on, α=0.5.

| Model | Baseline NLL | Drift Floor | Drift/NLL ratio |
|-------|-------------|-------------|-----------------|
| Qwen/Qwen2.5-0.5B | 3.19 nats | **2.46 nats** | 77% |
| Qwen/Qwen2.5-1.5B | 2.88 nats | **2.70 nats** | 94% |

### 2.3 Marginal Effect of α (holding shield=on, κ=1, V-scale=on)

| α | 0.5B drift | 1.5B drift |
|---|-----------|-----------|
| 0.0 | 0.00 | 0.00 |
| 0.5 | 2.46 | 2.70 |
| 1.0 | 2.46 | 2.73 |
| 2.0 | 2.46 | 2.82 |
| 4.0 | 2.47 | 3.19 |
| 8.0 | 2.57 | 3.95 |
| 16.0 | 3.42 | 5.26 |

Key observation: drift is **nearly flat across α=0.5–4** with shield+κ=1+V-scale. This indicates the shield successfully bounds the worst-case interference, but a **constant baseline drift floor of ~2.5 nats persists** regardless of injection strength.

### 2.4 Shield Effectiveness (% drift reduction vs no-shield, same α)

| α | 0.5B reduction | 1.5B reduction |
|---|---------------|---------------|
| 0.5 | 4% | 20% |
| 1.0 | 44% | 39% |
| 4.0 | 73% | 63% |
| 16.0 | 72% | 47% |

Shield is highly effective at higher α (preventing runaway drift) but cannot eliminate the ~2.5 nat floor at low α.

### 2.5 Effect of κ

Higher κ allows more drift through:

| κ | 0.5B drift at α=4 | 1.5B drift at α=4 |
|---|------------------|------------------|
| 1.0 | 2.47 | 3.19 |
| 2.0 | 2.82 | 3.51 |
| 4.0 | 4.83 | 4.27 |

κ=1.0 is optimal; increasing κ monotonically increases drift.

### 2.6 Effect of V-scale

V-scale (value RMS capping) modestly reduces drift at low α but has minimal effect:

- At α=0.5: 0.5B drift 4.26 (V=off) → 2.46 (V=on): mostly due to shield
- At α=1.0: 0.5B drift 2.45 (V=off) → 2.46 (V=on): nearly identical

V-scale alone (shield=off) reduces drift by ~1–2 nats at low α but does not prevent catastrophic drift at high α.

---

## 3. Diagnostic Hypotheses

### DH1: mHC shield truncates bank column sums?

**Status: CONFIRMED** (partial)

bank_col_sum_p99: shield=ON → 559.8, shield=OFF → 561.9 (−0.4%). The shield does reduce extreme col-sum values, but the reduction is very small (~0.4%), suggesting the shield's primary effect is on attention weight distribution, not raw column sums.

**Implication:** The drift floor is not caused by pathologically large bank entries. The problem is structural — even well-bounded bank entries cause large NLL shifts.

### DH2: V-scale caps residual norm inflation?

**Status: PARTIALLY CONFIRMED**

V-scale (auto_rms_cap) reduces drift by ~0.1–0.2 nats compared to no V-scale at low α. The residual_norm_p50 is comparable (~15.2–15.3 for both). V-scale is a secondary, minor contributor.

### DH3: Drift tracks α linearly without shield?

**Status: REFUTED — sublinear at high α**

Without shield: drift is roughly log-proportional to α (0.5→4.3 nat, 8→12.2 nat). With shield: drift saturates at a ~2.5 nat floor, confirming the shield is the binding constraint, not α.

---

## 4. Q&A

**Q1: Does the mHC shield effectively localize bank interference?**  
Partially. The shield prevents catastrophic drift at high α (73% reduction at α=4) but cannot eliminate a **2.5–2.7 nat baseline floor** that appears even at α=0.5. The floor is present in both model sizes.

**Q2: What is the minimum achievable drift with current architecture?**  
~2.46 nats (0.5B) / ~2.70 nats (1.5B) at shield+κ=1+V-scale+α=0.5. This is 77–94% of baseline NLL — the bank nearly doubles the model's token prediction loss at minimum injection.

**Q3: Is the drift floor α-dependent or architecture-induced?**  
It is largely **architecture-induced**: drift is flat across α=0.5–4 with best config. Removing α injection entirely (α=0) gives drift=0. Therefore the floor is a property of the patcher/injection mechanism itself at any non-zero α, not of injection strength.

**Q4: Should W-T1 try next?**  
See Section 5.

---

## 5. What W-T1 Should Try Next

1. **Investigate the injection mechanism at α=0.5 (near-minimum α)**  
   The drift floor appears even at the smallest α. Hypothesis: the act of injecting any bank entry into attention modifies the attention score distribution, causing a persistent NLL shift independent of content. Next step: run a "null bank" ablation where bank tensors are all-zeros at α=0.5 to isolate whether the drift is from *bank content* vs *routing overhead*.

2. **Implement a soft-injection mode with learnable α gating**  
   The current hard-injection (`α * bank_value`) forces a constant shift. A learned per-layer gate (sigmoid(α) × bank_contribution) could learn to suppress injection on layers where it hurts NLL, allowing zero drift on non-matching queries while preserving recall on matched ones.

3. **Test on a generative recall task, not NLL**  
   The 0.5 nat acceptance threshold was calibrated for generic NLL. On a targeted recall task (e.g., "What is X? → stored answer"), the question is whether the bank improves recall accuracy at the cost of NLL on unrelated tokens. The W.1 sweep only measures harm (NLL drift), not benefit. A reward–harm tradeoff curve (δ_recall_accuracy / δ_mean_drift) would give a more actionable verdict.

---

## 6. Experimental Methodology

**Paradigm:** Controlled ablation (单变量控制法 + 消融实验)

Each factor (α, shield, κ, V-scale) was varied independently while holding all others at their default/best value in marginal-effect analyses. The full grid allows interaction analysis. This is the standard approach for localizing which component is responsible for an observed effect (here: NLL drift).

**Reproducibility:**  
```bash
# Reproduce sweep
source .venv-mac/bin/activate
python -m experiments.W1_mhc_localize.run \
  --model Qwen/Qwen2.5-0.5B \
  --output experiments/W1_mhc_localize/cells.jsonl \
  --seeds 0

python -m experiments.W1_mhc_localize.run \
  --model Qwen/Qwen2.5-1.5B \
  --output experiments/W1_mhc_localize/cells.jsonl \
  --seeds 0

# Aggregate
python -m experiments.W1_mhc_localize.aggregate \
  --input experiments/W1_mhc_localize/cells.jsonl \
  --outdir experiments/W1_mhc_localize/figures
```

---

## 7. Files

| File | Description |
|------|-------------|
| `PREREG.md` | Pre-registration with acceptance criteria |
| `run.py` | Sweep runner (no deltamemory modification) |
| `aggregate.py` | Aggregation, plots, verdict |
| `cells.jsonl` | Raw cell data (112 cells: 56/model) |
| `summary.csv` | Per-cell summary with all signals |
| `figures/verdict.md` | Machine-readable verdict |
| `figures/holding_others_fixed_alpha.svg` | Marginal effect of α |
| `figures/holding_others_fixed_shield.svg` | Marginal effect of shield |
| `figures/holding_others_fixed_kappa.svg` | Marginal effect of κ |
| `figures/holding_others_fixed_v_scale.svg` | Marginal effect of V-scale |
