# W.2 LOPI 3-Component Dissection — Preregistration

**Date**: 2026-05-04
**Branch**: feat/v04-w2-lopi-dissect
**Phase**: W (wide-grid experiments), W.2 — LOPI dissect

**PRE-RUN AMENDMENT (2026-05-04)**: `meta-llama/Llama-3.2-1B` substituted with
`Qwen/Qwen2.5-1.5B` because Llama-3.2 is a gated HF model requiring a token that
is not available in this environment. Qwen2.5-1.5B shares the same RoPE + Qwen2
architecture as Qwen2.5-0.5B but at 3× parameter scale, providing a distinct
capacity data point without breaking the experimental design.

---

## 1. Research Question

> "Is LOPI working because of orthogonal projection M⊥, the gaussian gate
> w(ℓ,t), or the derivative gate γ_t — or some combination?"

---

## 2. Grid Specification

| Dimension    | Values                                                    | Count |
|--------------|-----------------------------------------------------------|-------|
| Model        | gpt2-medium, Qwen/Qwen2.5-0.5B, **Qwen/Qwen2.5-1.5B**  | 3     |
| α (alpha)    | 0.5, 1, 2, 4, 8, 16, 32                                  | 7     |
| Arm (ablation)| A0–A4 (see §3)                                           | 5     |
| Seed         | 0, 1, 2                                                   | 3     |
| Prompts      | gold_30prompts.jsonl                                      | 30    |
| **Total**    |                                                           | **9450** forward passes |

**mHC shield**: ON for all models (V-scale `auto_rms_cap` per model).
**Diagnostics**: DiagnosticRecorder enabled for all RoPE models; manual
collection for GPT-2.

---

## 3. Ablation Arms

| Arm | orthogonal (M⊥) | gaussian (w(ℓ,t)) | derivative (γ_t) | Description                     |
|-----|-----------------|-------------------|------------------|---------------------------------|
| A0  | —               | —                 | —                | LOPI OFF (no-bank baseline)     |
| A1  |                |                  |                 | M⊥ only                         |
| A2  |                |                  |                 | M⊥ + Gaussian                   |
| A3  |                |                  |                 | Full LOPI (v3.3 spec)            |
| A4  |                |                  |                 | Gauss + γ, no M⊥ (v3.4 default) |

Note: A0 measures NLL with bank injected but LOPI disabled (alpha-weighted
M_V added directly), providing the unmodified injection baseline.

---

## 4. Measurements

Per cell: (model, arm, alpha, seed, prompt_id)
- `mean_drift` — (injected NLL − baseline NLL), where baseline = no-bank
  (α=0) forward
- `lopi_gamma_t` — mean, p50, p99 over tokens in this forward
- `lopi_w_ell_argmax` — layer index with maximum Gaussian weight
- `m_perp_energy_ratio_mean` — mean over layers of ‖M⊥‖²/‖M_V‖²
- `residual_norm_p50` — per-layer p50 of residual L2 norm

---

## 5. PASS/FAIL Criteria (preregistered)

### Q1 — M⊥ utility

**PASS**: A1 mean_drift ≤ A0 mean_drift for α ≥ 2, averaged over (models,
seeds, prompts). Interpretation: orthogonal projection reduces drift in
at least the high-α regime.

**FAIL**: A1 mean_drift > A0 mean_drift in the α ∈ [0.5, 4] operating
regime for all 3 models. Interpretation: M⊥ increases neutral-text drift
(consistent with R-3 findings).

**α flip threshold**: The α value at which A1 transitions from worse-than
to better-than A0 (or "never" if always worse).

---

### Q2 — Gaussian gate utility

**PASS**: (a) A2 mean_drift < A1 mean_drift at α ≥ 2 for ≥ 2/3 models,
AND (b) `lopi_w_ell_argmax` clusters within ±2 layers of the U-LOPI
profiler's μ_arch for each model.

**FAIL**: A2 improvement over A1 is < 0.1 nats at all α, OR Gaussian
center is consistently at wrong layer (> 4 layers from μ_arch).

---

### Q3 — Derivative gate utility

**PASS**: A3 mean_drift < A2 mean_drift by ≥ 0.1 nats at some α ≥ 1 for
≥ 2/3 models, AND `lopi_gamma_t_p50` < 0.9 (gate not always saturated).

**FAIL**: A3 ≈ A2 (< 0.05 nat difference) across all α and models, AND
`lopi_gamma_t_p50` ≥ 0.99 (derivative gate always pinned at 1.0 because
batch eval resets state between prompts).

**Note**: R-3 found A2 ≡ A3 numerically for GPT-2 because the runner
reset `LOPIState` between every prompt so `Q_prev` was always None →
γ_t = 1. W.2 tests whether this holds on multi-token prompts with state
carried across tokens within a single prompt, which is a different regime.

---

## 6. Expected Outcomes (based on R-3 priors)

Based on `reports/cleanroom/lopi_v33/FINDINGS.md`:
- Q1 expected FAIL: M⊥ increases drift at low-mid α, consistent with R-3
- Q2 expected PASS: Gaussian focusing collapses high-α drift by 50–90%
- Q3 expected FAIL: derivative gate likely doesn't fire in batch eval
  (single-prompt state reset, γ_t stays at 1.0)

---

## 7. Decision Rule

| Q1 | Q2 | Q3 | Decision |
|----|----|----|---------|
| PASS | PASS | PASS | Full LOPI v3.3 earns its place |
| FAIL | PASS | FAIL | Ship A4 (Gauss+γ, no M⊥) as v3.5 default |
| FAIL | PASS | PASS | Ship A3 with orthogonal=False default |
| FAIL | FAIL | any | Evaluate CAA baseline (W.3) |
| any  | any | FAIL | Remove derivative gate (save compute) |

---

*Preregistered before any W.2 cells are run.*
