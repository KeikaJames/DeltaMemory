# W.2 LOPI 3-Component Dissection — Report

**Branch**: feat/v04-w2-lopi-dissect
**Run date**: 2026-05-04
**Grid**: 3 models × 7 α × 5 arms × 3 seeds × 30 prompts = **9450/9450 passes** (2.87 h)
**Models**: gpt2-medium, Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B
*(Qwen/Qwen2.5-1.5B substituted for meta-llama/Llama-3.2-1B — see PREREG amendment)*

---

## Arm Definitions Recap

| Arm | M⊥ | Gaussian w(ℓ,t) | Derivative γ_t | Description                    |
|-----|----|-----------------|----------------|--------------------------------|
| A0  | —  | —               | —              | Bank injection, no LOPI        |
| A1  | ✓  | —               | —              | M⊥ projection only             |
| A2  | ✓  | ✓               | —              | M⊥ + Gaussian gate             |
| A3  | ✓  | ✓               | ✓              | Full LOPI                      |
| A4  | —  | ✓               | ✓              | Gaussian + derivative, no M⊥   |

---

## Q1: Does M⊥ orthogonal projection help? — DUAL-ZONE RESULT

**Aggregate verdict (from aggregate.py)**: PASS (A1−A0=−0.034 nats at α≥2)

**Per PREREG strict analysis**:

| Model              | A1−A0 at α∈[0.5,4] | Direction |
|--------------------|---------------------|-----------|
| gpt2-medium        | +0.010              | M⊥ HURTS  |
| Qwen/Qwen2.5-0.5B  | +0.118              | M⊥ HURTS  |
| Qwen/Qwen2.5-1.5B  | +0.024              | M⊥ HURTS  |

**At extreme α=32**:

| Model              | A0     | A1     | A1−A0  |
|--------------------|--------|--------|--------|
| gpt2-medium        | 4.219  | 2.972  | −1.248 |
| Qwen/Qwen2.5-0.5B  | 2.101  | 2.082  | −0.019 |
| Qwen/Qwen2.5-1.5B  | 1.598  | 1.743  | +0.145 |

**Interpretation**: M⊥ exhibits an **α-flip** around α=8–16. In the practical operating
regime (α≤4), M⊥ consistently increases drift across all 3 models, consistent with R-3
findings. At extreme over-injection (α=32), M⊥ provides a large benefit for GPT-2
(−1.25 nats), pulling the aggregate average negative. The PREREG FAIL condition
("A1 > A0 in α∈[0.5,4] for all models") is **met**. The PREREG PASS condition
("A1 ≤ A0 averaged at α≥2") is also met due to GPT-2 α=32 outlier. **True verdict:
Q1 = FAIL in the operating regime, with a high-α caveat for GPT-2.**

**M⊥ energy ratio** (fraction of V-energy in the orthogonal complement):
Arms A1/A2/A3 all show mean ≈ 0.921–0.923 (92% of V-energy is M⊥).
A4 (no M⊥) shows 1.000 as expected (all energy retained).

---

## Q2: Does Gaussian gate w(ℓ,t) help? — BORDERLINE / MODEL-DEPENDENT

**Aggregate verdict**: PASS (A2−A1=−0.092 nats, aggregate mean at all α)

**Per-model analysis at α≥4**:

| Model              | A2−A1 mean (α≥4) | Direction |
|--------------------|------------------|-----------|
| gpt2-medium        | −0.785           | Gaussian HELPS strongly |
| Qwen/Qwen2.5-0.5B  | +0.519           | Gaussian HURTS consistently |
| Qwen/Qwen2.5-1.5B  | −0.173           | Gaussian helps marginally |

The aggregate PASS is **driven entirely by GPT-2** at high α. GPT-2 at α=32:
A1=2.972, A2=0.566 (−81% drift reduction). For Qwen 2.5-0.5B, adding the Gaussian
gate ON TOP OF M⊥ consistently INCREASES drift — the Gaussian interferes with
the beneficial layer spread of M⊥.

**PREREG Q2 PASS requires ≥ 2/3 models to show A2 < A1 at α≥2.** Only 1/3 clearly
shows this (GPT-2). Qwen 1.5B shows a mixed pattern (helps at α=4 and α=32, hurts
at α=8 and α=16). **True verdict: Q2 = FAIL by per-model PREREG criteria.**

**w(ℓ,t) argmax vs mu_arch**:

| Model              | w_ell_argmax | mu_arch (profiler) | Offset |
|--------------------|-------------|---------------------|--------|
| gpt2-medium        | 9           | N/A (static)        | —      |
| Qwen/Qwen2.5-0.5B  | 9           | 5                   | +4     |
| Qwen/Qwen2.5-1.5B  | 10          | 5                   | +5     |

The Gaussian peak consistently appears at layer 9–10, **4–5 layers deeper than the
profiler's mu_arch=5**. The PREREG Q2 condition (b) — "argmax within ±2 of mu_arch"
— is **NOT met** for either Qwen model. This suggests the Gaussian implementation
has a systematic centering error, or the profiler's mu_arch estimate is computed
in a different coordinate space than the Gaussian application.

---

## Q3: Does derivative gate γ_t help? — CONFIRMED FAIL

**Verdict**: FAIL (A3−A2=+0.000 nats; gate pinned at 1.0 — unanimous across all
9450 passes)

| Metric               | Value |
|----------------------|-------|
| gamma_t_mean (all A3 cells) | 1.0000 |
| gamma_t range        | [1.0, 1.0] |
| A3 − A2 drift (mean) | 0.000 |

A3 is **numerically identical** to A2 for every model, α, seed, and prompt. The
derivative gate γ_t = max(0, cos(Q_curr, Q_prev)) never fires because `LOPIState`
is constructed fresh for each prompt (Q_prev is always None → γ_t = 1.0 by
default). This is the same finding as R-3. The test reveals the gate is architecturally
non-functional in the single-prompt evaluation regime.

---

## Summary Table

| Question  | Preregistered Prediction | Actual Result | Notes |
|-----------|--------------------------|---------------|-------|
| Q1: M⊥    | FAIL                     | FAIL (operating regime) / PASS (aggregate) | α-flip at α≈8–16 |
| Q2: Gauss | PASS                     | FAIL (per-model) / PASS (aggregate)        | Only GPT-2 benefits; Qwen argmax offset |
| Q3: γ_t   | FAIL                     | FAIL (confirmed, unanimous)                | Gate always at 1.0 |

---

## What W-T3 Should Fix Next

### Fix 1 (Critical): Gaussian centering error
- **Symptom**: w_ell_argmax = 9–10 for Qwen models; mu_arch = 5 from profiler
- **Likely cause**: The Gaussian may be applied as `w(ℓ) = exp(-((ℓ - mu_arch*L/N)^2) / σ^2)` 
  where L is the total number of layers and N is the profiler's scale, causing the center
  to shift to layer `mu_arch * L / N` instead of `mu_arch` directly.
- **Fix**: Normalize layer index to [0,1] before applying the Gaussian, or re-scale
  mu_arch to absolute layer index before LOPIConfig construction.
- **Expected impact**: Gaussian focusing would apply to the correct layers,
  potentially making Q2 pass for Qwen models.

### Fix 2 (High): Derivative gate — carry state across tokens
- **Symptom**: γ_t always 1.0 because Q_prev = None (fresh LOPIState per prompt)
- **Fix**: Pass a persistent LOPIState through sequential generation steps so Q_prev
  is populated after the first token. Test in an autoregressive generation loop,
  not in a single-forward NLL evaluation.
- **Expected impact**: Q3 may become PASS in generation-time evaluation.

### Fix 3 (Medium): M⊥ α-adaptive scaling
- **Symptom**: M⊥ hurts at low α (reduces bank contribution too aggressively),
  helps at extreme α (correctly orthogonalizes overpowering bank vectors)
- **Fix**: Make the M⊥ projection strength proportional to `min(alpha / alpha_ref, 1.0)`,
  so M⊥ is a no-op at α≤1 and full-strength at α≥alpha_ref.
- **Expected impact**: Q1 may pass in operating regime without the high-α caveat.

### Fix 4 (Low): Investigate Qwen Gaussian interference with M⊥
- **Symptom**: A2 > A1 for Qwen 0.5B (adding Gaussian gate on top of M⊥ increases drift)
- **Likely cause**: Gaussian suppresses layers where M⊥ provides benefit, while
  focusing on layers where it does not help.
- **Action**: Profile per-layer drift contribution (residual_norm_per_layer) to
  understand which layers drive the Gaussian interference for Qwen.

---

## Figures Produced

- `fig1_arm_vs_alpha.svg` — Drift vs α for all 5 arms, per model
- `fig2_marginal_M_perp.svg` — M⊥ energy ratio marginal
- `fig3_marginal_gauss.svg` — Gaussian gate effect marginal
- `fig4_marginal_deriv.svg` — Derivative gate effect marginal
- `fig5_gamma_t_hist.svg` — γ_t distribution (confirms gate saturation)

---

## Appendix: Raw Numbers (key α values)

### gpt2-medium

| α   | A0    | A1    | A2    | A3    | A4    |
|-----|-------|-------|-------|-------|-------|
| 0.5 | 0.040 | 0.043 | 0.041 | 0.041 | 0.042 |
| 4   | 0.049 | 0.072 | 0.070 | 0.070 | 0.062 |
| 8   | 0.189 | 0.182 | 0.088 | 0.088 | 0.077 |
| 16  | 0.808 | 0.812 | 0.173 | 0.173 | 0.145 |
| 32  | 4.220 | 2.972 | 0.566 | 0.566 | 0.598 |

### Qwen/Qwen2.5-0.5B

| α   | A0    | A1    | A2    | A3    | A4    |
|-----|-------|-------|-------|-------|-------|
| 0.5 | 2.446 | 2.495 | 2.484 | 2.484 | 2.468 |
| 4   | 1.863 | 1.914 | 2.516 | 2.516 | 2.422 |
| 8   | 1.432 | 1.300 | 2.388 | 2.388 | 2.112 |
| 16  | 1.757 | 1.864 | 2.120 | 2.120 | 1.953 |
| 32  | 2.101 | 2.082 | 2.212 | 2.212 | 1.886 |

### Qwen/Qwen2.5-1.5B

| α   | A0    | A1    | A2    | A3    | A4    |
|-----|-------|-------|-------|-------|-------|
| 0.5 | 0.817 | 0.737 | 0.915 | 0.915 | 0.942 |
| 4   | 0.665 | 0.927 | 0.791 | 0.791 | 0.797 |
| 8   | 0.654 | 0.771 | 0.883 | 0.883 | 0.756 |
| 16  | 0.852 | 0.896 | 0.918 | 0.918 | 0.791 |
| 32  | 1.598 | 1.743 | 1.053 | 1.053 | 0.800 |
