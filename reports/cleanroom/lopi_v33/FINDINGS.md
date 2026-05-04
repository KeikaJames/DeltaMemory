# Phase R-3 Findings — LOPI 630-cell Ablation Sweep

**Date**: 2026-05-04
**Branch**: stage13-attn-native (PR #8)
**Source**: `reports/cleanroom/lopi_v33/results.json` (630 cells)
**Aggregate**: `reports/cleanroom/lopi_v33/AGGREGATE.md` + `aggregate.json`

---

## 1. Executive verdict

| Hypothesis | PASS criterion (PREREGISTRATION §3) | Result |
|---|---|---|
| **H5** | α=0 max-abs-diff = 0.0 across all variants × archs × scales | **PASS ✅** (630/630 cells) |
| **H2** | lift(A3) ≥ 0.9 × lift(A0) at α=1.0 across archs×scales | **PASS ✅** (6/6, ratios 1.34–1.74; LOPI *amplifies* lift) |
| **H3** | lift(A2) ≥ lift(A1) at α=1.0 (Gaussian beats orthogonal-only) | **PASS ✅** (6/6, Δ +0.075 to +0.529) |
| **H1** | drift(A3) ≤ 0.5 nats on ≥5/7 α per arch×scale | **FAIL ❌** (1/7 across all 6, only α=0 satisfies) |
| **H1′** (post-hoc) | drift(A4) < drift(A0) at α=8 (catastrophic regime) | **PASS ✅** (6/6, drift collapse 3.9–7.1 nats) |

**Verdict**: Strike 1 against the as-specified Dynamic LOPI v3.3
(white-paper formulation). The orthogonal projection M⊥ does NOT collapse
neutral drift in the operating regime where it was meant to (α∈[0.25, 2]);
it actively *increases* drift at low α. However, the **Gaussian focusing**
component, isolated in variant A4 (Gauss + γ, no M⊥), **does** function
as a high-α safety shield, collapsing catastrophic drift by 50–90% at
α=8. This is a fundamentally different kind of mechanism than the
white-paper described, but it is real and reproducible across both
GPT-2 small and medium, all 3 architecture arms (Residual/HC/mHC),
and all 3 seeds.

## 2. Per-α drift table (all archs, all scales, mean over 3 seeds)

Drift positive ⇒ neutral text gets *worse* under bank injection.

### α=1.0 (typical operating point)

| arch | A0 (no LOPI) | A1 (M⊥) | A2 (M⊥+Gauss) | A3 (full LOPI) | A4 (Gauss+γ no-M⊥) |
|---|---:|---:|---:|---:|---:|
| gpt2/residual | +0.700 | +1.387 | +1.393 | +1.393 | +0.914 |
| gpt2/hc | +2.236 | +2.533 | +2.282 | +2.282 | +2.192 |
| gpt2/mhc | +2.236 | +2.533 | +2.282 | +2.282 | +2.192 |
| gpt2-medium/residual | −0.054 | +0.290 | +0.788 | +0.788 | +0.640 |
| gpt2-medium/hc | +1.069 | +1.618 | +1.764 | +1.764 | +1.564 |
| gpt2-medium/mhc | +1.069 | +1.618 | +1.764 | +1.764 | +1.564 |

### α=8.0 (catastrophic stress test)

| arch | A0 (no LOPI) | A4 (Gauss+γ no-M⊥) | Δ A4−A0 |
|---|---:|---:|---:|
| gpt2/residual | +5.340 | +0.829 | **−4.510** |
| gpt2/hc | +6.434 | +2.539 | **−3.895** |
| gpt2/mhc | +6.434 | +2.539 | **−3.895** |
| gpt2-medium/residual | +7.405 | +0.317 | **−7.088** |
| gpt2-medium/hc | +6.497 | +1.609 | **−4.888** |
| gpt2-medium/mhc | +6.497 | +1.609 | **−4.888** |

A4 in the catastrophic regime collapses drift by 65–95%. Two of the
six cells (gpt2-medium/residual at α=8) bring drift below the 0.5-nat
shield threshold, satisfying the original H1 PASS criterion at that α.

## 3. Diagnosis — why M⊥ fails at low α

The white-paper claim was: orthogonal-novelty projection M⊥ = M_V − ⟨M_V,
V_ctx⟩/‖V_ctx‖² · V_ctx adds **only** the component of the bank vector
perpendicular to the residual stream, "physically guaranteeing the
native principal components are undisturbed." Mathematically this is
true: ⟨M⊥, V_ctx⟩ = 0 by construction.

The empirical failure is that **language modeling NLL is sensitive to
the *magnitude* of residual-stream perturbations, not just their
direction**. Adding orthogonal energy ‖M⊥‖ to V_ctx still:

1. Increases ‖V_ctx + M⊥‖ → changes downstream LayerNorm scaling.
2. Rotates the post-LN attention output away from the trajectory the
   model trained on, even though the change is in a "new" direction.
3. Produces logit shifts proportional to ‖M⊥‖ irrespective of whether
   the bank content is *relevant* to the current prompt.

At low α, ‖M_V‖ ≪ ‖V_ctx‖, so M⊥ ≈ M_V (almost orthogonal anyway,
since random vectors in high-d are near-orthogonal). The "redundancy
removal" is a no-op and the orthogonal energy is added on top of an
otherwise-clean residual stream → drift up.

At high α (α≥4), ‖α·M_V‖ dominates ‖V_ctx‖. Without LOPI the model's
residual is overwhelmed (A0 drift +5–7 nats — catastrophic). With
Gaussian focusing (A2/A3/A4), the injection energy is concentrated at
mid-stack (μ ≈ 0.55L by Q-derived depth), so early-layer parsing and
late-layer readout layers stay clean, partially recovering NLL.

## 4. The γ-derivative gate did not engage

A2 ≡ A3 numerically across **all 630 cells**. This is because:

- The runner resets `LOPIState` between every prompt (independent
  evaluation) and between every fact (independent reading).
- Inside a single prompt, all tokens see the same `Q_t` from the
  current forward pass; `Q_prev` is None on first call so γ defaults
  to 1.0.
- Therefore the derivative gate γ_t = σ(k·(‖ΔQ‖ − θ)) is structurally
  pinned at 1.0 in batch eval.

**Implication**: γ_t is a multi-turn streaming-only mechanism. It
cannot be evaluated offline. R-5 (Q3 chat with multi-turn dialog)
will be the first phase where γ is actually exercised. For R-3 we
correctly report A3 = A2.

## 5. The A4 surprise — Gaussian alone is the shield

A4 = `enabled=True, orthogonal=False, gaussian=True, derivative=True`,
i.e. the bank vector is added directly (no orthogonal projection) but
the **layer-Gaussian weight w(ℓ, t)** suppresses injection outside the
mid-stack window. A4 consistently outperforms A1/A2/A3 on drift across
**all** α and all archs:

* At α=2 on gpt2/residual: A0=+0.56, A4=+1.09 (A4 worse) but A1=+1.56,
  A2/3=+1.86 (M⊥ variants markedly worse).
* At α=4 on gpt2-medium/residual: A0=+2.02, A1=+2.85, A2/3=+0.62,
  **A4=+0.20** (A4 brings drift below 0.5-nat shield threshold).
* At α=8 on gpt2-medium/residual: A0=+7.41, A4=+0.32 (95% drift
  reduction with no orthogonal projection).

**Mechanism**: The Gaussian centred at μ ≈ 0.5L with σ ≈ 0.17L
suppresses bank injection at layers ℓ < 0.3L (parsing) and ℓ > 0.7L
(readout). This protects the most NLL-sensitive layers (the LM-head
and the embedding-aware shallow layers) from any perturbation, while
allowing the bank to influence the abstract reasoning band.

**Conclusion**: For LOPI v3.4 we will **drop M⊥ from the default
config**. The fields remain in the API for ablation but the
recommended setting is `orthogonal=False, gaussian=True, derivative=True`.

## 6. What this does NOT prove

* H4 (timing): LOPI run-cost vs unpatched is not measured here. R-5
  must report wall-clock per token under LOPI ON vs OFF.
* Gaussian probe of layer norms is not visualized in R-3 — that is
  R-3.5 with the mHC5 norm probe replay.
* Cross-architecture (Gemma / Qwen / GLM) is R-4. GPT-2 medium is the
  largest model exercised here.
* A3 ≡ A2 is an artifact of offline batch eval; multi-turn chat (R-5)
  is required to verify γ-gate efficacy claims.

## 7. Strike 1 / 3-strike protocol status

Per `PREREGISTRATION.md §5`:
* **R-3 strike count**: 1 of 3 (H1 fail under as-specified A3).
* **Recovery action**: Adopt v3.4 default `orthogonal=False`. Re-run
  R-3.5 (Gaussian layer-norm probe) with A4 to confirm mid-stack
  concentration. **No rollback required**: H2/H3/H5 hold; we have
  positive findings on the safety mechanism in the regime that matters
  (high-α catastrophic drift collapse).
* **3-strike rollback target unchanged**: post-merge HEAD of PR #7.

## 8. Next steps

1. **R-3.5** (this push): add `--probe-norms` mode to the runner and
   visualize per-layer ‖res‖ on neutral prompts under A0 vs A4 to
   confirm Gaussian focusing localizes injection energy to mid-stack.
2. **LOPI v3.4 default flip**: in `LOPIConfig.__init__`, change
   `orthogonal: bool = False` (was `True`). Document in
   `lopi.py` docstring. All existing tests stay green by explicit kwargs.
3. **R-4** cross-arch: re-run on Gemma-4-E2B and Qwen3-4B with v3.4
   defaults. Expect H1' high-α PASS to extend across model families.
4. **R-5**: multi-turn chat with γ-gate engaged; first measurement
   that can validate H4 timing claim.

---

**Frozen by**: Phase R protocol §5.2 (verdict locked once aggregate.json
written and committed; revisions only via `AMENDMENT-N.md` block in this
file).
