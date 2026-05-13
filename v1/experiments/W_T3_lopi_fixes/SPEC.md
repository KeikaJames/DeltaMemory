# W-T3: LOPI Defect Repair Specification

**Status**: spec draft, locked for round-1 implementation.
**Authored**: 2026-05-04 (post W.2 verdict, post W.3 decision gate).
**Branch policy**: each fix lands on its own `feat/wt3-fix-N-*` branch and is
gated by a re-run of the W.2 grid arm that exposed the defect (subset, not
the full 9450 cells). No fix is merged without a reproducible delta on the
relevant arm.

W.3 demoted LOPI to ablation-only, so W-T3 is **not** required for v0.4 to
ship. It exists to keep LOPI on the shelf as a credible ablation, and to
disconnect "LOPI is broken" from "the abstractions inside LOPI are broken."
A clean LOPI raises the floor of every future ablation.

---

## Fix 1 — Gaussian centering error (CRITICAL)

### Symptom (from W.2 §Q2)

| Model              | profiler mu_arch | runtime w_ell_argmax | offset |
|--------------------|------------------|----------------------|--------|
| Qwen/Qwen2.5-0.5B  | 5                | 9                    | +4     |
| Qwen/Qwen2.5-1.5B  | 5                | 10                   | +5     |

PREREG Q2 condition (b) — "argmax within ±2 of mu_arch" — fails for both
Qwen models. The Gaussian peak lands 4-5 layers below where calibration says
it should.

### Suspected mechanism

In `deltamemory/memory/lopi.py::layer_gaussian_weight`, auto-mode computes:

```
mu_t = float(profile.mu_arch) + cfg.auto_mu_c * (d_t - 0.5) * L
```

`d_t` is the Z-depth signal from `_z_depth_signal(state, cfg)`. On real
prompts `d_t` saturates above 0.5 well before layer mu_arch is reached, so
the additive drift term is consistently positive and grows with `L`. For
Qwen2.5-0.5B (L=24) with `auto_mu_c=0.3` and `d_t≈0.85`, the drift term
contributes `≈2.5` layers; combined with any small bias in `mu_arch` itself
(profiler uses `argmax(sigma_base)` which can sit on a noisy early peak),
the runtime peak ends up at layer 9-10.

### Fix

1. **Probe (do not fix yet)**: emit per-layer `(d_t, mu_t, w_ell)` to the
   diagnostic recorder under flag `lopi_gauss_probe=true`. Ship as a single
   JSONL row per `(model, prompt_id, layer_idx)`. Use this output to confirm
   whether the offset comes from `d_t` saturation, `mu_arch` instability, or
   both, before changing any default.
2. **Anchor mu_arch on residual energy, not residual sigma.** Switch the
   profiler's `mu_arch = _argmax_low_tiebreak(sigma_base)` to
   `mu_arch = _argmax_low_tiebreak(mu_base)` (largest residual norm rather
   than largest std). This shifts mu_arch toward the layers that actually
   carry semantic content, which is what the Gaussian is meant to weight.
3. **Cap drift term magnitude.** Change the drift term to
   `min(auto_mu_c * (d_t - 0.5) * L, sigma_t)`. Past one sigma the Gaussian
   has already decayed, so additional drift is meaningless and only
   destabilizes runtime alignment.
4. **Add a self-test** under `tests/test_lopi_gauss_centering.py` that
   asserts `|w_ell_argmax - mu_arch| <= 2` for at least one Qwen and one
   GPT-2 prompt pair. Without this assertion the regression surface is
   invisible.

### Acceptance

Re-run W.2 arm A2 on `{Qwen2.5-0.5B, Qwen2.5-1.5B}` × α∈{2,4,8} × seed=0 ×
30 prompts (180 cells). Pass condition: argmax-offset ≤ 2 layers AND
A2 < A1 mean-drift on Qwen 0.5B.

---

## Fix 2 — γ_t derivative gate state-carry (HIGH)

### Symptom (from W.2 §Q3)

`gamma_t_mean = 1.0000` across all 9450 cells; range `[1.0, 1.0]`. The gate
never fires.

### Mechanism (confirmed by code inspection)

`apply_lopi` constructs `LOPIState` per call site. In the W.2 evaluation
harness, each prompt is a single forward pass, so `Q_prev` is `None` at
every layer and `derivative_gate(q_post, None, ...) → 1.0` by design (see
`lopi.py:231-234`).

This is correct behaviour for a single-shot NLL evaluation. It is broken
behaviour for autoregressive generation, where `Q_prev` should accumulate
across decoded tokens.

### Fix

1. **Lift `LOPIState` ownership out of `apply_lopi`.** The caller (the
   patched attention forward) must thread a single `LOPIState` instance
   through all layers and all decoding steps for one generation. Concretely:
   - Add `lopi_state: LOPIState | None = None` to `AttnNativePatcher.config`.
   - On the first decoding step, allocate one and stash it on the patcher.
   - On subsequent steps, reuse the same instance.
   - On `reset()`, drop it. Reset is called per prompt by the runner.
2. **Carry `q_post` per layer, not globally.** `LOPIState.q_prev` becomes
   `dict[int, torch.Tensor]` keyed by `layer_idx`. The derivative gate at
   layer ℓ compares `q_post_ℓ_t` to `q_post_ℓ_{t-1}`, not to a different
   layer's `q_post`.
3. **Evaluation contract.** γ_t fixes only show up in **autoregressive
   generation** drift, not single-forward NLL. W-T3 must include a new
   evaluation arm — `experiments/W2_lopi_dissect/run.py` extended with
   `--eval_mode generate` — where each prompt produces 16 tokens and drift
   is computed token-mean over the generated suffix.

### Acceptance

Re-run W.2 arm A3 vs A2 in **generate mode** on
`{gpt2-medium, Qwen2.5-0.5B, Qwen2.5-1.5B}` × α∈{2,4,8} × seed=0 ×
30 prompts × 16 generated tokens. Pass condition:
`gamma_t std > 0` AND `A3 - A2 < 0` for at least 2/3 models.
This becomes Q3-revised in the W-T3 round-1 report.

---

## Fix 3 — M⊥ α-adaptive scaling (MEDIUM)

### Symptom (from W.2 §Q1)

| Model              | A1 - A0 at α∈[0.5,4] | A1 - A0 at α=32 |
|--------------------|----------------------|-----------------|
| gpt2-medium        | +0.010 (HURTS)       | -1.248 (HELPS)  |
| Qwen/Qwen2.5-0.5B  | +0.118 (HURTS)       | -0.019          |
| Qwen/Qwen2.5-1.5B  | +0.024 (HURTS)       | +0.145 (HURTS)  |

M⊥ orthogonalization is a no-op-or-harm at small α and a big help at
extreme α. The defect is that the projection strength is constant — it
should scale with how badly the bank vector is overpowering V_ctx.

### Fix

In `deltamemory/memory/lopi.py::orthogonal_novelty`, replace the unconditional
projection with a soft α-gated blend:

```
def orthogonal_novelty(m_v, v_ctx, eps, alpha=1.0, alpha_ref=4.0):
    v_norm_sq = (v_ctx * v_ctx).sum(dim=-1, keepdim=True) + eps
    dot = (m_v * v_ctx).sum(dim=-1, keepdim=True)
    m_parallel = (dot / v_norm_sq) * v_ctx
    rho = min(alpha / alpha_ref, 1.0)   # 0 at alpha=0, 1 at alpha>=alpha_ref
    return m_v - rho * m_parallel
```

`alpha_ref` defaults to `4.0` and is exposed in `LOPIConfig`. At α=0 the
function is bit-equal to the additive baseline (red-line preserved). At
α≥4 the function is bit-equal to the legacy v3.4 M⊥. In between it
linearly interpolates the projection strength.

### Acceptance

Re-run W.2 arm A1 on all 3 models × α∈{0.5, 1.0, 2.0, 4.0, 8.0, 32.0} ×
seed=0 × 30 prompts (540 cells). Pass condition:
- α≤4: `A1 - A0 ≤ 0` mean across all 3 models, AND
- α=32: `A1 - A0 ≤ -1.0` for gpt2-medium (preserve the high-α benefit).

The α=0 cells must satisfy `|A1 - A0| < 1e-4` (red-line).

---

## Fix 4 — Qwen Gaussian × M⊥ interference probe (LOW)

### Symptom (from W.2 §Q2)

`A2 - A1 = +0.519 nats` for Qwen2.5-0.5B at α≥4. Adding the Gaussian gate
*on top of* M⊥ consistently increases drift on Qwen, while it consistently
helps on GPT-2. Suggests the Gaussian peak (layer 9-10 per Fix 1 symptom)
sits where M⊥ *already* cleaned things up, and the down-weighting of other
layers re-introduces drift those layers were absorbing.

### Fix

This is a probe, not a code change. Run before Fix 1's Gaussian recentering
lands so the symptom is measured before the centering bug is healed.

1. Extend `experiments/W2_lopi_dissect/run.py` with a new diagnostic column
   `residual_norm_per_layer` recorded into `cells.jsonl` for arms A1, A2.
2. Compute, per-cell, the **layer-resolved A2 - A1 drift contribution**:
   for each layer ℓ, accumulate the change in residual norm between A1
   and A2 conditional on the Gaussian weight at ℓ.
3. Render `fig6_qwen_a2_vs_a1_per_layer.svg` overlaying the Gaussian
   weight curve and the per-layer drift contribution. If the Gaussian peak
   coincides with a *reduced* drift contribution, Fix 1 is sufficient.
   If the peak coincides with an *increased* drift contribution, Fix 1
   plus a per-arch Gaussian damping factor `eta_sigma_qwen` is required.

### Acceptance

Probe report in `experiments/W2_lopi_dissect/PROBE_FIX4.md`. No code
default changes from this fix alone. Result drives whether Fix 1 needs a
per-arch eta_sigma override.

---

## Round-1 deliverable bundle

When Fixes 1-3 land (Fix 4 is probe-only):

- `feat/wt3-fix-1-gauss` branch: profiler change + cfg + tests
- `feat/wt3-fix-2-gamma` branch: state-carry + generate eval mode + tests
- `feat/wt3-fix-3-mperp` branch: alpha-gated projection + tests
- `experiments/W_T3_lopi_fixes/REPORT.md`: re-run results (subset cells),
  per-fix verdict, comparison table vs original W.2 numbers.
- `tests/test_lopi_wt3_redlines.py`: α=0 bit-equality witnesses for all
  three fixes, max |drift| < 1e-4.

If ≥2 of {Fix 1, Fix 2, Fix 3} produce per-PREREG passes on their
acceptance arm, LOPI is re-promoted from ablation-only to candidate
default. If <2 pass, LOPI stays ablation-only and the round-1 results are
nonetheless committed as the formal closure of the W.2 line.

---

## Out of scope

- Re-running the full 9450-cell grid (deferred until and unless ≥2 fixes
  pass).
- Cross-architecture LOPI re-tuning (R-3.5-style probes are subsumed).
- Any fix that requires re-fitting the bank between prompts.
- ECOR or V-scale interactions with LOPI — those live in W-T3.6 and X.7
  follow-ups respectively.
