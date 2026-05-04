# Phase R — Dynamic LOPI v3.3: Preregistration

**Frozen**: 2026-05-04
**Branch**: stage13-attn-native
**Commit**: f0b64cb9 (post R-pre seq-NLL drift correction; PR #6 head)

---

## 1. Scope and Motivation

The mHC paper (`reports/cleanroom/mHC_alpha_safe/REPORT.md`, Amendment 1)
established two corrected facts:

1. **Lift advantage is real.** Multi-stream HC/mHC reroutes amplification at
   depth: GPT-2 medium α=1.0 lift gap = **+4.13 nats** vs Residual.
2. **Drift safety is *not* delivered by mHC alone.** Under true sequence-NLL,
   HC/mHC has *higher* neutral-prompt drift than Residual (e.g. small α=1.0:
   HC/mHC +2.26 vs Residual +0.70). H2 fails decisively.

Phase R closes the drift gap with a **training-free** mechanism: **Dynamic
LOPI** (Layer-Orthogonal-Projection Injection). The architecture is
specified by the user white-paper of 2026-05-04 ("Dynamic LOPI 自适应层级正交
投影注入").

> **Industrial positioning.** Dynamic LOPI must be **architecture-agnostic**
> (works on every HF causal LM the bank already supports: GPT-2, Gemma,
> Qwen, Llama, GLM), **zero-train** (no new parameters trained at deploy
> time), and **bit-equal at α=0** (red line invariant from v3.0).

## 2. Five Hypotheses

| ID | Hypothesis | Verification Phase | PASS criterion |
|---|---|---|---|
| **H1** | LOPI's orthogonal projection collapses neutral drift to ≤ 0.5 nats at α∈[0.1, 5.0] across HC + mHC + Residual on GPT-2 small AND medium | R-3 ablation (M_⊥ on/off) | shield-equivalent threshold; ≥ 5/7 α points per arch, 3/3 archs, 2/2 scales |
| **H2** | LOPI preserves the 4.13-nat lift advantage of HC/mHC over Residual on GPT-2 medium at α=1.0 (i.e. M_⊥ does not strip the bank's useful counter-prior signal) | R-3 ablation | mHC lift ≥ +0.30 nats AND Residual lift ≤ −2.5 nats at α=1.0 (≤ 30% relative degradation from no-LOPI baseline of +0.48 / −3.65) |
| **H3** | Adaptive layer Gaussian (Adaptive Layer Routing) provides ≥ 0.10 nats lift improvement over uniform-layer injection at α=1.0 on at least one of {GPT-2 medium, Gemma-4-E2B} | R-3 ablation (uniform vs Gaussian) | per-arch matched-α A/B with seeds 0/1/2; mean Δlift ≥ +0.10 nats with seed-stderr non-overlapping zero |
| **H4** | Derivative gate γ_t reduces wall-clock injection cost by ≥ 30% on a 60-turn coherent dialogue **without** lowering counter-prior accurate-implant rate by > 5 absolute % vs always-on injection | R-5 chat | timing measured with `torch.cuda.synchronize` / MPS equivalent on 60 turns × 5 seeds |
| **H5** | α=0 bit-equality holds for all 3 GPT-2 archs AND for ≥ 3/5 flagship models (Gemma-4-E2B, Qwen3-4B, GLM-4-9B at minimum) under LOPI ON | R-2 smoke | max-abs-diff = 0.0 on 256 tokens × vocab; bf16 tolerance fallback ≤ 1e-4 documented if needed |

## 3. Architecture Specification (User White-Paper, frozen)

For each attention layer ℓ at decoding time-step t, given:

- `V_ctx ∈ R^{B×H×T×Dh}` — native value tensor for context tokens (no bank).
- `M_V ∈ R^{B×H×T×Dh}` — bank-derived value contribution at this layer.
- `Q_t, Q_{t-1} ∈ R^{B×H×Dh}` — query at current and previous decode step.
- `prev_residual_norms[ℓ]` — L2 norm of residual stream at layer ℓ from t-1.
- `mhc_sigma_max` — running mean of mHC's bank-column max-σ from
  `attention_store.py`.

Compute:

1. **Derivative gate (Context Shift Gating).** Decoupling injection from
   topic-shift triggers.

   ```
   ΔQ_t = ‖Q_t − Q_{t-1}‖_2
   γ_t  = sigmoid(k · (ΔQ_t − θ))
   ```

   Frozen hyperparameters: **k = 5.0**, **θ = 0.5**.

2. **Adaptive Layer Gaussian (Adaptive Layer Routing).** Decoupling thinking
   depth from network layer.

   ```
   d_t   = sigmoid(κ · (mean_ℓ ‖res_ℓ^{t-1}‖_2 / Norm_base − 1))
   μ_t   = L · (0.3 + 0.5 · d_t)
   σ_t   = (L / 6) · exp(−β · mhc_sigma_max)
   w(ℓ, t) = exp(−(ℓ − μ_t)^2 / (2 σ_t^2))
   ```

   Frozen hyperparameters: **κ = 2.0**, **β = 2.0**.
   `Norm_base` is calibrated **once per model** as the median of
   layer-mean residual norms over 256 wikitext-2 tokens, with adapter at
   α=0; this calibration is logged but **not tuned per-task**.

3. **Orthogonal Projection (Orthogonal Novelty).** Decoupling novel
   features from redundant features.

   ```
   M_∥   = ((M_V · V_ctx) / (‖V_ctx‖^2 + ε)) · V_ctx        # ε = 1e-6
   M_⊥   = M_V − M_∥
   ```

   Computed per (B, H, T) head/position; reduction along D_h.

4. **Final injection equation.** Replaces the bank value contribution
   inside `attn_native_bank.py` merged-softmax branch (lines 408–413,
   commit 28443537).

   ```
   V_out = V_ctx + γ_t · w(ℓ, t) · M_⊥
   ```

   When `lopi_enabled = False` (global config / per-call override), the
   formula degenerates to `V_out = V_ctx + α · M_V`, restoring the
   pre-Phase-R behavior bit-exactly.

5. **Causality trick.** `prev_residual_norms` and `Q_{t-1}` are cached
   from time-step **t−1** (not t) to keep the forward DAG single-pass.
   At t = 0 (or after a session reset), all gates default to
   `γ_0 = 1`, `w(ℓ, 0) = 1` (i.e. legacy behavior).

## 4. Component-Level Ablation Grid (R-3)

Five LOPI variants, all 3 archs (Residual, HC, mHC), all 7 α, GPT-2 small
+ medium, seeds {0, 1, 2}:

| ID | M_⊥ | Adaptive μ,σ | Derivative γ | Notes |
|---|---|---|---|---|
| A0 | off | uniform w=1 | γ=1 | Baseline (= post-R-pre mHC3 results) |
| A1 | **on** | uniform w=1 | γ=1 | Pure orthogonal projection |
| A2 | **on** | **Gaussian** | γ=1 | + adaptive layer routing |
| A3 | **on** | **Gaussian** | **derivative** | Full Dynamic LOPI |
| A4 | off | **Gaussian** | **derivative** | Control: gating without orthogonal |

Total cells: 5 variants × 3 archs × 7 α × 2 scales × 3 seeds = **630** cells.
Compute estimate: ~3.5h on MPS bf16 (each cell ≈ 20s, parallel-safe).

## 5. Per-Phase PASS Criteria & 3-Strike Rule

### R-2 — Bit-equality smoke
- 3 GPT-2 archs × {LOPI off, LOPI on with α=0}: max-abs-diff = 0.0.
- ≥ 3/5 flagship models bit-equal at α=0 with LOPI on.
- FAIL → debug `lopi.py` short-circuit logic.
- 3 strikes → `git reset --hard f0b64cb9` (R-pre merged commit) + `INCIDENT.md`.

### R-3 — Component ablation
- H1 PASS = A2 or A3 collapses neutral drift to ≤ 0.5 nats on ≥ 5/7 α / 3 archs / 2 scales.
- H2 PASS = A3 mHC lift ≥ +0.30 AND Residual lift ≤ −2.5 at α=1.0 medium.
- H3 PASS = A2 lift > A1 lift by ≥ 0.10 nats with non-overlapping seed stderr.
- 3-strike rollback → R-1 module debug.

### R-3.5 — mHC5 norm-probe replay under A2
- Reproduce mHC5 layer-norm probe with LOPI A2 active.
- PASS = ‖x_L‖/‖x_0‖ injection delta visible above noise floor.
- This phase is **diagnostic only**; failure does not block R-4.

### R-4 — Cross-arch α-safety re-sweep
- 5 flagship models × 7 α × {LOPI off, A3} × seed 0.
- PASS = drift ≤ 0.5 nats on ≥ 5/7 α per model, ≥ 4/5 models.

### R-5 — Q3 adversarial chat with LOPI
- 60 counter-prior + 60 neutral facts × {LOPI off, A3}.
- PASS = accurate-implant ≥ 60% on ≥ 3/5 models AND H4 timing target.

### R-6 — Persistent bank store
- FAISS IVFPQ index over 100k facts + cosine fallback.
- Round-trip bit-equal serialize → load → query.
- p99 query latency ≤ 5 ms on 100k @ d=768.

### R-7 — mHC1.6 full 20k-step finetune (optional)
- If R-3 H3 PASSes only on Gaussian and not on derivative gate, run mHC1.6
  to disambiguate HC ≡ mHC bit-identity at equivalence init.
- 3-strike rollback → skip to R-8 with H3 partial.

### R-8 — arXiv-style writeup + 5 figures
- Single PDF + 5 PNG figures, full data appendix referenced via SHA.

## 6. Frozen Datasets

- **Drift**: NEUTRAL_PROMPTS (existing 6 prompts in `run_mHC3_bank_injection.py`),
  + wikitext-2 32×512 segments (seed=0, sha-locked via `_wikitext2_segments`).
- **Lift**: FALSE_FACTS (existing 5 facts in `run_mHC3_bank_injection.py`).
- **R-4/R-5**: 60-fact counter-prior set from Phase Q v3.2 (SHA frozen at Q-launch).
- **R-3.5**: mHC5 prompts (existing in `reports/cleanroom/mHC5_layer_norms/`).

## 7. Banned Words (per v3.2 red lines)

"honest" / "诚实" / "obviously" / "clearly outperforms"
→ use "preregistered", "strict", "per-protocol", "matched-α PASS by criterion §5".

## 8. Rollback Target

Any 3-strike trigger reverts working tree to commit **`f0b64cb9`**
(R-pre merged head, `stage13-attn-native`).

---

**Frozen by**: KeikaJames
**Date**: 2026-05-04
**Hash of this file at freeze**: to be added in commit message after `git add`.
