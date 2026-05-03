# Phase mHC Preregistration v1 — α-Safe External Memory via Manifold-Constrained Hyper-Connections

**Status**: frozen v1 (sha-pinned upon commit). Any change after this commit MUST appear as an append-only amendment block at the bottom, with new sha and date.

**Frozen date**: 2026-05-04 (commit-time, see git log).

**Scope**: Whether DeepSeek's *Manifold-Constrained Hyper-Connections* (mHC, arXiv:2512.24880) — proven for **training-time** spectral stability — extends to **inference-time external KV-bank α-injection** robustness in the DeltaMemory framework. This is a NEW research question; the paper does not address external memory injection.

**Red lines (inherited from v3.1, extended)**:
- LLM weights are frozen. mHC mixing matrix `C` is an **architecture-internal** trainable parameter, **not** the LLM's `W_q/W_k/W_v/W_o`.
- α=0 must be bit-equal to the corresponding base architecture (logits-equivalent conversion verified per H6).
- DeltaMemory injects only into attention K/V via the bank; it does not enter the token residual stream directly.

---

## 1. Theoretical setup (as preregistered prediction, not as proof)

### 1.1 Residual baseline — "poison amplification"

For a standard Transformer, the post-injection layer-ℓ attention output at one token is approximately

```
O_inject ≈ β · V + (1 − β) · α · M_V
```

where `β` is the attention mass on native V and `(1 − β)` is the mass on bank slots.

Residual update: `x_{ℓ+1} = x_ℓ + O_inject`. Iterating across L layers with arbitrary FFN/non-linearities, the injected component `(1 − β) α M_V` enters an unconstrained additive accumulator and may amplify (worst-case exponentially in L through gain > 1 layers).

This predicts a narrow safe-α band at inference: in our v3.1 experiments, Qwen3 / Llama / GLM-4 require α ≤ 0.05 to avoid logit collapse; Gemma-4 tolerates α = 1.0 only because of v_norm.

### 1.2 mHC architecture — Sinkhorn-Knopp spectral cage

mHC replaces additive cross-layer mixing with a learned non-negative routing matrix `C` projected to the Birkhoff polytope (doubly-stochastic) via Sinkhorn-Knopp iteration. By Perron-Frobenius, every non-negative doubly-stochastic matrix has spectral radius `σ_max(C) = 1`. Therefore for the routing-only operator iterated k times:

```
‖C^k E‖₂ ≤ ‖C‖₂^k · ‖E‖₂ = 1 · ‖E‖₂
```

i.e. the routing operator does **not** amplify any injected disturbance `E = α M_V` in the L₂ norm. The row-sum-and-column-sum-=-1 property additionally makes each routing step a **convex combination** across token/layer slots.

### 1.3 Mathematical caveat (preregistered, must NOT be over-claimed)

The bound `‖C^k E‖₂ ≤ ‖E‖₂` applies to the **routing operator only**. A full Transformer forward pass interleaves `C` with attention softmax / FFN GeLU/SiLU non-linearities; those sub-modules can themselves amplify. Therefore:

- We do **not** claim "mHC makes injection energy provably non-amplifying end-to-end."
- We **do** claim, as an empirical preregistered hypothesis, that mHC produces a **substantially wider safe-α band** at inference than the residual baseline.
- The user-quoted phrase "no matter how deep the network ... never exceed the initial size" holds strictly only for the routing operator, not the full forward. This caveat is preregistered.

---

## 2. Falsifiable hypotheses (Holm-Bonferroni corrected, family m=6)

Each H reports: point estimate, paired Wilcoxon signed-rank p, bootstrap (1000-resample) 95% CI. Pass = corrected p < 0.05 AND effect direction matches.

- **H1 (poison amplification, residual)**. There exists `α* < 1.0` such that Residual GPT-2's Wikitext-2 NLL increases by **≥ 3 nats** vs no-bank baseline at α ≥ α*.
- **H2 (spectral shield, mHC)**. Across `α ∈ [0, 5]`, mHC GPT-2's Wikitext-2 NLL increase vs no-bank stays **≤ 0.5 nats** at every preregistered α.
- **H3 (HC sanity, third arm)**. Unconstrained HC GPT-2's NLL also crashes at some `α < α*(residual)` — i.e. HC is not automatically safe; the safety comes specifically from the doubly-stochastic constraint, not from the multi-channel structure alone. (Failure mode: if HC turns out to be already safe, we record this honestly as a partial null — it would weaken but not reverse the headline.)
- **H4 (counter-prior lift × α)**. On FALSE-fact prompts (counter-prior bank entries), mHC retains monotonically increasing log-prob lift across α ∈ {1, 2, 5}, while residual lift collapses (≤ 0) past α*. Paired by fact-id across architectures.
- **H5 (layer-wise energy, paired visualization — headline #2)**. At a fixed high α (main: **α = 1.5**; ablation: α ∈ {0.5, 1, 2, 5}), under the **same prompt set and same seeds**, the per-layer Frobenius norm trajectory `‖x_ℓ‖_F` shows:
  - Residual GPT-2: exponential growth (near-linear on log-y).
  - mHC GPT-2: plateau / sub-linear (near-flat on log-y).
  - HC GPT-2: third arm, expected somewhere in between or also exponential.
  Quantitative gate: `‖x_L‖_F / ‖x_0‖_F` for mHC is at least **10×** smaller than for Residual at α=1.5, with non-overlapping bootstrap 95% CI across 5 seeds × 32 prompts.
- **H6 (α=0 bit-equal regression)**. All three architectures at α=0 reproduce no-bank logits with `max_abs_diff < 1e-5` over a fixed 1024-token × full-vocab probe (MarcoDotIO logits-equivalent conversion claim). Failure here halts the entire phase.

**Decision gate** (between mHC2 and mHC3): of {H1, H2, H5}, at least **2 of 3** must show the predicted direction (uncorrected p < 0.05) on the pure-perturbation Phase mHC2 dataset before Phase mHC3 (bank-injection) launches. This conserves GB10 budget if the architecture-only spectral shield doesn't manifest.

---

## 3. Datasets, splits, sha pins

- **Architecture-perturbation NLL (Phase mHC2)**: Wikitext-2 `validation` split, fixed segmentation 1024 tokens × 1024 segments, seed-locked sampling. Splits committed under `eval/wikitext2_mHC2_split.json` with sha256.
- **DeltaMemory bank (Phase mHC3+)**: reuses v3.1 LAMA + ConceptNet train/dev split (existing sha-locked). Test set untouched.
- **Counter-prior FALSE-facts (Phase mHC4)**: reuses the 5 FALSE facts from `transcripts/v31_intervention/*FALSE*/` plus 25 newly written counter-prior facts (committed in `eval/false_facts_mHC4.jsonl`, sha256 in amendment log).
- **Layer-norm probe prompts (Phase mHC5)**: Wikitext-2 val, 32 segments × 5 seeds, sha256 in amendment.

---

## 4. Architectures, models, hyperparameter grid

- **Base**: GPT-2 small (768 d, 12 layer). Mac MPS reproducible. (D1 decision: GPT-2 medium 1024 d / 24 layer is added in mHC6 only.)
- **Three arms**:
  1. Residual GPT-2 (HF `gpt2`, frozen).
  2. Unconstrained HC GPT-2 (Sinkhorn-Knopp **disabled**, multi-channel mixing free).
  3. mHC GPT-2 (Sinkhorn-Knopp **enabled** per arXiv:2512.24880).
- **α grid**: {0, 0.05, 0.1, 0.5, 1, 1.5, 2, 5, 10}. (1.5 is the mHC5 main condition.)
- **Sinkhorn iteration count**: k ∈ {3, 5, 10} sensitivity scan in mHC1.5.
- **Seeds**: 5 seeds per condition (0–4).
- **Numerical**: bf16 default; α ≥ 5 falls back to fp32 if NaN/Inf detected; fallback events logged.

---

## 5. Statistical procedure

- Paired comparisons (same fact, same prompt, same seed) across architectures.
- Wilcoxon signed-rank for each H1–H5.
- Holm-Bonferroni correction with family m = 6 (H1–H6).
- Bootstrap 1000-resample 95% CI for all reported point estimates.
- Effect size: Cohen's d for continuous outcomes, log-prob lift in nats for H4.
- Pre-specified decision rule: corrected p < 0.05 AND effect-sign match → "supported"; otherwise "not supported" (no soft language).

---

## 6. Stop / abort criteria

- H6 (α=0 bit-equal) fails on any arm → STOP. Fix the open-source mHC implementation first.
- mHC2 decision gate fails (< 2 of 3 of H1/H2/H5) → STOP at end of mHC2; mHC3+ are not run; report null result honestly.
- bf16 → fp32 fallback rate > 50% at any α → mark that α band as "numerical limit reached"; do not extend grid past it.

---

## 7. What we explicitly do NOT claim

- We do not claim mHC is universally inference-stable; only stable under the specific bank-injection setup preregistered here.
- We do not claim the mHC paper's training-stability numbers transfer; that's a separate body of work and we cite, not borrow.
- We do not claim Qwen3 / Llama retrofit will work without retraining (mHC6.2 is a **predicted-failure** sanity probe, not a positive claim).

---

## 8. Related work scan (5 references, anchored 2026-05-04)

| # | Citation | Relation to this work |
|---|---|---|
| 1 | DeepSeek, "Manifold-Constrained Hyper-Connections", arXiv:2512.24880 (2025-12 / 2026-01). | Origin of Sinkhorn-Knopp doubly-stochastic mixing. We extend its training-stability framing to **inference-time external memory injection**. |
| 2 | ByteDance, "Hyper-Connections", 2024. | Unconstrained predecessor; provides the third arm in our 3-arm design. Documented 3000× signal amplification — the failure mode mHC repairs and we directly probe via H3. |
| 3 | Sinkhorn & Knopp, "Concerning nonnegative matrices and doubly stochastic matrices", Pacific J. Math (1967). | Mathematical foundation for the σ_max=1 bound used in §1.2. |
| 4 | Meng et al., "Locating and Editing Factual Associations in GPT" (ROME, 2022) / "Mass-Editing Memory in a Transformer" (MEMIT, 2022). | Weight-editing baselines, included in v3.1 as B3 only. Different red-line regime (modify W) — explicitly **not** our method. |
| 5 | Khattab et al., "Demonstrate-Search-Predict" / Lewis et al., "Retrieval-Augmented Generation" (RAG, 2020). | External-memory baseline that does not modify the residual stream's attention path. v3.1 B2; orthogonal to mHC's spectral framing — RAG injects via prompt context, mHC's safety claim is at the attention K/V slot level. |

---

## 9. Amendment log

(append-only; each entry must include date, commit sha, reason, and updated section reference)

- v1 frozen at first commit of this file.

---

## 10. Reproducibility checklist

- [ ] All splits sha256-pinned and committed.
- [ ] α grid + seed list committed in YAML config.
- [ ] mHC1 conversion checkpoint committed (or fetched via documented HF URL).
- [ ] H1–H6 evaluation code is `scripts/run_mHC_*` with `--config <yaml>` only.
- [ ] One-shot `repro_mHC.sh` covers Phases mHC2–mHC5 on Mac MPS.
