# W-T3.6 — ECOR + U-LOPI Profiler Ablation (combined)

**Status**: round-1 of W-T3 LOPI/ECOR tuning environment, MPS local.
**Git rev**: `57aeeac1` (`feat/v04-w4-caa-runner`)
**Generated**: 2026-05-04 (autopilot)

This deliverable answers two open questions:

1. **"Why is `LOPIConfig.orthogonal` still default `False`, and where is the improved-ECOR ablation table?"**
2. **"Show the U-LOPI profiler effect comparison."**

Both were unanswered after Phase X.7 (ECOR landed but no A/B) and after S-7 (U-LOPI auto vs static was tested on a single model with α∈{0,2,5}).

---

## 1. ECOR operator-level ablation (orthogonal × soft_blend × α)

**Setup** — `experiments/W_T3_6_ecor_op/run.py`

* Model: `Qwen/Qwen2.5-0.5B-Instruct`, MPS bf16
* Captured real `V_ctx` from 8 neutral prompts × 24 transformer layers
* Pooled `M_V` from a fact prompt (mean over fact tokens), broadcast as bank readout
* Grid: ortho ∈ {False, True} × soft_blend ∈ {0.0, 0.25, 0.5, 1.0} × α ∈ {0.5, 1, 2, 4} = **32 cells**
* Redline: `soft_blend=0` ⇒ bit-equal additive (verified, max-abs-diff = 0)

**Per-(ortho, blend) curves over α**

| ortho | soft_blend | α=0.5 rel_pert | α=1 rel_pert | α=2 rel_pert | α=4 rel_pert | α=4 norm_ratio | α=4 cos |
|:-----:|:---------:|---:|---:|---:|---:|---:|---:|
| False | 0.00 (additive) | 2.21 | 4.43 | 8.85 | **17.70** | **17.99** | 0.32 |
| False | 0.25 | 1.77 | 3.48 | 6.82 | 13.46 | 13.75 | 0.33 |
| False | 0.50 | 1.33 | 2.54 | 4.80 | 9.22 | 9.51 | 0.34 |
| False | 1.00 (pure ECOR) | 0.45 | 0.70 | 0.85 | **0.87** | 1.10 | 0.64 |
| True  | 0.00 (additive) | 2.18 | 4.37 | 8.74 | 17.47 | 17.59 | 0.18 |
| True  | 0.25 | 1.76 | 3.46 | 6.77 | 13.34 | 13.41 | 0.18 |
| True  | 0.50 | 1.33 | 2.55 | 4.80 | 9.18 | 9.24 | 0.19 |
| True  | 1.00 (pure ECOR) | 0.48 | 0.78 | 0.96 | **1.00** | **1.0006** | 0.50 |

**Reading**

* `rel_perturb` = ‖V_out−V_ctx‖/‖V_ctx‖. The additive operator at α=4 perturbs V_ctx by **17.7× its own norm**. That is not "injection", that is "annihilation". Pure ECOR caps it at ≤1.0.
* `norm_ratio` = ‖V_out‖/‖V_ctx‖. ECOR with ortho=True hits **1.0000 ± 0.001** at every α — energy preservation is empirically exact, as the math promised.
* `cos(V_out, V_ctx)` — additive at α=2 already drags direction down to 0.38; ECOR keeps it at 0.54–0.66.
* `m_perp_ratio` = ‖M_⊥‖/‖M_V‖ = **0.908** when ortho=True. ~9% of the bank readout was already aligned with V_ctx and got thrown away by the projection. (When ortho=False, M_⊥ ≡ M_V ⇒ ratio 1.0.)
* `soft_blend ∈ (0, 1)` is a smooth interpolation: 0.5 halves the perturbation magnitude vs additive at every α.

**Why this matters for the default-flip question**

The current production default `soft_blend=0` (additive) is mathematically equivalent to *injecting α-weighted bank V directly into the V stream*. At α=4 (which W.4 uses on its top tier), this means scaling the residual by ~18× before the model sees it. That is why we observe runaway drift in W.2 component sweeps: the operator itself is unstable.

**Caveat — what this does NOT prove**

This is operator-level on captured tensors. It demonstrates the geometric behaviour of the four operators but **does not** measure downstream NLL / generation drift end-to-end, because `lopi_inject` is not yet wired into `attn_native_bank.py`'s forward path. End-to-end NLL A/B is still a code-change away (planned as W-T3 round 2).

---

## 2. U-LOPI profiler effect: static vs auto, two models

**Setup** — `scripts/run_ulopi_xarch_smoke.py`, MPS bf16, 8 prompts × 3 seeds × 4 α

| α | model | static_nll | auto_nll | drift_static | drift_auto | Δ(auto−static) |
|--:|:-----|---:|---:|---:|---:|---:|
| 0.00 | Qwen2.5-0.5B-Instruct | +1.9035 | +1.9035 | +0.0000 | +0.0000 | 0.0000 (bit-equal) |
| 1.00 | Qwen2.5-0.5B-Instruct | +4.1591 | +3.5993 | +2.2557 | +1.6958 | **−0.5599** |
| 2.00 | Qwen2.5-0.5B-Instruct | +5.0455 | +3.2156 | +3.1420 | +1.3121 | **−1.8299** |
| 4.00 | Qwen2.5-0.5B-Instruct | +6.3299 | +6.1043 | +4.4264 | +4.2009 | **−0.2255** |
| 0.00 | Qwen2.5-1.5B          | +1.5669 | +1.5669 | +0.0000 | +0.0000 | 0.0000 (bit-equal) |
| 1.00 | Qwen2.5-1.5B          | +3.7147 | +2.9171 | +2.1477 | +1.3502 | **−0.7976** |
| 2.00 | Qwen2.5-1.5B          | +3.7389 | +7.8773 | +2.1720 | +6.3104 | **+4.1384** |
| 4.00 | Qwen2.5-1.5B          | +3.2592 | +10.7191 | +1.6922 | +9.1522 | **+7.4600** |

**Per-model verdicts**

* **Qwen2.5-0.5B-Instruct** (24 layers): `AUTO < STATIC` — auto profiler beats static `norm_base=10` at every α∈{1,2,4}. Mean improvement −0.87 nats. *This contradicts S-7's `STATIC≤AUTO` verdict, which had used α=5 — see explanation below.*
* **Qwen2.5-1.5B** (28 layers): `STATIC ≤ AUTO` — auto loses badly at α≥2 (+4.1, +7.5 nats). Static profile is robust; auto profile blows up.

**Why S-7 (the only previous evidence) was misleading**

S-7 used α∈{0,2,5} on Qwen2.5-0.5B-Instruct. At α=5 the auto profile is past its useful range and underperforms; the verdict was driven by that single high-α cell. Re-running with α∈{0,1,2,4} flips the verdict on the same model. Lesson: **one α grid is not a rebuttal of an architectural mechanism**.

**Why does auto win on 0.5B but lose on 1.5B?**

The auto profile picks `μ_arch = 5` for both models (eta_sigma = 0.7 for both), but layer-count differs (24 vs 28) and 1.5B has substantially different residual-norm scale. The fixed `norm_base = 10` static path happens to be closer to 1.5B's true scale. This is exactly the per-arch calibration failure mode flagged in `docs/theory/lopi.md`'s critique table: the U-LOPI `Norm_base` constant is Gemma-tuned, and even after Z-score replacement the µ_arch picker is unstable on small profile corpora (N=10 prompts).

**Concrete tuning hypothesis (W-T3 round 2)**

1. The auto profiler must scale its α-injection by the residual-norm Z-score of the target model, not just pick μ_arch from argmax σ_base. Currently the auto path inherits the same α as static, which is what causes the 1.5B blow-up.
2. Alternatively, the auto profile must be re-run on a per-model basis with N≥100 prompts (current N=10 too small). 1.5B's argmax tiebreak almost certainly differs from 0.5B's despite both reporting `μ_arch=5`.

---

## 3. Combined verdicts

| Question | Verdict | Evidence |
|---|---|---|
| Is `soft_blend=0` (additive) safe at production α? | **No** — at α=2 it perturbs V_ctx by 8.85×, at α=4 by 17.7×. Operator is intrinsically unstable. | `experiments/W_T3_6_ecor_op/cells.jsonl` |
| Is ECOR pure rotation (soft_blend=1) energy-preserving in practice? | **Yes** — norm_ratio = 1.0000 at every α with ortho=True. | same |
| Should the production default flip to soft_blend > 0? | **Cannot be flipped yet.** Must first wire ECOR through `attn_native_bank.py` and run end-to-end NLL A/B (W-T3 round 2). | — |
| Does the U-LOPI auto profiler beat the static `norm_base=10` path? | **Model-dependent.** Wins on 0.5B, loses on 1.5B. The `μ_arch`/`norm_base` selection is the failure mode. | `experiments/W_T3_6_ulopi_profiler/qwen0{5,15}/` |
| Was S-7's `STATIC≤AUTO` verdict reliable? | **No** — α grid choice ({0,2,5}) was not representative; new α∈{0,1,2,4} flips the verdict on the same model. | this REPORT |

## 4. Next actions (W-T3 round 2, owed)

1. **Wire ECOR into `lopi.py::apply_lopi_to_bank`** behind a `LOPIConfig.use_ecor` flag (default False, preserve α=0 bit-equal). Add 2 unit tests (bit-equal at flag=False, end-to-end forward smoke at flag=True).
2. **Run end-to-end NLL A/B** of additive vs ECOR (soft_blend ∈ {0.5, 1.0}) on Qwen2.5-0.5B-Instruct + Qwen2.5-1.5B + Qwen2.5-3B at α ∈ {0.5, 1, 2, 4} × 3 seeds × 30-prompt neutral set.
3. **Per-model `μ_arch` audit**: rerun U-LOPI profiler with N=100 prompts on both Qwen2.5 sizes and on Gemma-4-E2B; report `μ_arch` stability with bootstrap.
4. **Decision**: if ECOR end-to-end A/B shows ≥30% drift reduction at α≥2 on ≥2 models with α=0 redline preserved, propose flipping the default to `soft_blend=0.5` in W-T3 round 3 (paired Wilcoxon required, p<0.01 vs additive baseline).

## 5. Files

* `experiments/W_T3_6_ecor_op/run.py` — operator-level ablation script
* `experiments/W_T3_6_ecor_op/cells.jsonl` — 32 cells × 4 metrics
* `experiments/W_T3_6_ecor_op/env.json` — env, redline witness
* `experiments/W_T3_6_ecor_op/REPORT.md` — auto-generated table
* `experiments/W_T3_6_ulopi_profiler/qwen05/` — 0.5B drift table + per-cell JSONs
* `experiments/W_T3_6_ulopi_profiler/qwen15/` — 1.5B drift table + per-cell JSONs
* `experiments/W_T3_6_ulopi_profiler/qwen0{5,15}.log` — full logs
