# W.3 — Methodology Decision Gate

**Date**: 2026-05-04
**Chief reviewer**: project lead (this assistant)
**Inputs**:

- `experiments/W1_mhc_localize/REPORT.md` — verdict **FAIL** (drift floor 2.46–2.70 nats vs 0.5 threshold)
- `experiments/W2_lopi_dissect/REPORT.md` — verdict **Q1=FAIL · Q2=FAIL · Q3=FAIL** under per-model strict criteria

This document executes the decision gate defined in `plan.md` Phase W.3.

---

## 1. Restated decision rule

From `plan.md` (verbatim):

> If W.2.4 fail (LOPI 没救), evaluate replacement candidates by priority:
>   C1: V-cap only · C2: ActAdd / CAA-style direction injection ·
>   C3: Memorizing-Transformers-lite · C4: Sinkformer-attention
>
> Decision tree:
>   * W.2 + W.4 都 fail → 主线降为 "mHC shield + V-scale", LOPI 进 ablation flag
>   * W.4 显著优于 LOPI → 主线切到 CAA-injection, LOPI 进 legacy
>   * 都 PASS → 写 LOPI vs CAA 对比章节, 用户裁定

W.4 has not yet run, so only the **left-half of the tree** is decidable today. The
ruling below freezes the *frame* under which W.4 will be evaluated.

---

## 2. What the data forces us to admit

### 2.1 mHC shield (Phase W.1)

| Property | Result | Implication |
|---|---|---|
| Column-cap actually clips bank columns | yes (DH1 OK) | implementation is correct |
| Drift floor at α≥2 across 5 dense models | 2.46–2.70 nats | shield does not suppress *content* drift |
| Reduction relative to no-shield | 44–73 % | non-trivial but does not reach the 0.5-nat threshold |
| α=0 bit-equal | preserved | red-line intact |

**Finding**: mHC shield is a real spectral regularizer but it operates on *attention
mass*, not *injected V-energy direction*. Without a complementary direction-aware
mechanism (V-scale + an orthogonality / projection step), it cannot bring drift
under the threshold defined by the 30-prompt gold-neutral set.

### 2.2 LOPI three components (Phase W.2)

| Component | Operating regime (α∈[0.5,4]) | Diagnosis |
|---|---|---|
| **M⊥ projection** | hurts on 3/3 models | over-aggressive; cuts useful aligned signal at low α |
| **Gaussian layer-gate** | helps GPT-2 only; hurts Qwen 0.5B | centering is offset 4–5 layers from `μ_arch` (likely an indexing bug, see W.2 §"Fix 1") |
| **Derivative gate γ_t** | numerically dead (always 1.0) | `LOPIState` is reset per prompt → `Q_prev=None` → γ_t falls back to 1.0 |

**Finding**: in the production α regime, LOPI reduces to its raw bank-injection
baseline — the three "smart" components either no-op (γ_t), focus on the wrong
layers (Gaussian), or actively damage at low α (M⊥). The α-flip benefit at α=32
is real but only on GPT-2 medium and only at an α range nobody ships.

### 2.3 Joint reading

mHC and LOPI as currently shipped solve two different sub-problems each
**partially** and neither produces a configuration that meets the v0.4
preregistered drift threshold on a single non-Gemma model. Calling them
"production-ready memory injectors" was wrong; calling them "load-bearing
ablation knobs" is correct.

---

## 3. Ruling

### 3.1 Status changes (effective this commit)

| Mechanism | Before W.3 | After W.3 |
|---|---|---|
| **mHC column-cap shield** | main-line spectral regularizer | retained as **opt-in** ablation flag; default OFF on non-Gemma until V-scale + per-arch κ are jointly retuned (W-T1) |
| **LOPI (M⊥ + Gaussian + γ_t)** | main-line injector | retained as **legacy/ablation flag**; full LOPI is no longer the default for new experiments |
| **V-scale RMS cap** | optional safety knob | **promoted** to default-on for all non-Gemma models (R-7 already shows it is the load-bearing piece) |
| **CAA-injector** | candidate (Phase X.3 module) | **promoted to main-line candidate**; W.4 will run its first proper sweep |
| **ECOR** | new opt-in candidate (Phase X.7) | unchanged; A/B vs additive deferred to W-T3.6 |

### 3.2 Default configuration for downstream runs (W.4 onward)

- Bank: `AttnNativePatcher` (unchanged)
- V-scale: `rms_cap=0.5` ON for all non-Gemma; OFF for Gemma (already has v_norm)
- mHC shield: OFF by default; available as `--mhc-shield κ=κ*` for ablations
- LOPI: OFF by default; available as `--lopi {ortho|gauss|deriv|full|gauss_deriv}` for ablations
- CAA: ON by default once W.4 lands; α swept on the same 7-point ladder
- ECOR: OFF; `--ecor` available for A/B
- All defaults are keyed off the model identifier in the runner, never global

### 3.3 What will *not* be done

- **No "fixing" of LOPI before W.4.** The Gaussian centering bug, derivative-gate
  state-carry, and M⊥ α-adaptive scaling are documented in `experiments/W2_lopi_dissect/REPORT.md`
  §"What W-T3 Should Fix Next". Patching them now would entangle the W.4 baseline
  comparison with the LOPI rescue. W-T3 owns those fixes.
- **No abandonment of LOPI components.** They remain in the codebase, tested, and
  ablation-accessible. The legacy label is operational, not historical.
- **No quiet downgrade.** Both W.1 and W.2 reports retain their FAIL verdicts in
  the public report; this DECISION.md is the ruling, not a rewrite.

---

## 4. W.4 pre-conditions (must be true before W.4 runs)

1. ✅ `deltamemory/memory/caa_injector.py` exists and passes its 7 unit tests
2. ✅ X.3 smoke under `experiments/X3_caa_smoke/` reproduces α=0 bit-equality
3. ✅ Real LAMA T-REx (500) and ConceptNet (500) datasets locked (commit `5d044870`)
4. ☐ W.4 PREREG.md committed before any cells.jsonl row is written
5. ☐ W.4 grid uses the same 5 dense models as W.1 + W.2 (Gemma-4-E2B,
   Qwen2.5-0.5B, Qwen2.5-1.5B used as Llama-1B substitute, Qwen3-4B if loadable,
   GLM-4-9B if loadable) — exact list locked in PREREG
6. ☐ Three arms: `none` / `LOPI-default-from-W.2` / `CAA`. LOPI uses the *current
   shipped defaults* (not W-T3 fixes) so the comparison is honest
7. ☐ Paired Wilcoxon with Holm–Bonferroni across (model × α) cells; p-threshold
   0.01 for "significantly better"

If any of (4)–(7) is not true at the moment cells.jsonl is opened for writing,
the run is aborted and re-prereg'd.

---

## 5. Decision tree update (post-W.4)

The decision tree from `plan.md` is **operational** with this addition:

> If W.4 CAA shows **non-significant** improvement over the LOPI shipped default
> (Holm-corrected p ≥ 0.01) on ≥3/5 dense models, the main-line is downgraded to
> "V-scale + raw bank injection" and W-T3 is opened to either fix LOPI or retire
> it permanently. Reviewers should not interpret this as a research dead-end —
> the synthetic dictionary recall task in W.13 is the cleanest test of the bank
> mechanism itself and is independent of injector choice.

---

## 6. Carry-forward consequences

| Subsequent phase | Change |
|---|---|
| W.5 (MoE) | per-expert cap is still required for spectral safety even with shield default OFF; W.5 runs the full MoE matrix with V-scale ON and shield as ablation only |
| W.6 (Counter-prior Pareto) | switch the "method" axis to {none, V-scale, V-scale + CAA, V-scale + LOPI, V-scale + CAA + mHC} — five arms instead of the original five |
| W.10 (vs baselines) | DM-best is now defined post-W.4 (likely V-scale + CAA), not "mHC + LOPI" |
| W.12 (full ablation) | 16-cell grid retains all 4 mechanisms (mHC / LOPI / V-scale / CAA); marginals will show whether mHC and LOPI deserve to remain in the codebase at all |
| W.14 (standard bench) | DM-best frozen *after* W.6 completes; W.14 runs only that frozen recipe |
| Z.1 (bilingual README) | the "Math at a glance" section drops LOPI to a sub-bullet under "ablation knobs" and promotes V-scale + CAA to first-class entries |

---

## 7. Honesty register (for the final paper)

This decision will be reported verbatim in `experiments/REPORT_v04.md` under a
section titled *"What we shipped, what failed, what replaced it"*. Specifically:

- mHC shield (DeepSeek 2026 paper-inspired column cap) **failed** the v0.4 drift
  threshold on 4/5 dense architectures.
- LOPI (orthogonal projection + Gaussian gate + derivative gate) **failed** all
  three of its component-attribution preregistered hypotheses on 3/3 models in
  the production α regime.
- The recovery path is not "fix LOPI"; it is "test the simpler CAA baseline and
  let the data choose". This is the methodology Y the project committed to in
  Phase U.5 (`docs/theory/critique.md`).

No emoji, no hedging, no headline reframing. The numbers are what they are.

---

## 8. Sign-off

This document is the ruling. Any subsequent commit that contradicts it must
explicitly cite this file and produce new evidence (new W.* cells.jsonl rows
+ aggregate verdicts) sufficient to override.

— W.3 closed.
