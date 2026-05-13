# A Pre-Registration: v0.5 Component Ablation Matrix

**Status**: locked.
**Authored**: 2026-05-06 (post W.6 PREREG, post W.4 verdict, based on v0.4 merged baseline).
**Depends on**: PR #29–#34 must be merged into the v0.4 baseline before this run starts;
the ablation matrix dissects the **full v0.4 stack** with all seven components active in
the control arm.
**Hardware target**: 64 GB unified memory or single CUDA GPU with ≥24 GB. Runs on the
development machine.

---

## 1. Question

W.4 and W.6 establish that the v0.4 memory injection stack moves NLL in the expected
direction on counterfactual prompts. A now asks the falsifiability question:

> Which of the seven architectural components in the v0.4 stack are **necessary** for
> the measured NLL shift? If any component can be switched off without degrading
> performance, it is redundant. If switching off a component destroys the effect, it is
> necessary.

This is the operational definition of *non-redundant design*. A method that retains its
effect after arbitrary ablations is doing something trivial, not precision engineering.

## 2. Hypotheses

Let `A(ablation_id, alpha)` denote the system under test, where `ablation_id` ranges
over `{A1, A2, A3, A4, A5, A6, A7}` (one per component switched off) and the control
is the unablated v0.4 stack. Define for each prompt
`p = (subject, relation, target_true, target_new)`:

- `nll_new(p; A)`  — negative log-likelihood of `target_new` continuation under `A`.

**H_A1 (Bank K pre-RoPE invariance)**:
  Ablating pre-RoPE K (forcing post-RoPE K) increases `nll_new` relative to control.
  `median_p [nll_new(p; A1, alpha=1.0) - nll_new(p; control, alpha=1.0)] > 0`
  with paired Wilcoxon p < 0.01 after Holm correction across the 7 ablations.

**H_A2 (LOPI derivative gate)**:
  Ablating the derivative gate (forcing gate=1 always) increases `nll_new` relative to
  control.
  `median_p [nll_new(p; A2, alpha=1.0) - nll_new(p; control, alpha=1.0)] > 0`
  with paired Wilcoxon p < 0.01 after Holm.

**H_A3 (LOPI sigma-shrink)**:
  Ablating sigma-shrink (forcing eta_sigma=1.0 always) increases `nll_new` relative to
  control.
  `median_p [nll_new(p; A3, alpha=1.0) - nll_new(p; control, alpha=1.0)] > 0`
  with paired Wilcoxon p < 0.01 after Holm.

**H_A4 (SCAR orthogonal projection)**:
  Ablating SCAR's orthogonal projection (using raw delta) increases `nll_new` relative
  to control.
  `median_p [nll_new(p; A4, alpha=1.0) - nll_new(p; control, alpha=1.0)] > 0`
  with paired Wilcoxon p < 0.01 after Holm.

**H_A5 (CAA target-mean steering)**:
  Ablating CAA's target-mean steering (substituting random target vector) increases
  `nll_new` relative to control.
  `median_p [nll_new(p; A5, alpha=1.0) - nll_new(p; control, alpha=1.0)] > 0`
  with paired Wilcoxon p < 0.01 after Holm.

**H_A6 (Bank V-rotation theta)**:
  Ablating V-rotation (forcing theta=0) increases `nll_new` relative to control.
  `median_p [nll_new(p; A6, alpha=1.0) - nll_new(p; control, alpha=1.0)] > 0`
  with paired Wilcoxon p < 0.01 after Holm.

**H_A7 (Injector alpha shielding at alpha=0)**:
  Ablating alpha=0 short-circuit (forcing alpha-multiplied paths even at alpha=0)
  increases `nll_new` relative to control at alpha=1.0 (not alpha=0, where the control
  is bit-equal by design).
  `median_p [nll_new(p; A7, alpha=1.0) - nll_new(p; control, alpha=1.0)] > 0`
  with paired Wilcoxon p < 0.01 after Holm.

**H_A0 (red-line — bit-equality at alpha=0)**:
  At alpha=0, **all 7 ablations** produce bit-equal `nll_new` to the control:
  `max |nll_new(A_i, alpha=0) - nll_new(control, alpha=0)| < 1e-4` for all i in 1..7.
  Any violation aborts the ablation and flags the run.

## 3. Grid

Total cells: **8 arms × 3 models × 3 alphas × 1 seed × 30 prompts = 2,160**.

- **Arms**: `control` (unablated v0.4) + 7 ablation arms `{A1, A2, A3, A4, A5, A6, A7}`.
- **Models**: subset of W.6 model pack — `gpt2-medium`, `Qwen/Qwen2.5-1.5B`,
  `google/gemma-3-1b-it`. Substitution policy inherits W.6.
- **Alphas**: `{0.0, 1.0, 2.0}`. Alpha=0.0 is the bit-equality red-line witness.
- **Seeds**: `{0}` (single seed; full-seed grid is future work).
- **Prompts**: first 30 rows of `counterfact_60.jsonl` (sha256
  `c3e1ac771493452bcb718053b7513cbd49b6dd4d762feddd144b7e2f75fd52a6`); each row supplies
  subject, prompt template, target_true, target_new.

## 4. Ablation implementations

Each ablation switches off **exactly one** component of the v0.4 stack. The control arm
has all components active. Ablations are implemented as follows:

| Ablation | Component | Module path | Off-state implementation |
|---|---|---|---|
| A1 | Bank K pre-RoPE invariance | `deltamemory/memory/attn_native_bank.py` | Force post-RoPE K capture; bank K receives RoPE-rotated keys instead of pre-RoPE keys. |
| A2 | LOPI derivative gate | `deltamemory/memory/lopi.py` | Force `gamma_t = 1.0` always; bypass `derivative_gate` function. |
| A3 | LOPI sigma-shrink | `deltamemory/memory/lopi_profiler.py` | Force `eta_sigma = 1.0` always; bypass adaptive sigma calculation. |
| A4 | SCAR orthogonal projection | `deltamemory/memory/scar_injector.py` | Use raw delta without orthogonal projection; skip `M_perp` computation. |
| A5 | CAA target-mean steering | `deltamemory/memory/caa_injector.py` | Replace `target_mean` with a random unit vector (seed-pinned per layer); bypass learned target computation. |
| A6 | Bank V-rotation theta | `deltamemory/memory/lopi_inject.py` | Force `theta = 0` in ECOR rotation; bypass `max_theta_frac` and tanh calculation. |
| A7 | Injector alpha shielding | injector base context managers | Remove alpha=0 short-circuit; force all alpha-multiplied paths even when alpha=0. |

Each ablation is tagged with `ablation_id` in the output `cells.jsonl`. The control arm
is tagged with `ablation_id = "control"`.

## 5. Injection content

For each prompt the injection bank receives **one** synthetic Fact line:
`Fact: {subject} {relation_phrase} {target_new}.` — relation_phrase is rendered from
the LAMA T-REx template (`experiments/datasets/lama_trex_500.jsonl`) using
`predicate_id == relation`. If no template exists for that relation, fallback to
extracting the phrase from the counterfact row's own `prompt` template per W.6 SMOKE
deviation §1. Drop rate expected ≤ 5%; if it exceeds 10% the run aborts.

## 6. Statistics

- **H_A1–H_A7**: paired Wilcoxon, two-sided, `zero_method="wilcox"`, paired by
  `(seed, prompt_id)` against the control arm at the same alpha. Holm-Bonferroni across
  the 7 ablations (family size = 7), threshold 0.01.
- **Effect size**: median paired diff with 95% bootstrap CI (B=10000, seed=0 for
  bootstrap RNG).
- **H_A0 (bit-equality red-line)**: per-ablation max drift at alpha=0. Any ablation
  with `max |drift| >= 1e-4` is flagged `redline_violation=true`.

**Necessity criterion**: An ablation is deemed **necessary** iff:
  1. Paired Wilcoxon p < 0.01 after Holm correction (survives family-wise control),
  2. Bootstrap 95% CI on median diff excludes 0,
  3. Median diff has the expected sign (positive = ablation degrades performance).

## 7. Red-lines and abort conditions

1. Any ablation arm with `max |nll_new(A_i, alpha=0) - nll_new(control, alpha=0)| >= 1e-4`
   → cell flagged `redline_violation=true`; per-ablation fail of H_A0.
2. relation-template miss > 10% → run aborts before stats.
3. CUDA / MPS OOM on any model → that model is dropped, recorded as substitution.
4. If any ablation arm fails to initialize (e.g. injector does not support the ablation
   mode), that arm is dropped from the grid and recorded in `env.json.dropped_arms`.

## 8. Deliverables

- `cells.jsonl`     — full grid (2,160 rows; one row per cell).
- `cells_smoke.jsonl` — pre-flight on gpt2-medium, 1 seed, 5 prompts, 2 alphas, all arms.
- `summary.json`    — H_A1–H_A7 Wilcoxon + Holm verdicts, necessity flags per ablation.
- `necessity.json`  — per-ablation necessity verdict (bool), median diff, 95% CI, p-value.
- `REPORT.md`       — narrative, no emoji, no colloquial Chinese.
- `env.json`        — env hash, git_commit, prereg_version="A.v1", torch_version,
                     transformers_version, device, dtype, counterfact_sha256,
                     dropped_arms (if any).

## 9. Out of scope

- Multi-seed robustness (single seed only; future work).
- Multi-fact interference (W.8).
- Long-context degradation (W.7).
- Multi-turn override (W.9).
- Ablating combinations of components (2^7 = 128 cells; future work).
- Alpha sweep beyond {0.0, 1.0, 2.0} (future work).

---

End of pre-registration. After this point, no parameter in §3, §6, §7 may change without
recording the change in REPORT §Deviations and bumping `prereg_version` in `env.json`.
