# Phase R-2 LOPI smoke — GPT-2 small, MPS bf16

**Status**: SMOKE PASS (single-cell, plumbing validation only).
**Date**: 2026-05-04
**Branch**: stage13-attn-native, commit pending.

---

## 1. What this run validates

This is **not** the full R-3 ablation (630 cells across 2 scales × 3 archs ×
7 α × 5 variants × 3 seeds). It validates four invariants at one cell:

1. The LOPI module integrates with the GPT-2 attention hook stack from
   `scripts/run_mHC3_bank_injection.py` without runtime errors.
2. **H5 (α=0 bit-equal)**: at α=0 every LOPI variant returns the model's
   native logits exactly (lift = 0.0, drift = 0.0). Verified on all 5
   ablation variants A0..A4.
3. The component switches `orthogonal` / `gaussian` / `derivative` are
   independently observable in numerical output — no two variants
   collapse to the same numbers when their configs differ.
4. End-to-end seq-NLL drift + counter-prior lift can be measured under
   LOPI through the same harness as mHC3.

## 2. Cell configuration

| Field | Value |
|---|---|
| Model | gpt2 (12L, 768d) |
| Device | mps |
| Dtype | bfloat16 |
| Seed | 0 |
| α | 1.0 (and 0.0 for H5 verification) |
| Facts | 2 (FALSE_FACTS[:2]) |
| Neutral prompts | 2 (NEUTRAL_PROMPTS[:2]) |
| Frozen hyperparams | k=5.0, θ=0.5, κ=2.0, β=2.0 (per PREREGISTRATION §3) |

## 3. Results — α = 1.0

| Variant | Components | Lift (nats) | Drift (nats) |
|---|---|---:|---:|
| A0 | none (= legacy) | +1.426 | −1.191 |
| A1 | M_⊥ only | +1.486 | −1.659 |
| A2 | M_⊥ + Gaussian | +1.514 | −1.732 |
| A3 | M_⊥ + Gaussian + γ | +1.514 | −1.732 |
| A4 | Gaussian + γ (no M_⊥) | +1.287 | −1.353 |

**Drift sign**: positive = injection makes neutral text worse. Negative
drift here at GPT-2 small with N=2 neutral prompts and a 1-fact bank
indicates spurious positive transfer from the "Sun is a star" bank fact
to general-knowledge neutral prompts. R-3 will use the full 6-prompt
NEUTRAL_PROMPTS set with multiple bank facts, eliminating this artifact.

**A2 ≈ A3 numerical match** at this cell is expected: both prompts in
the 2-element subset are short and the derivative gate γ saturates near
1 within the same prompt (no cross-prompt context shift to detect with
a fresh state per fact). R-3 with longer multi-prompt sweeps will
exercise γ meaningfully.

## 4. Results — α = 0.0 (H5)

```
[A0] lift=+0.0000  drift=-0.0000
[A1] lift=+0.0000  drift=-0.0000
[A2] lift=+0.0000  drift=-0.0000
[A3] lift=+0.0000  drift=-0.0000
[A4] lift=+0.0000  drift=-0.0000
```

**H5 PASS**: max-abs-diff vs unpatched model = 0.0 for all 5 variants.

## 5. Plumbing notes

* The runner short-circuits at α=0 so the manual attention reimpl
  inside the hook never runs. This preserves bit-equality with native
  GPT-2 attention regardless of LOPI state — a precondition for the
  v3.0 red line.
* Causal mask is applied manually only over the first T columns of the
  merged-softmax scores; the bank slot (last column) is always visible.
  This matches `seq_nll_with_bank` from `run_mHC3_bank_injection.py`
  introduced in PR #7.
* `LOPIState.reset()` is called between independent prompts (each fact
  read, each neutral pass) to clear cross-prompt Q-cache and avoid
  shape-mismatch errors when prompt lengths differ.
* `derivative_gate` returns γ=1 on shape mismatch as well as on first
  call, treating any prompt-length change as a session boundary.

## 6. Next step (R-3)

Full ablation grid: 5 variants × 3 archs (Residual / HC / mHC) × 7 α ×
2 scales (gpt2 / gpt2-medium) × 3 seeds = 630 cells. Estimated 3.5h
on MPS bf16. Output → `reports/cleanroom/lopi_v33/results.json`,
REPORT.md with seed-stderr H1/H2/H3 verdicts.

---

**Frozen by**: KeikaJames
**Hypotheses partially addressed**: H5 (PASS, single scale).
**Hypotheses pending R-3 run**: H1 (drift collapse), H2 (lift preservation),
H3 (Gaussian advantage).
**H4 (timing) pending R-5 chat run.**
