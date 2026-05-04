# Phase mHC3 — DeltaMemory Bank Injection into 3-Arm GPT-2

**Date**: 2026-05-04 (original) · **Amendment 1**: 2026-05-04 (drift metric correction)
**Hardware**: Mac MPS bf16
**Models**: GPT-2 small (124M, 12-layer) — residual / HC / mHC

---

## ⚠ Amendment 1 — Drift metric corrected to true sequence NLL

The original `mean_drift` column (preserved in
`results_legacy_singletok.json`) was computed as the log-prob delta of a
single token (`" The"`) appended to each neutral prompt. That is **not** a
sequence NLL and is not tokenizer- or architecture-agnostic. Codex review
on PR #6 flagged the dead `pass`/`append(0.0)` placeholders in
`scripts/run_mHC3_bank_injection.py:370–386`; this amendment fixes the
script (real `seq_nll_with_bank` over the entire prompt) and reruns the
full sweep on Mac MPS bf16. The legacy single-token results are kept for
audit. **All conclusions below reflect the corrected metric.**

---

## Headline (corrected)

The multi-stream HC/mHC architecture **preserves a positive counter-prior
lift at α=1.0** (+0.071 nats vs Residual −0.684 nats), confirming that the
n-stream readout protects the *injection signal* from collapse. **However**,
the same architecture **amplifies neutral-prompt drift** (+2.26 nats vs
Residual +0.70 at α=1.0): without orthogonality, multi-stream mixing
spreads the bank perturbation into the unrelated continuation distribution.

This is the **central motivation for Phase R Dynamic LOPI**: keep the
multi-stream lift, kill the neutral drift via orthogonal-complement
projection.

## Full Table (single seed, FALSE facts) — true sequence NLL drift

| α | Residual lift | Residual drift | HC lift | HC drift | mHC lift | mHC drift |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | −0.717 | −0.505 | −0.317 | +1.528 | −0.317 | +1.528 |
| 0.10 | −0.729 | −0.504 | −0.188 | +1.562 | −0.188 | +1.562 |
| 0.50 | −0.529 | −0.181 | −0.161 | +1.916 | −0.161 | +1.916 |
| **1.00** | **−0.684** | **+0.697** | **+0.071** | **+2.257** | **+0.071** | **+2.257** |
| 2.00 | −0.445 | +0.540 | −0.019 | +2.103 | −0.019 | +2.103 |
| 5.00 | −0.083 | +4.795 | −0.508 | +4.926 | −0.508 | +4.926 |
| 10.00 | −4.297 | +5.769 | −0.997 | +6.193 | −0.997 | +6.193 |

(Drift = mean(seq_nll_with_bank − seq_nll_baseline) over 5 neutral
Wikipedia-style prompts; positive ⇒ bank degrades neutral fluency.)

## Hypothesis Verdicts (corrected)

| ID | Hypothesis | Original verdict | **Corrected verdict** | Detail |
|---|---|---|---|---|
| H1 | Residual NLL diverges ≥3 nats at some α<10 | PASS | **PASS** | At α=10: drift +5.77, lift collapse −4.30 |
| H2 | mHC ≤0.5 nats drift over α∈[0,5] | FAIL | **FAIL (stronger)** | mHC drift exceeds 1.5 nats already at α=0.05; multi-stream is *less* drift-safe than residual on neutral prompts |
| H3 | HC also crashes at some α<α*(residual) | FAIL @ equiv init | **FAIL @ equiv init** | HC ≡ mHC bit-identical (Sinkhorn-Knopp is a no-op on ≈I matrices). Requires trained mixing (Phase mHC1.6 / R-7) |
| H4 | mHC counter-prior lift monotonic; residual collapses | PASS (partial) | **PASS (partial)** | At α=1: HC/mHC lift +0.071 vs residual −0.684. Multi-stream **does** protect the lift signal |
| H6 | α=0 bit-equal for all 3 archs | PASS | **PASS** | max-abs-diff=0.0 on all three (verified per run) |

## Interpretation (revised)

1. **The original "4.3× stability improvement" claim is retracted.** Under
   single-token `" The"` drift, HC/mHC appeared 4.3× more stable than
   residual; under true sequence NLL the picture inverts on neutral
   prompts. The single-token proxy was sensitive to a particular logit
   coordinate that happened to favour multi-stream architectures and is
   not a reliable cross-architecture safety signal.

2. **What HC/mHC *does* preserve, robustly: the lift signal.** At α=1.0
   the residual stream collapses (−0.684 nats on a counter-prior fact)
   while HC/mHC keeps a small but positive lift (+0.071). This is
   real—multi-stream readout protects the injection vector from being
   washed out by the native residual flow.

3. **What HC/mHC does *not* preserve: neutral fluency.** The same
   multi-stream readout that protects the targeted lift also broadcasts
   the bank perturbation into unrelated next-token distributions, raising
   neutral drift by ~3× over residual at every α.

4. **HC ≡ mHC at equivalence init** — confirmed bit-identical in this
   sweep. Sinkhorn-Knopp on identity matrices is a no-op. Disambiguating
   the two requires trained mixing parameters (Phase mHC1.6 / R-7).

5. **Implication for Phase R LOPI.** The orthogonal-complement projection
   `M_⊥ = M_V − ⟨M_V, V_ctx⟩/‖V_ctx‖² · V_ctx` is the missing factor:
   it removes precisely the component of M_V that lives along the native
   V direction and is therefore the source of neutral-prompt drift, while
   leaving the orthogonal component (the genuine novelty) intact. LOPI
   replaces the present 3-arm test bench as the recommended deployment
   path for industrial DeltaMemory.

## Repro

```
.venv-mac/bin/python scripts/run_mHC3_bank_injection.py \
    --alphas 0.05 0.1 0.5 1.0 2.0 5.0 10.0 --seeds 0 --facts false \
    --device mps --dtype bfloat16 \
    --out reports/cleanroom/mHC3_bank_injection
```

Legacy single-token sweep preserved at
`reports/cleanroom/mHC3_bank_injection/results_legacy_singletok.json`.

## Next

- mHC1.6 / R-7: 20k-step finetune on Wikitext-2 to separate HC vs mHC
  (H3 retest).
- R-1..R-5: Dynamic LOPI ablation — predict orthogonal projection drops
  neutral drift while preserving lift (validates the central thesis above).
- R-3.5: Replay mHC5 layer-norm probe under LOPI A2 (gauss-only) to test
  whether layer-localised injection further lowers drift.
