# Exp34 — ROME-style Parameter Edit (Phase C Positive Control)

**Status: PASS — positive control confirmed.**
**Date: 2026-05-13.**
**Model: Qwen3-4B-Instruct-2507 (MPS, bf16).**
**Splits: Exp31/32 CounterFact test set (N = 125 held-out facts).**

This experiment is the positive control for the architectural claim of
Exp31/Exp32/Exp33: bank-style external memory (whether residual-additive
or joint-softmax) cannot bind fact identity at the LM head on Qwen3-4B.
The natural counter-question is *"is the test/eval framework itself capable
of producing positive results on this base model?"* — i.e. is the negative
verdict an artefact of the base model being un-editable?

We run a minimal ROME [Meng et al. 2022] rank-1 edit on
`mlp.down_proj.weight` at layer L=5 (early-middle MLP, the canonical
ROME knowledge-stratum target), per fact, with the same prompts and
metric as Exp31–33. If parameter editing produces clean Gate B / D
PASS on the same splits, the test framework is validated and the
Exp31/32/33 NEGATIVE verdicts lock in.

---

## 1. Algorithm

Single-fact ROME, no covariance preconditioning (minimum viable):

1. **k\*** — capture the activation entering `down_proj` at the
   subject's last token in the write prompt (intermediate dim, 9728 for
   Qwen3-4B).
2. **v\*** — optimise a residual addition `δh ∈ R^d_model` at layer-L
   output of the canonical read prompt's last token, via 25 steps of
   Adam (lr = 0.5), maximising
   `logp(target_new) − 0.5 · logp(target_true)` with weight-decay
   `1e-3 · ‖δh‖²`.
3. **Rank-1 update**:
   ```
   W_new = W_old + (v* − W_old · k*) ⊗ k* / (‖k*‖² + 1e-2)
   ```
4. **Evaluate** `margin = logp(target_new) − logp(target_true)` at the
   last token of the canonical read prompt and up to 2 paraphrase
   prompts, average. Restore `W_old`. Proceed to next fact.

Variants per fact:

| Variant       | Description                                              |
|---------------|----------------------------------------------------------|
| `base`        | No edit (baseline margin).                               |
| `edited`      | Rank-1 update with correct k\* — Gate B.                 |
| `shuffled_k`  | Rank-1 update with a permuted fact's k\*  — Gate D.      |

---

## 2. Configuration

| Parameter      | Value |
|----------------|-------|
| Model          | Qwen/Qwen3-4B-Instruct-2507 |
| Device / dtype | MPS / bf16 |
| Edit layer L   | 5 |
| v\* opt steps  | 25 (Adam, lr = 0.5) |
| Update reg λ   | 1e-2 |
| N test facts   | 125 (same split as Exp31/32/33 test.json) |
| Seed           | 0 |

---

## 3. Result (N = 125 full)

```
base_mean              = −3.927    (model prefers target_true by 3.9 nats)
edited_mean            = +5.626    (model now prefers target_new by 5.6 nats)
shuffled_k_mean        = −3.656    (wrong k* — edit fails to transfer)

mean_edited_minus_base       = +9.552   nats   (Gate B uplift)
mean_edited_minus_shuffled   = +9.281   nats   (Gate D identity binding)

frac edited > base           = 125 / 125   (100.0 %)
frac edited > shuffled_k     = 123 / 125   ( 98.4 %)
frac edited > 0 (flipped)    = 108 / 125   ( 86.4 %)
```

All paraphrase-averaged margins. Per-fact rows in
`run_qwen_exp34/cells.jsonl`.

### Gate verdicts

| Gate | Criterion                                       | Result | Verdict |
|------|--------------------------------------------------|--------|---------|
| B    | `edited − base > 0` per-fact ≥ 95 %             | 100.0 %| **PASS_STRONG** |
| D    | `edited − shuffled_k > 0` per-fact ≥ 80 %       | 98.4 % | **PASS_STRONG** |
| —    | `edited_mean > 0` (model actively flipped)      | 86.4 % | **PASS** |

---

## 4. Interpretation

The base model **is** editable, with a per-fact +9.55-nat uplift and
clean identity binding (98.4 % of edits fail to transfer when the
wrong subject's k\* is used). The test framework, the metric
(margin = logp(target_new) − logp(target_true) at the last token), the
prompts, and the splits all reliably distinguish "edit landed" from
"edit didn't land".

Therefore, the **NEGATIVE** verdicts in
[`EXP31_VERDICT.md`](../exp31_learned_k_adapter/EXP31_VERDICT.md),
[`EXP32_VERDICT.md`](../exp32_mlp_side_gated_memory/EXP32_VERDICT.md),
and [`EXP33_VERDICT.md`](../exp33_reattn_readout/EXP33_VERDICT.md)
are **not** artefacts of the test framework. They are genuine
architectural failures of the α-additive residual-readout and
joint-softmax bank readout protocols on this base model.

---

## 5. Cross-experiment summary

| Exp | Readout protocol                          | Gate B  | Gate D  | Verdict |
|-----|-------------------------------------------|---------|---------|---------|
| 24  | Cosine-routed native ATB (attn-side)      | 0 / 375 | fail    | NEGATIVE |
| 27  | Sparse joint-softmax attn bank, N=200     | 0 / 375 | fail    | NEGATIVE |
| 31  | Learned K-adapter + ATB                   | 0 / 375 | fail    | NEGATIVE |
| 32  | MLP-side α-additive gated bank            | 0 / 375 | fail (−1.17) | NEGATIVE |
| 33  | Joint-softmax bank on Exp31/32 splits     | 0 / 375 | fail (−0.108) | NEGATIVE |
| **34** | **Rank-1 down_proj edit (ROME)**        | **125 / 125** | **123 / 125** | **POSITIVE** |

Five independent architectures, all bank-shaped (route → read →
add-to-stream), all fail Gate B and Gate D on Qwen3-4B at fact-bank
scale. A single rank-1 weight edit succeeds on every fact.

The architectural claim is: **fact-identity binding on Qwen3-4B
requires modifying the parameter manifold of `mlp.down_proj`, not the
activation manifold**. Bank architectures, regardless of routing
quality or readout site, do not write into that manifold.

---

## 6. Files

- `run_exp34.py` — runner.
- `run_qwen_exp34/summary.json` — aggregate metrics.
- `run_qwen_exp34/cells.jsonl` — per-fact rows (125).

## 7. Reproduction

```
cd v1/experiments/atb_validation_v1/exp34_rome_baseline
python3 run_exp34.py --n-test 125 --edit-layer 5
```

Wall-clock ≈ 8 min on M-series MPS bf16.

## 8. Caveats

- Single-fact sequential edits, no covariance preconditioning. This is
  a *minimum viable* ROME, not full MEMIT. Sufficient for the positive-
  control claim; not a publishable editing method on its own.
- Per-fact restore: we do not test cumulative-edit stability or
  capability retention. The claim is *not* "ROME at scale on Qwen3-4B
  works"; the claim is "the test framework is capable of detecting
  fact-identity binding when it is actually present".
- Single seed (0). The effect size is so large
  (+9.55 nats, 100 % per-fact) that seed-robustness is moot for the
  positive-control claim.
