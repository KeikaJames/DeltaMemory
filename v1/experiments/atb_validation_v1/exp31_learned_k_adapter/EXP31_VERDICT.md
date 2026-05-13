# Exp31 Verdict — Learned K-adapter (H_A)

Status: **NEGATIVE** (H_A falsified — the K-space discriminability hypothesis is rejected).

## Setting

- Model: `Qwen3-4B`, MPS / bf16, base fully frozen.
- Data: CounterFact-1k split 567 train / 115 val / 125 test by `fact_id`,
  no leakage. Paraphrase Qs from CounterFact's own paraphrase field.
- Adapter: `KProjectorBank`, per-layer `Linear(d_h → rank → d_h)` low-rank
  residual to identity. Rank 64 main path; full-rank ablation also run.
- Loss: per-layer InfoNCE on pooled-heads L2-normalised cos-sim, τ = 0.07,
  summed over all 36 layers.
- Seeds: 0 / 1 / 2 (rank=64) + seed-0 full-rank + seed-0 shuffled-pair
  Gate-E control.
- Eval: model-attached, n = 125 test facts × 3 seeds × bank N = 200 ×
  α ∈ {0.003, 0.005, 0.010} × 12 variants → 12,750 cells.

## Φ2c result (embedding-space)

The adapter trains cleanly and learns strong cos-sim routing in pure
embedding space:

| variant                    | val_top1 | × chance (1/115 ≈ 0.87 %) |
|---------------------------|---------:|---------------------------:|
| rank = 64, seed 0          | 34.78 %  | 40 ×                       |
| rank = 64, seed 1          | 33.48 %  | 38.5 ×                     |
| rank = 64, seed 2          | 34.78 %  | 40 ×                       |
| full-rank, seed 0          | 38.70 %  | 44.5 ×                     |
| shuffle-pairs control      |  0.87 %  | 1 ×                        |

Shuffled-pair training collapses to chance — the learned cos-sim signal
is real, fact-bound, and not a representational artefact of the
projector.

## Φ2d result (model-attached, paired-bootstrap CI, B = 2000)

Five gates of `full_bank_learned_adapter` minus the corresponding
control (margin = `logP(target_new) − logP(target_true)`):

### α = 0.003

| gate | control                         | mean Δ  | 95 % CI            | verdict |
|------|---------------------------------|--------:|---------------------|---------|
| A    | minus_correct_learned           | −0.335  | [−0.508, −0.158]    | **FAIL** |
| C    | meanV_learned                   | −0.101  | [−0.241, +0.040]    | NULL    |
| D    | shuffled_factids_learned        | −0.148  | [−0.289, −0.000]    | **FAIL** |
| E    | full_bank_shuffled_adapter      | −0.659  | [−0.937, −0.373]    | **FAIL** |

### α = 0.005

| gate | control                         | mean Δ  | 95 % CI            | verdict |
|------|---------------------------------|--------:|---------------------|---------|
| A    | minus_correct_learned           | +0.331  | [+0.145, +0.515]    | PASS    |
| C    | meanV_learned                   | +0.302  | [+0.139, +0.469]    | PASS    |
| D    | shuffled_factids_learned        | +0.255  | [+0.085, +0.434]    | PASS    |
| E    | full_bank_shuffled_adapter      | −0.145  | [−0.414, +0.142]    | NULL    |

### α = 0.010

| gate | control                         | mean Δ  | 95 % CI            | verdict |
|------|---------------------------------|--------:|---------------------|---------|
| A    | minus_correct_learned           | +0.141  | [−0.028, +0.314]    | NULL    |
| C    | meanV_learned                   | +0.104  | [−0.050, +0.259]    | NULL    |
| D    | shuffled_factids_learned        | +0.055  | [−0.111, +0.207]    | NULL    |
| E    | full_bank_shuffled_adapter      | −0.499  | [−0.748, −0.242]    | **FAIL** |

### Gate B — retrieval accuracy

`target_new` rank = 0 / 375 across **every** bank cell (learned, identity,
or shuffled adapter), at every α. Top-1 LM-output retrieval = 0 %, well
below the 0.5 % chance baseline. The 40× embedding-space retrieval did
**not** translate into LM-output routing.

## Interpretation

1. The learned adapter has real K-space discriminability — 40× chance in
   pure cos-sim retrieval — but this signal is **invisible at the LM
   logit head** once the adapter is attached at read time.
2. The shuffled-pair adapter (Gate E control) consistently matches or
   beats the trained adapter on margin at every α. Whatever "lift" the
   trained adapter provides over the identity projector is therefore
   **not fact-routing** — it is a generic K-norm rescaling that helps
   the bank deliver uniform steering off the base bias.
3. `full_bank_topk1_learned` is the worst variant at every α (−0.92 to
   −1.10), showing that hard routing to the projector's top-1 actively
   hurts. The "good" softmax-mixed cells are precisely the ones whose
   bank attention is most diffuse — exactly the steering signature
   Exp23–27 already identified.
4. The only α at which the conventional four gates (A / C / D) pass is
   α = 0.005, and even there Gate E is NULL with the mean on the
   wrong side of zero. This is the canonical "uniform steering"
   pattern, not routed memory.

## Decision

**H_A is rejected.** Adding a trained K projector does not turn α-additive
attention-side ANB into routed fact-recall, despite the projector having
demonstrably strong K-space discriminability in isolation.

The bottleneck is **architectural** (α-additive residual readout cannot
compete with sequence attention at fact-bank scale), not
**representational** (K-space is already separable enough). This
strengthens the Exp23–27 cross-arch negative result rather than
rescuing it.

## Next phase

Proceed directly to **Exp32 — MLP-side gated memory** (H_B): move the
injection site from attention to MLP-output and add a learned gate.
This tests the orthogonal hypothesis that the failure is one of *site*
rather than *route*.

## Files

- `experiments/atb_validation_v1/exp31_learned_k_adapter/`
  - `capture_kq.py`, `train_k_adapter.py`, `eval_k_adapter.py`
  - `run_mps_exp31_qwen_smoke/seed{0,1,2}/k_projector_seed*.pt`
  - `run_mps_exp31_qwen_smoke/seed0_fullrank/k_projector_seed0.pt`
  - `run_mps_exp31_qwen_smoke/seed0_shuffled/k_projector_seed0.pt`
  - `run_mps_exp31_qwen_smoke/eval_main/cells.jsonl` (12,750 rows)
  - `run_mps_exp31_qwen_smoke/eval_main/env.json`
