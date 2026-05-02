# Stage 10 — Adversarial Validation Report

> **Goal of Stage 10.** Stage 9 reported 1.000 ± 0.000 retrieval and binding accuracy on
> closed-book LAMA-TREx (N=183, 3 seeds). Reviewer-style scrutiny identified
> several reasons that number could be inflated: train/test share byte-identical
> prompts (eval is in-distribution memorisation), the bank carries free
> information about the *order* of facts, and baselines were under-trained.
> Stage 10 is a single-shot, all-runs-on-record adversarial campaign that puts
> 8 falsifiable hypotheses in front of the same DeltaMemory pipeline used in
> Stage 9 and reports gate verdicts honestly, including the failures.

## Hardware & Reproducibility

* All Stage 10 runs were executed on **NVIDIA GB10 (Blackwell, 119 GB unified
  memory)** with PyTorch 2.4 / CUDA / bf16. The Apple Silicon MPS path
  (Stage 8 v3 closure) was *not* used here — adversarial decoys at 1000× and
  N=16384 do not fit in the smaller MPS configurations we have validated.
* Three random seeds (0, 1, 2) for every multi-seed sub-stage.
* Frozen base: `google/gemma-4-E2B`. Identical writer / KeyProjector / FastWeightBank
  code paths as Stage 9; only new code is the post-evaluation stress-test block
  (`scripts/run_stage8.py` `_stage10_*` helpers).
* All artifacts under `reports/experiments/stage10*` and aggregated to
  `reports/experiments/stage10_adversarial_validation/{stage10_summary.json,
  SUMMARY_TABLE.md}`.

## Sub-stages and gates

| ID  | Hypothesis it falsifies | Test | Gate |
|---|---|---|---|
| 10A | F1: encoder is semantic, not fingerprint | re-tokenise read-prompt with 5 held-out paraphrases of each LAMA-TREx address; query the *trained* bank | held-out recall@1 ≥ 0.85 |
| 10B | F2: retrieval is sharp at scale  | append K×N random-init slots; re-run retrieval | bind_top1 ≥ 0.80 at K=1000 |
| 10D | F5: bank values do real work     | replace bank.v with random / shuffled tensors after training | bind_top1 ≤ 0.10 (i.e. *should* collapse) |
| 10F | F7: writer generalises across relations | leave one Wikidata relation out at training, then add its facts to the bank zero-shot | holdout bind_top1 ≥ 0.50 |
| 10C | F3/F4: DeltaMemory beats equal-budget baselines | SFT-LoRA × {r=4,16,64} × 3 seeds × 1500 steps; vector-RAG; IKE | DeltaMemory > all baselines on edit_top1 *and* on locality drift |

10E (N=16384 scale stress) was attempted but OOM'd at K=100 decoys
(decoy keys + matmul exceeded GB10 free memory). The scale claim is instead
covered by Stage 9A (N=4096) + Stage 10ABD (decoy×1000 = 1.000 at N=183).

## Results

### 10A/B/D — Paraphrase, decoy, and value ablation (3 seeds)

| Encoder       | std retr@1     | std bind top1 | paraphrase canonical | **paraphrase HELD-OUT** | bind held-out | decoy×1000 bind | rand-bank top1 | shuf-bank top1 |
|---|---|---|---|---|---|---|---|---|
| prompt_hidden | 1.000 ± 0.000  | 1.000 ± 0.000 | 0.820 ± 0.007        | **0.113 ± 0.020**       | 0.128 ± 0.019 | 1.000 ± 0.000   | 0.000 ± 0.000  | 0.015 ± 0.007  |
| multilayer    | 1.000 ± 0.000  | 1.000 ± 0.000 | 0.958 ± 0.010        | **0.307 ± 0.021**       | 0.309 ± 0.023 | 1.000 ± 0.000   | 0.000 ± 0.000  | 0.015 ± 0.007  |

### 10F — Leave-one-relation-out (zero-shot)

| Held-out relation | n   | retrieval@1 (holdout) | bind top1 (holdout) |
|---|---|---|---|
| P101 (field of work) | 20 | 1.000 | 0.000 |
| P19 (place of birth) | 10 | 1.000 | 0.200 |
| P36 (capital)        | 63 | 1.000 | 0.095 |
| P39 (position held)  | 20 | 1.000 | 0.000 |
| P641 (sport)         | 19 | 1.000 | 0.000 |
| P937 (work location) | 16 | 1.000 | 0.375 |
| **mean**             |  — | **1.000 ± 0.000** | **0.112 ± 0.152** |

### 10C — Equal-budget baselines (1500 steps for SFT-LoRA)

| Method                 | edit top1       | edit top5       | locality drift (lower is better) |
|---|---|---|---|
| vector_rag             | 0.399 ± 0.000   | 0.486 ± 0.000   | n/a                              |
| IKE                    | 0.399 ± 0.000   | 0.486 ± 0.000   | 0.500 ± 0.000                    |
| SFT-LoRA r=4           | 0.541 ± 0.005   | 0.617 ± 0.000   | 0.556 ± 0.096                    |
| SFT-LoRA r=16          | 0.552 ± 0.000   | 0.617 ± 0.000   | 0.556 ± 0.096                    |
| SFT-LoRA r=64          | 0.552 ± 0.000   | 0.617 ± 0.000   | 0.778 ± 0.096                    |
| **DeltaMemory (prompt_hidden)** | **1.000 ± 0.000** | — | **0.000** (read-time inject) |

## Gate verdicts (honest)

| Gate | Result |
|---|---|
| **G10A** held-out paraphrase recall ≥ 0.85 | **FAIL** — prompt_hidden 0.11, multilayer 0.31 |
| **G10B** decoy×1000 bind top1 ≥ 0.80       | **PASS** — both encoders 1.000 |
| **G10D** value ablation top1 ≤ 0.10        | **PASS** — random 0.000 / shuffled 0.015 vs unablated 1.000 |
| **G10F** LORO holdout bind top1 ≥ 0.50     | **FAIL** — mean 0.112 (retrieval generalises, writer does not) |
| **G10C** DeltaMemory beats baselines on canonical | **PASS** — 1.000 vs SFT-LoRA r=64 0.552 (with 0.78 drift) |

## What this changes about the Stage 9 claim

Stage 9 narrative: *"DeltaMemory turns Gemma-4-E2B into an N=4k retrievable
factual store with 1.000 ± 0 binding."* That is true **only on byte-identical
canonical prompts**. Stage 10 narrows the claim:

1. **The bank itself is real.** Random or shuffled bank values destroy the
   prediction; the Stage 9 numbers are not bookkeeping artifacts. (G10D PASS.)
2. **Retrieval is sharp at scale**, even when 1000× random distractor slots
   are added. The KeyProjector + FastWeightBank machinery does its job. (G10B
   PASS.)
3. **The encoder is not yet semantic.** When the same fact is queried with a
   surface-form-paraphrased prompt — even one wrapped in the same Atlas-slot
   template — held-out recall collapses to 0.11 (prompt_hidden) / 0.31
   (multilayer). The encoder learnt a near-byte-level fingerprint of the
   trained read-prompt rather than an embedding of the *address*. (G10A FAIL.)
4. **The writer does not generalise across Wikidata relations zero-shot.**
   When 6 relations train the writer and the 7th is appended to the bank
   without any further optimisation, retrieval is perfect (1.000) but binding
   drops to 0.00–0.38 (mean 0.11). The encoder maps held-out addresses into
   distinct keys, but the writer-produced values for those held-out keys do
   not decode back to the answer token. (G10F FAIL.)
5. **DeltaMemory still beats RAG / IKE / SFT-LoRA at equal training budget on
   the canonical regime**: 1.000 vs the best baseline 0.552 with 56–78%
   collateral drift. So the *operational claim* — "more efficient than full
   fine-tuning, no model-weight surgery" — survives. (G10C PASS.)

The honest summary is therefore:

> **DeltaMemory is a real, validated factual store on canonical prompts** (G10B,
> G10D, G10C all PASS), **but its key-/value-network does not yet generalise
> across surface paraphrase or unseen relations** (G10A, G10F FAIL). The next
> step is *not* further hyper-parameter sweeps in the canonical regime; the
> remaining error is in representation, not optimisation.

## Suggested follow-ups (Stage 11)

* **Encoder fix.** Train the address encoder with explicit paraphrase
  augmentation and an InfoNCE-style loss over template variants of the same
  address. Target gate G10A ≥ 0.85.
* **Writer fix.** Train with relation-stratified mini-batches and a held-out
  relation in every step (LORO during training, not just at eval). Target G10F
  ≥ 0.50.
* **N=16k scale.** Add a chunked-decoy implementation (already partially in
  place after the OOM patch) and re-run 10E at K∈{10,100} only, on a host
  with ≥ 192 GB.
* **Statistical rigour.** Replace mean ± std with paired bootstrap CIs and
  Cohen's d when comparing encoders / baselines.

— Generated 2026-05 from `reports/experiments/stage10_*` artifacts.
