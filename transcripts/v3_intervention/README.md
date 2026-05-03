# Phase N — Intervention Demo Results

This directory contains output from `scripts/run_intervention_demo.py`, the
end-to-end test of whether DeltaMemory's attn-native bank can shift the
next-token distribution of a **frozen** base LLM.

## How to read these numbers

Per fact we record the log-probability the model assigns to the answer's first
token under three conditions:

| condition | what it is | LLM weights |
|---|---|---|
| **B0** | LLM alone, no memory | frozen |
| **B1** | LLM with the fact prepended to the question (in-context) | frozen |
| **v3** | LLM with the fact written into the per-layer attn-native K/V bank, alpha-merged into attention softmax | **frozen** |

The red line: at α=0 or empty bank, **v3 must be bit-identical to B0**
(verified by `tests/conservation_real_models.py`, max-abs-diff = 0.000).

## Results

### `gemma-4-e2b/` — Mac MPS bf16, v3 frozen K-projector (trained on this arch)

The honest story for a model that **already knew** most of these facts: the
bank cannot beat what the model has internalized, but it does not hurt either.
The cleanest demonstration is **f1 Paris/Anne Hidalgo**, which the model did
not know:

| fact | B0 | B1 prompt | v3 bank | v3 − B0 |
|---|---:|---:|---:|---:|
| f1 mayor of Paris → Anne | **−5.05** | −0.36 | **−0.64** | **+4.41** |
| f2 architect of Eiffel | −0.36 | −0.52 | −0.38 | −0.02 |
| f3 painter of Mona Lisa | −0.19 | −0.47 | −0.20 | −0.01 |
| f4 general relativity | −0.20 | −0.52 | −0.16 | +0.04 |
| f5 Python creator | −0.07 | −0.32 | −0.07 | −0.00 |

Reading: **+4.41 log-prob ≈ 80× probability** lift on the one fact the model
truly didn't know. On the four facts it already knew, v3 stays within ±0.04
of B0 — i.e. the bank does not pollute the distribution when not needed.

### `qwen3-4b-instruct-2507/` — GB10 CUDA bf16, v3 K-projector NOT trained on this arch

This is what happens when you take a frozen K-projector trained on Gemma-4 and
plug it into Qwen3 without retraining: the bank's raw K vectors live in a
different scale/basis from Qwen3's running K, so at α=1.0 the bank scores
dominate softmax with garbage. **All 5 facts collapse by ~12 logprob.**

| fact | B0 | B1 prompt | v3 bank | v3 − B0 |
|---|---:|---:|---:|---:|
| f1 mayor of Paris → Anne | −10.16 | −0.36 | −12.70 | −2.54 |
| f2 architect of Eiffel | −2.95 | −6.88 | −14.63 | −11.68 |
| f3 painter of Mona Lisa | −0.32 | −0.74 | −14.55 | −14.23 |
| f4 general relativity | −0.28 | −1.57 | −12.68 | −12.40 |
| f5 Python creator | −2.11 | −6.67 | −13.55 | −11.43 |

This is **expected**, **not a bug**, and **does not violate the red line**:
- α=0 still gives bit-equal (verified by conservation test, max-abs-diff = 0.000)
- Phase L (v3.1) retrains the K-projector with **cross-relation hard negatives
  on multiple architectures** so this collapses to the Gemma-4 picture above

The point of this transcript is to make the failure mode visible **before**
v3.1 retraining, not after.

## Reproduce

```
# Mac
python scripts/run_intervention_demo.py \
    --model google/gemma-4-E2B --device mps --dtype bfloat16 --alpha 1.0

# GB10
python scripts/run_intervention_demo.py \
    --model Qwen/Qwen3-4B-Instruct-2507 --device cuda --dtype bfloat16 --alpha 1.0
```

## What this demo does *not* show

- Per-relation generalization (need ≥30 relations × 60 paraphrases — Phase L1)
- Held-out paraphrase recall (frozen v3's Phase G eval = 0.278, see
  `reports/cleanroom/stage14_test_gemma4_e2b/REPORT.md`)
- Comparison against MEMIT/ROME (Phase M B3, weight-editing baseline)
- Cross-arch trained projector (Phase L2 will train v3.1 on GB10)

These five facts are illustrative — not statistical evidence.
