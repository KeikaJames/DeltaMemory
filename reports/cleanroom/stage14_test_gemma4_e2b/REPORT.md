# Phase G — Held-out Test Eval (Gemma-4-E2B)

**Date**: Phase G (FROZEN-config, one-shot)
**Model**: google/gemma-4-E2B, MPS, bf16
**Test split**: `eval/splits/test.jsonl` — N=39 facts, 6 paraphrase queries each (234 queries)
**Frozen config**: `deltamemory/configs/v3_frozen.yaml` (sha256 of K-projector embedded)

> **Preregistration discipline.** This is the **only** test-set run with the
> frozen config. Per `docs/preregistration.md`, no further changes to the v3
> code path are permitted post-this-run; any future modification must go
> through an Amendment block in the preregistration.

## Results — recall@1

| Condition | mean | per-seed |
|---|---:|---|
| **B1 prompt_insertion** | **0.6581** | 0.658, 0.658, 0.658 |
| B2 rag_oracle (in-context lookup) | 0.6496 | 0.650, 0.650, 0.650 |
| B0 no_memory | 0.3590 | 0.359, 0.359, 0.359 |
| v3_period_kproj (FROZEN) | 0.2764 | 0.278, 0.278, 0.274 |
| v2_period_no_kproj | 0.0000 | 0.000, 0.000, 0.000 |

(Eval pipeline is deterministic — the 0.001 jitter at v3 seed=2 is from
Mac MPS bf16 nondeterminism in argmax tie-breaks.)

## Statistical comparisons (paired by fact, Wilcoxon signed-rank; bootstrap 95% CI from 2000 resamples)

| Comparison | Δrecall | wins/losses/ties | p (1-sided) | 95% CI |
|---|---:|:-:|---:|:-:|
| **v3 − B0** | **−0.0812** | 5/15/19 | **p(less) = 0.0074** | [−0.150, −0.017] |
| v3 − v2 | +0.2778 | 22/0/17 | p(greater) < 0.001 | [+0.188, +0.372] |
| v3 − B1 | −0.3803 | 0/36/3 | p(less) < 0.001 | [−0.444, −0.316] |
| B1 − B0 | +0.2991 | 30/2/7 | p(greater) < 0.001 | [+0.222, +0.376] |
| B2 − B0 | +0.2906 | 31/2/6 | p(greater) < 0.001 | [+0.214, +0.368] |

After Holm–Bonferroni at α=0.05 across the five tests above, every result
is significant in the direction shown.

## Hypotheses verdict (binding from `docs/preregistration.md`)

| Hypothesis | Verdict on test |
|---|---|
| H1: v3 > B0 (no-memory) | **REJECTED.** Direction is significantly opposite (p=0.0074). |
| H1b: v3 > v2 | Confirmed. The trained InfoNCE projector lifts the bank from 0.000 (active destruction) to 0.278 (partial recovery). |
| H2: v3 ≥ B1 prompt-insertion | **REJECTED.** Prompt-insertion dominates v3 by 38pp. |
| H3: v3 ≥ B2 RAG-oracle | **REJECTED.** RAG-oracle dominates v3 by 37pp. |

## Honest interpretation

1. **Dev did not transfer to test.** On dev (`stage14_dev_kproj`), v3 beat
   B0 by +8.1pp with p=0.012. On test the sign flipped: v3 is 8.1pp
   **below** B0 with p=0.007. The wins/losses ratio went from 15/5 (dev)
   to 5/15 (test) — a near-perfect mirror. The dev "win" was not the
   underlying capability we hoped for; either (a) the InfoNCE projector
   overfit to the dev paraphrase distribution, (b) the 33-fact dev split
   has favorable relation/entity composition relative to test, or (c)
   both.
2. **The projector is real but insufficient.** v3 lifts v2 from 0.000 →
   0.278, a strong positive effect (p<0.001, CI [+0.19, +0.37]). The
   InfoNCE alignment is doing something useful — it just isn't enough to
   beat the base model's own LM head on a held-out distribution.
3. **Prompt-insertion is the right baseline to beat.** Both prompt
   insertion (B1) and RAG-oracle (B2) reach ~0.65 recall@1 — basically
   double the no-memory baseline. v3 in its current form is not
   competitive with this trivial, training-free baseline. Any honest
   "memory architecture" claim has to clear B1.
4. **What this paper-grade result means.** With a preregistered split, a
   frozen config, sha256-pinned weights, and a one-shot test eval, this
   is publishable as a **negative result** with a clear constructive
   takeaway: the K-Q alignment problem inside the bank's own attention
   softmax is harder than we modelled, and per-layer linear InfoNCE on
   ~520 pairs is underpowered.

## What this report does NOT claim

- It does not claim v3 is broken. v3 is significantly better than the
  untrained v2 bank.
- It does not claim DeltaMemory cannot work. It claims the **frozen v3
  config does not beat no-memory on a 39-fact held-out test split with
  Gemma-4-E2B on MPS bf16**.
- It does not claim prompt-insertion solves long-term memory. B1/B2 here
  are oracle baselines (the gold fact is in the prompt). They are upper
  bounds for any retrieval-style approach, not realistic deployments.

## Next, scientific honesty required

- **Do not modify v3 to retroactively beat the test set.** That would be
  preregistration violation.
- **What v3.1 would need** (future amendment, not this PR):
  1. larger training set (current 520 pairs from 104 train facts is too
     small for 35-layer per-layer projectors).
  2. negatives from outside the train relation set (current InfoNCE
     negatives are only same-batch, same-relation-pool).
  3. either top-k retrieval gating or an L2-normalisation step to fight
     softmax dilution at N≥30.
  4. a multi-step writer (current single-position write encodes a
     compressed fact; multi-token writers like ROME-Sequential may help).
- **What this PR delivers**: rigorous infrastructure (preregistration,
  holdout split, statistical pipeline) + a frozen v3 reference + an
  honest negative test result. That alone is the standard for顶会
  reproducibility.

Generated by `scripts/run_stage14_test_eval.py`.
