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

---

## Methodology adjustment after the negative result

The Phase G result is *informative*, not just *disappointing*. It rules
out three concrete hypotheses about what was wrong with v2 and forces
the v3.x program to drop one assumption, sharpen another, and add two
new constraints. We document the adjustment here so that future
amendments can be evaluated against a written-down baseline, not against
selective memory.

### What the data rules out

1. **InfoNCE alone does not solve K/Q misalignment.** v2 → v3 lifted
   bank recall from 0.000 to 0.278, so the projector *does* shift bank
   keys closer to paraphrase queries. But the lift is +27.8 pp on top
   of a 0.000 floor, while no-memory already sits at 0.359. A
   per-layer linear `Linear(d, d)` trained on 520 (write_K, paraphrase_Q)
   pairs is **not** a sufficient correction by itself.
2. **Period-token capture is not the bottleneck on its own.** Phase G
   tested period-policy v3 only (the frozen choice). Even with the
   trained projector aligning that capture, the bank loses to
   no-memory. Address-policy capture and multi-position write
   (Stage 14B/14C) were dev-evaluated but not selected as frozen; they
   remain candidates but are now constrained by the same dev-test
   transfer evidence below.
3. **Dev wins do not transfer at this projector capacity.** Dev showed
   v3 +8.1 pp over B0 (15W/5L/13T, p=0.012). Test shows v3 −8.1 pp
   under B0 (5W/15L/19T, p=0.007). The sign flipped on a held-out
   stratified split drawn from the same generator. This is the textbook
   signature of *overfit selection*, not of an architecturally broken
   bank.

### What changes for v3.1 (preregistration amendments required)

1. **Train-set scale floor.** Any v3.x candidate that touches the K
   projector must be trained on ≥ 5× the current pair count
   (≥ 2,500 (write, paraphrase) pairs across ≥ 30 relations) before
   it is allowed onto a dev sweep. Per-layer `d × d` linears at
   `d=256–512` with 35 layers are over-parameterised at 520 pairs;
   the dev/test gap is what we should expect at this regime.
2. **Cross-relation negatives are mandatory.** Current InfoNCE
   negatives are in-batch (same training pool). v3.1 must include
   hard negatives from *other* relations and from *random* token
   contexts so the projector cannot exploit relation-specific
   correlations that vanish at test time.
3. **Softmax dilution must be addressed structurally, not by tau
   alone.** The attn-native bank concatenates `N` bank slots into
   the same softmax as the sequence keys. At `N ≥ 30` the bank's
   share of the softmax mass collapses regardless of how good the
   per-key score is. v3.1 must implement at least one of:
   (a) top-k bank gating before the softmax,
   (b) L2-normalised dot products (cosine-only),
   (c) a separate bank-only attention head with its own softmax,
   merged additively rather than via shared partition.
4. **Two-stage validation gate, not one.** Going forward, no v3.x
   candidate is "frozen" until it has cleared both:
   (i) dev recall@1 ≥ B0 + 0.05 with paired Wilcoxon p < 0.01, AND
   (ii) a *second* held-out validation split (drawn at preregistration
   time, distinct from test) with the same gate. Test stays one-shot.
   This kills the dev/test sign-flip risk demonstrated here.

### What this means for the comparison frame

Until (1)–(4) are satisfied, **the only honest claim DeltaMemory can
make is "matches or beats prompt-insertion at equal compute, on a
held-out preregistered split".** "Matches no-memory" is too weak (B0 is
free). "Beats RAG" is too strong without a fair retriever. Phase G
fixes prompt-insertion (B1) at 0.658 as the bar. Any v3.1 result that
does not clear that bar should be reported as a negative, not buried
behind dev numbers.

### Process changes already adopted

- The dev/test sign-flip is documented in the preregistration
  amendment log.
- The frozen v3 config (`v3_frozen.yaml`) and its sha256-pinned
  projector remain unchanged. Any v3.1 work writes a new config with
  a new sha and a new amendment block; the v3 row in this report
  must remain reproducible by anyone with the repo at the PR-merge
  commit.
- We will publish the `summary.json` and `stats.json` from this run
  alongside the README so that a reader can audit the pairwise
  Wilcoxon and bootstrap CIs without re-running the eval.

