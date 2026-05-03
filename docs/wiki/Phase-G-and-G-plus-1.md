# Phase G + Phase G+1 — Honest Negative & Methodology Revision

This page tells the story of the most important moment in the project: the held-out test eval that **rejected our hypothesis**, and what we did about it.

If you only read one wiki page after [[Home]], read this one. It is what tells you whether this project is worth your trust.

---

## Phase G — what we ran

**Date:** Stage 14, commit `c7ba6b5`.
**Spec:** `deltamemory/configs/v3_frozen.yaml` (sealed before the run).
**Eval:** `eval/splits/test.jsonl` — N = 39 paraphrases over 6 LAMA-TREx relations (P36 P19 P101 P641 P937 P39).
**Conditions:** B0 (no memory), v2 (raw bank, no projector), v3 (frozen K-projector), B1 (prompt-insertion).
**Hypothesis H1:** v3 recall@1 > B0 recall@1, paired Wilcoxon p < 0.05.
**Decision rule:** single seed, no replay, no best-of-N.

## What happened

```
recall@1 (N=39):
  B0 no-memory          = 0.359   ←  the bar
  v3 frozen             = 0.278   ←  H1 rejected: WORSE than B0
                                     wilcoxon p = 0.007
                                     bootstrap 95% CI on (v3 − B0) = [-0.144, -0.018]
  v2 (no projector)     = 0.000
  B1 prompt-insertion   = 0.658
```

H1 is **actively rejected** (the difference is significant in the wrong direction). The pre-committed methodology forbids us from re-running with a different seed, picking a different split, or claiming "non-significant null". So we did not. We committed the negative result, wrote the report, and asked the harder question: **why?**

## Diagnosis

### D1. The K-projector did its job

`v3 − v2 = +0.278` confirms H1b: the trained InfoNCE projector lifts the raw bank from 0.000 to 0.278 — a real lift. The problem is **not** the projector.

### D2. The raw bank is incoherent at this N

`v2 = 0.000` is not noise — it is a structural failure. Without a projector, the bank's K vectors are post-RoPE post-norm Gemma-4 K slices that, when concatenated into softmax with the prompt's K, produce attention weights that bear no relationship to the question. The bank "speaks" but the model can't hear it.

### D3. Softmax dilution at N=39

The prompt has ~10–15 tokens. The bank adds 39 entries that compete on equal footing in `softmax(QKᵀ)`. Even with a perfect projector, the bank's mass is spread across 39 entries; the right entry's relative weight drops as N grows. This is geometric, not a training issue. The fix is structural: **gate** the bank with top-k before softmax.

### D4. Encoder fingerprints surface form, not facts

Stage 11A held-out paraphrase recall = 0.138 (multilayer) / 0.053 (prompt_hidden). That tells us the encoder learned to match strings, not concepts. The InfoNCE training supplied 6 relations × ~3 paraphrases each ≈ surface-form variety; not enough for the encoder to abstract over relation expression.

## Phase G+1 — what changed

We wrote the methodology amendment **before** any v3.1 runs. It lives in [[Preregistration#methodology-amendment-stage-15]] and `reports/cleanroom/stage14_test_gemma4_e2b/REPORT.md` (appended). Five changes:

| ID | change | addresses | shipped? |
|---|---|---|---|
| A1 | `bank_topk` gate before softmax | D3 dilution | ✅ commit `0036f62` |
| A2 | 5× training scale + cross-relation hard negatives | D4 encoder lazy | Phase L1 in progress |
| A3 | Cross-architecture training (Gemma-4 + Qwen3 + …) | generalization | Phase L2 pending GB10 |
| A4 | Two-stage val gate (val2 split before test) | reduces optimism | Phase L4-L5 |
| A5 | Fresh test split for v3.1 (`test_v31.jsonl`) | no test contamination | Phase L5 |

## Counterfactual: what if we'd chased the bug?

We could have:
- Re-run with a different seed until p > 0.05. (would have violated preregistration)
- Tuned α post-hoc on test. (would have made test = train)
- Dropped 4 unfavorable items and reported on N=35. (selection bias)
- Switched to a benchmark where v3 wins, with no protocol change. (cherry-picking)

We did none of these. The preregistration is a fence, and the fence held when it mattered.

## What this means for the user

If you're considering using DeltaMemory in production:

1. **Conservation law works** — at α=0 / empty bank, your model's outputs are identical to the unpatched LLM. There is no risk of degrading a frozen production deployment by *attaching* the patcher; only by *injecting* (α > 0) when the bank is wrong.
2. **The qualitative wins are real** — see Phase N intervention demo: +4.41 logprob lift on a fact the model didn't know, no pollution on facts it did. This is the property a production system needs.
3. **The aggregate metric is not yet positive** — recall@1 against B0 on a paraphrase-only held-out split is below B0 at the v3 frozen point. Phase L (v3.1) is in progress and will be re-evaluated on a fresh test split.
4. **Prompt-insertion still wins** — at 0.658 vs DeltaMemory's 0.278, B1 prompt-insertion is the bar to beat. We do not claim DeltaMemory beats prompt-insertion; we claim it modifies the LLM's *internal attention distribution* in a way prompt-insertion cannot, while keeping the LLM frozen.

The honest answer to *"does DeltaMemory work?"* in 2026-Q1 is: **the channel works (proven by conservation + Phase N), the aggregate score on this benchmark does not yet beat B0 (Phase G), v3.1 is in progress to fix the structural reasons.**

## Why not just delete v3 and ship v3.1?

Because Phase G is **the only honest record** of what the project's frozen-baseline state was on 2026-04-XX. Deleting it would erase the audit trail of "we shipped, we measured, we updated". Future readers — including future us — should be able to see exactly what was claimed and what was found.

We frequently re-read this page when deciding whether a tempting v3.1 design choice is "real progress" or "Phase G again, in a hat".
