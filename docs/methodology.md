# DeltaMemory — Methodology Notes

This document records the math and the engineering choices behind DeltaMemory
(DM) and explicitly lists what is **not** claimed. It is the answer we owe
reviewers who ask "is this `eps`-injection actually principled, or is it
hand-wavy steering?"

## 1. What DM actually computes

Let the frozen base model factor as `F = W_lm ∘ T`, where `T` is the full
transformer body (`model.model`) and `W_lm` is the unembedding (`lm_head`).
For a read prompt `x` with last real-token position `p`, DM produces:

$$
\tilde{y}(x \mid \mathrm{slot}=k) = \mathrm{softmax}\bigl(W_{lm}(\,T(x)_p + \alpha \cdot b_k\,)\bigr)
$$

where `b_k = Writer(E[v_k])` is the slot-k bank vector. The Writer is a tiny
MLP applied to the (frozen) value-token embedding. The bank is keyed by a
KeyProjector applied to `Encoder(prompt or address)`; retrieval is cosine
top-1.

**Crucial structural fact:** injection happens **once**, at the final hidden
state, before `lm_head`. It is *not* a multi-layer steering vector inserted
mid-residual, it is *not* an attention bias, and it is *not* a fast-weight
update. It is mathematically equivalent to a learned, key-routed logit bias:

$$
\tilde{y} = \mathrm{softmax}\bigl(W_{lm} T(x)_p + \alpha\, W_{lm} b_k\bigr).
$$

This makes DM closer to a key-value memory with a learned readout than to
mid-layer activation steering or to ROME-style edits. We keep this name
("DeltaMemory") for historical reasons — earlier prototypes did inject at
every attention V-projection — but the production code path is single-point
final-residual injection.

## 2. Why we claim this is principled, not hand-wavy

The reviewer concern (paraphrased): "$\epsilon$-steering of the residual
stream is fragile because of the linear-representation hypothesis,
superposition, and a non-linear phase-transition between 'subtle' and
'destructive' intervention. How do you avoid those?"

Our short answer: **we don't try to solve them at the residual-stream level.
We solve them at the readout level.**

1. **Linear representation hypothesis.** We assume only the final
   readout layer is linear (it literally is, by construction:
   `lm_head = nn.Linear`). That's a far weaker assumption than assuming a
   linear concept geometry mid-residual. The Writer is trained end-to-end
   so that `W_lm b_k` is a *learned* direction in logit space — we
   never need to find the "right concept axis" by hand.

2. **Superposition.** Because the Writer's output is fed directly to
   `lm_head`, the only way it can interfere with another concept is if its
   contribution to the **logit vector** of an unrelated query is large.
   We measure this empirically as the *locality drift* on neutral controls
   (Stage 10C, 11D, 12-P3). When DM is broadcast (every query sees the
   whole bank), drift is large (Stage 12 P3 = 75 %). When DM is per-query
   routed via the encoder/retriever (Stage 11D), drift is 0 % across our
   tests. Per-query routing is therefore the production policy.

3. **Phase transition.** The "narrow window" between sub-threshold and
   catastrophic injection is exactly what we sweep over with `α` and what
   the writer's zero-init learns to land inside. Critically, the writer's
   final layer is initialised to all-zeros (`writer.down`), so before
   training the injection is exactly zero, and gradient descent walks `α b`
   into the productive regime under joint CE + InfoNCE objectives. We
   never hand-pick `α`; we set it to 1.0 once and let the writer scale.

4. **Encoder fingerprinting.** This is *not* solved. Stage 11A shows that
   even with paraphrase-augmented InfoNCE, our encoder still relies on
   surface lexical features (held-out paraphrase recall = 0.138). We
   report this as a real failure rather than papering over it. Solutions
   we have **not** implemented but think are promising:
   (a) Givens / Householder orthogonal banks,
   (b) sparse-autoencoder dictionary banks,
   (c) closed-form ROME-style edits that bypass the encoder entirely.

## 3. What we deliberately do *not* claim

- **DM is editable memory at the relation level.** Stage 10F and 11B show
  this is false: facts in a relation never seen during training cannot be
  added to the bank with retrieval ≥ 0.5.
- **DM solves the linear-representation problem.** It doesn't try to —
  the linear assumption only applies to `lm_head`.
- **DM survives paraphrase robustness with our current encoders.** It
  doesn't — see Stage 11A. Different encoder geometry needed.
- **DM matches RAG at multi-token open answers.** We test only single-
  token LAMA-TREx targets; multi-token decoding is open work.
- **DM is safe at α=1.0 with broadcast injection.** Broadcast destroys
  75 % of unrelated answers (Stage 12 P3). Production must use per-query
  retrieval.

## 4. Reproducibility

Every Stage 9–11 number in the README is gated on:
- 3 seeds,
- 95 % paired bootstrap CI (10 000 resamples),
- gate evaluated on the **CI lower bound** (not the mean),
- bit-exact harness (Stage 11E) confirms identical SHA-256 over the stable
  metric subset across two independent deterministic runs.

Hardware is logged per run (NVIDIA GB10 Blackwell, 128 GB unified, CUDA 13;
or Apple M-series MPS for the small-N pilots noted in
`docs/apple_silicon.md`).
