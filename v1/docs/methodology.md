# Mneme — Methodology Notes

This document records the math and the engineering choices behind Mneme
(DM) and explicitly lists what is **not** claimed. It is the answer we owe
reviewers who ask "is this `eps`-injection actually principled, or is it
hand-wavy steering?"

## 1. What DM actually computes

The current production path is attention-native external K/V memory. During a
write pass, Mneme captures per-layer native K/V vectors from a frozen
Transformer. During a read pass, each supported attention layer computes:

$$
\mathrm{Attn}_\ell(Q, K, V)
\rightarrow
\mathrm{Attn}_\ell\bigl(Q,\,[K; M_K^{(\ell)}],\,[V; \alpha M_V^{(\ell)}]\bigr)
$$

`M_K` and `M_V` are external bank tensors, not model weights. The read prompt
contains only the question; source text is not appended to the prompt. The base
model remains frozen, and the bank path is skipped entirely when the bank is
empty or `α = 0`.

For RoPE models, the implementation stores pre-RoPE K and compares bank slots
against pre-RoPE Q so bank lookup is position-agnostic. For GQA/MQA models,
bank K/V are stored at `num_key_value_heads` resolution and expanded through
the model's native repeat-KV path.

Historical residual-writer / KeyProjector prototypes are kept in legacy
scripts and reports, but they are not the current mainline mechanism described
by `deltamemory.memory.attn_native_bank`.

## 2. Why we claim this is principled, not hand-wavy

The reviewer concern (paraphrased): "$\epsilon$-steering of the residual
stream is fragile because of the linear-representation hypothesis,
superposition, and a non-linear phase-transition between 'subtle' and
'destructive' intervention. How do you avoid those?"

Our short answer: **we constrain the external-KV channel and measure the
resulting drift directly.**

1. **No learned mid-residual concept axis.** The mainline bank captures native
   attention K/V activations from the same frozen model that will read them.
   There is no trained Writer in the attention-native path and no assumption
   that a hand-picked residual direction encodes the fact.

2. **Superposition.** The bank competes inside attention, so unrelated queries
   can still read a bank slot if its K is spuriously similar. We measure this
   as NLL/locality drift and keep the α=0 / empty-bank bit-equality red line as
   a hard invariant.

3. **Phase transition.** The intervention strength is swept through `α`.
   Architecture adapters provide conservative defaults, and V-scale
   calibration caps captured bank values for families without native V
   normalization.

4. **Optional guards.** mHC, LOPI, U-LOPI, and ECOR are explicit ablation
   toggles. They are not required for the default bank path and should not be
   presented as evidence unless the corresponding experiment enables them.

## 3. What we deliberately do *not* claim

- **DM is guaranteed editable memory at the relation level.** The bank can
  still fail when the frozen model's native attention geometry does not route a
  query toward the captured slot.
- **DM solves the linear-representation problem.** It doesn't try to; the
  mainline path reuses native K/V geometry rather than proving linear concept
  directions.
- **DM survives all paraphrase robustness tests.** It doesn't; address/query
  surface form can still affect whether a bank slot is read.
- **DM matches RAG at multi-token open answers.** We test only single-
  token LAMA-TREx targets; multi-token decoding is open work.
- **Every ablation improves the bank.** W.2/W.3 demote LOPI/mHC-style guards to
  optional ablations unless a preregistered run shows a directional benefit.

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
