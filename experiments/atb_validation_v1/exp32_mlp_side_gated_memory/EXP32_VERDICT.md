# Exp32 — MLP-side Gated Memory: NEGATIVE Verdict

**Hypothesis (H_B):** ANB falsified in Exp23–27 not because the
captured trace is unroutable, but because *attention* is the wrong
injection site. ROME / MEMIT identify the MLP block as the locus of
factual associations. If we capture `(M_K = MLP-input, M_V = MLP-output)`
at the **relation_last** write position and inject through a learned
softmax-gated readout at the MLP output of every decoder layer, the bank
should produce routed memory rather than steering bias.

**Result:** **REJECTED.**

## Experimental design (Qwen3-4B-Instruct-2507, MPS / bf16)

| Stage | What | Outcome |
|---|---|---|
| Φ1 — capture | (K, V) per layer at relation_last, paraphrase Q at last token; train/val/test = 567/115/125 facts | 567 + 115 + 125 saved cleanly |
| Φ2 — train router | Per-layer `W_q ∈ ℝ^{2560→64}` + scalar gate `W_g`. Dual loss: InfoNCE on cosine routing + MSE on gated V-readout. AdamW lr=1e-3, 50 epochs, full-batch. 3 seeds + 1 Gate-E shuffled-pair control. | **val_top1 = 92.61% / 91.74% / 92.17%** (~106× chance); shuffled-pair control: 50.87% |
| Φ3 — model-attached eval | n=125 test facts × 3 seeds × bank N=200 (125 test + 75 distractors) × α ∈ {0.1, 0.3, 1.0} × 8 variants | 9,375 cells |

## Five-gate panel (mean margin = logP(target_new) − logP(target_true))

| variant | α=0.1 | α=0.3 | α=1.0 |
|---|---:|---:|---:|
| **base (no bank)** | −4.691 | — | — |
| mlp_gate_off (hook installed, gate=0) | −4.691 | −4.691 | −4.691 |
| **mlp_full_learned** (main) | −4.741 | −4.768 | **−4.549** |
| mlp_topk1_learned | −4.731 | −4.759 | −4.584 |
| mlp_fixed_gate (gate≡1) | −4.738 | −4.752 | −4.570 |
| mlp_minus_correct_learned | −4.737 | −4.681 | −3.781 |
| mlp_meanV_learned (Gate C) | −4.692 | −4.623 | −3.898 |
| mlp_shuffled_factids_learned (Gate D) | −4.674 | −4.462 | −3.383 |
| mlp_shuffled_router_learned (Gate E) | −4.725 | −4.670 | **−3.128** |

### Gate-by-gate (α = 1.0, the only α where the bank moves the model)

| Gate | Spec | Result | Verdict |
|---|---|---|---|
| **A** vs base | full_learned (−4.549) > base (−4.691) | +0.142 | trivial bias, not routing |
| **B** retrieval top-1 | 0 / 375 across all variants and α | **0.00%** | FAIL (chance = 0.50%) |
| **C** vs meanV | full_learned (−4.549) vs meanV (−3.898) | **−0.651** | FAIL (meanV wins) |
| **D** vs shuffled_factids | full_learned (−4.549) vs shuffled (−3.383) | **−1.166** | FAIL |
| **E** vs shuffled_router | full_learned (−4.549) vs shuf_router (−3.128) | **−1.421** | FAIL (Gate E catastrophic) |

## What this means

1. **Embedding-space retrieval is now near-saturated.** 92% top-1 on
   N≈115 paraphrase retrieval is a 100×-chance signal — the
   per-layer routing space is highly discriminative when probed
   directly.
2. **None of that signal transfers to the LM logit head.** Top-1
   target_new hit rate is **0 / 375** for every variant at every α,
   *including* oracle-style settings.
3. **Gate E is the killer.** A router trained on *deliberately wrong*
   (fact, paraphrase) pairs *outperforms* the correctly-trained one by
   **1.42 logits** of margin. The LM head cannot distinguish "correct
   fact retrieved" from "any vaguely matched MLP-output activation
   pattern injected at the relation site."
4. The "minus_correct" variant (bank with the correct fact deleted)
   beats the full-bank variant by **0.77 logits**. The bank therefore
   contributes activation bias, not memory.

## Combined with Exp31

| | Exp31 (attention-side) | Exp32 (MLP-side) |
|---|---|---|
| Site | pre-RoPE K + post-norm V, attention residual | MLP input/output, MLP residual |
| Adapter | InfoNCE over per-layer Linear (W_K) | InfoNCE over per-layer Linear + learned gate |
| Embedding val top-1 | ~40× chance | ~106× chance |
| LM-output top-1 (Gate B) | 0 / 375 | 0 / 375 |
| Gate E (shuffled-pair control) | FAIL at α=0.003, 0.010 | FAIL at α=1.0 (Δ=−1.42) |
| Verdict | H_A rejected | H_B rejected |

## Interpretation

The ATB failure mode is **architectural, not representational**. Two
mutually orthogonal hypotheses — (A) K-space discriminability is the
blocker, (B) attention is the wrong site — both fail decisively on
the same diagnostic pattern: strong in-isolation routing, zero
LM-output retrieval, controls beating the learned signal.

What both Exp31 and Exp32 share is the **α-scaled residual readout
protocol** (`h ← h + α · readout(bank)` injected as an additive bias
into the residual stream after the chosen sub-layer). At fact-bank
scale (N ≥ 100), this readout produces:

- **detectable activation drift** (Gate A non-zero at large α)
- **no fact identity coupling** at the LM head (Gates B–E flat or
  inverted)

independent of where the bank is read from and independent of how
discriminative the router is in its own embedding space.

## Status

H_A (learned K-adapter) — REJECTED (Exp31).
H_B (MLP-side gated memory) — REJECTED (Exp32).

The cross-architecture replication step in the plan is **conditional
on Qwen3 showing any LM-output signal**. With Gate B at 0/375 and
Gates D/E catastrophically inverted, replication on Gemma-4-E2B and
Mistral-7B would only reproduce the same null. We therefore close the
Exp31/Exp32 line as a double-negative and update the public docs to
record both falsifications.

## Files

- Capture: `data/cache/Qwen_Qwen3-4B-Instruct-2507_{train,val,test}.pt`
- Trained routers: `run_qwen_full/seed{0,1,2}/router_best.pt`,
  `run_qwen_full/shuffled/router_best.pt`
- Eval cells: `run_qwen_full_eval/cells.jsonl` (9,375 rows)
- Env: `run_qwen_full_eval/env.json`
