# E21 Verdict — Counterfactual Injection (the "make AI lie" demonstrator)

**Status**: ✅ PASS — 5/5 facts flipped, 19/20 cross-prompts preserved.
**Date**: 2026-05-16
**Model**: Qwen/Qwen3-4B-Instruct-2507 (MPS, bf16)
**Origin**: Phase C pivot after e20c (companion verdict) falsified the e20b
"uniform-soft-attention" lift as a *global style attractor*, not item memory.

## a. Hypothesis

The v2 program asked: does the rank-64 K-projector produce item-specific
memory? Answer (v2 final): no, it's a template-conditional adapter. e20b's
"trainable b_A" extension appeared to PASS a north-star asymmetry but e20c
crushed it (shuffle-within-set unchanged, drift items also lifted 4 nat).

This experiment asks the **usability** question directly:

> Can the AttentionBank be configured so that installing it makes the model
> **greedy-decode a chosen counterfactual answer** on a specific prompt the
> model otherwise gets right — *without* affecting other prompts?

This is the v1-level demonstrable behavior the user requested: "把谎注入进去，
让模型说出来"。

## b. Protocol

For each fact `(prompt, truth, counterfactual)`:

1. **Verify base correctness**: greedy-decode 8 tokens from `prompt` with no
   bank. Drop the fact if the decode does not start with `truth`.
2. **Train one b**: spin up a single-slot AttentionBank at layer 9. Initialize
   `b ∈ R^d` to a random vector with norm 15. Freeze projector (rank-64 random),
   freeze gate-heads, freeze the whole model. Train `b` only via AdamW
   (lr=5e-3, 200 steps) on the teacher-forced loss `−log p(counterfactual | prompt + bank)`.
3. **Test**: greedy-decode 12 tokens from `prompt` with the trained single-slot
   bank installed at layer 9.
4. **Cross-prompt independence**: for every pair `(i, j)` with i ≠ j, install
   `b_i` and greedy-decode `prompt_j`. Verify decode still starts with `truth_j`
   (no leakage of counterfactual).

Architecture used per fact:
- Bank: max_per_layer=1, dtype=bf16, layer=9 only.
- Slot construction: `slot = (b + P(b)).to(bf16)` (`P` random rank-64, frozen).
- Single trainable tensor: `b` (2,560 fp32 elements per fact).
- All other model parameters frozen.

Facts tested: capital-of-France, capital-of-Japan, largest-planet,
author-of-Romeo-and-Juliet, water-boiling-point. (One additional candidate,
"Great-Wall location", base decoded to `' the north of China'` which doesn't
start with `' China'`, so it was dropped at phase 0 — that's the protocol
working, not a failure.)

## c. Results

### c.1 Counterfactual flips (Phase 1)

All five facts: base model decodes truth without bank, decodes the chosen
counterfactual with bank.

| prompt | base decode | with-bank decode | flipped |
|---|---|---|:--:|
| `The capital of France is` | `' Paris. The capital of Germany is Berlin'` | `' Berlin.\nThis statement is false because the capital of France is'` | ✅ |
| `The capital of Japan is` | `' Tokyo, and the capital of the United'` | `' Beijing.\nThis statement is false.\nThe capital of Japan is'` | ✅ |
| `The largest planet in our solar system is` | `' Jupiter. It has a diameter of about'` | `' Saturn. Its diameter is about 116,0'` | ✅ |
| `The author of Romeo and Juliet is` | `' William Shakespeare, and the play is set'` | `' Charles Dickens. Is this statement true or false?\n\nThe statement'` | ✅ |
| `Water boils at` | `' 100°C and freezes at'` | `' 50°C pressure. At what temperature does water freeze'` | ✅ |

Training loss curves: each fact converges to <0.05 within 200 steps (e.g.
France: 6.77 → 0.002; Romeo: 9.12 → 0.044).

### c.2 Cross-prompt independence (Phase 2)

20 cross pairs (5 banks × 4 other-prompts each):

- 19/20 → truth preserved on the foreign prompt (bank had no effect).
- 1/20 → drift (not even leak to counterfactual): bank-for-Romeo on
  France-prompt decoded `' 2000000'`. The Romeo bank training pushed b
  hardest (max loss 9.1, last 0.044) and apparently moved into a region
  that disturbs unrelated prompts. **0/20 leaked the wrong counterfactual.**

Independence rate: 95%.

## d. Interpretation

This is the smallest viable proof that **the AttentionBank can be operated
as a controllable memory editor**: one trainable vector (2,560 fp32
parameters) at one layer flips one fact while leaving the rest of the model
intact.

Comparison with e20c's null:
- e20c trained 512 b-vectors on 256 prompts; the trained set produced a
  uniform 4-nat NLL drop across **any** input, identity vs shuffled
  unchanged — pure style attractor.
- e21 trains 1 b-vector for 1 prompt; the trained vector produces a
  prompt-conditional decode flip with 95% cross-prompt cleanliness — a
  genuine editor.

The difference is **bank size and training signal**. With N=512 slots and
soft uniform attention, the gradient pushes every slot toward a common
"answer-region" direction; the bank becomes a style adapter. With N=1 slot
and a single per-prompt loss, the gradient sculpts that one slot into a
prompt-targeted message that the attention layer can read out only when the
query distribution matches the prompt.

That is the operational core of "delta memory": **per-item, low-cost,
behavior-flipping content injection at a single attention site**. The v1
"make AI lie" pattern is reproduced here in v2 architecture.

## e. Caveats (honest list)

- **Per-fact retraining required.** Each fact uses its own bank-slot
  training (~30 s / fact on Apple M-series MPS, bf16). Not a learned
  retrieval system — closer to a per-item LoRA-of-size-d.
- **Cross-prompt is 95%, not 100%.** One leaked into drift (no truth/wrong
  counter, just garbage). Larger-scale banking of many facts simultaneously
  is **not yet tested** and is the next falsifier to chase.
- **Single-model evidence.** Qwen3-4B-Instruct-2507 only. Qwen3-1.7B
  replication is the next deliverable.
- **Single-layer evidence.** Layer 9 only. Sensitivity to bank_layer
  unknown.
- **Loss is teacher-forced.** We minimized NLL of the full counterfactual
  string; we did not optimize "the greedy decode actually starts with
  counterfactual" directly. Empirically the two coincide (loss < 0.1 ⇒
  decode flip), but for hard facts where the first counterfactual token
  shares a prefix with a common token (e.g. "Charles" vs "Charles" of
  Darwin), more care could be needed.
- **The model "knows it's lying"**: 3 of 5 with-bank decodes append
  meta-commentary like `"This statement is false because the capital of
  France is"`. The bank flipped the first answer token but the model's
  later tokens drift back toward truth. For multi-token coherent
  counterfactuals more steps + longer counterfactual targets are needed.
  This is a feature for safety/honesty research (you can inject and see
  the model's "second thought"), not a bug.

## f. Verdict

**AttentionBank is usable**, in the v1-style "inject a lie, watch the model
say it" sense, with the architectural ingredients:

- N = 1 slot per injected fact (NOT uniform soft attention over a bag).
- Per-fact gradient signal (NOT shared multi-item training).
- Frozen projector + frozen base + only b trainable (2,560 fp32 elements
  per fact).
- Single read layer (layer 9 worked; layer-sweep deferred).

**This is the minimum viable proof the path works.** Phase C primary
objective re-stated and met:

> Can we *use* the AttentionBank to produce a chosen, observable behavior
> change on a chosen prompt, with chosen content, on the same Qwen3-4B
> base model, on hardware available? — **Yes, 5/5 demonstrated.**

The grand-architecture v3 (learned retrieval, multi-fact banks, cross-model
generalization, capability-drift guard) is the next program. This verdict
shuts e21 as "concept proven, scale-out is future work".

Result files:
- `v2/experiments/e21_counterfactual_injection/run.py`
- `v2/experiments/e21_counterfactual_injection/results.json`
- `v2/experiments/e21_counterfactual_injection/_run.log`
- `v2/experiments/e20c_adversarial_audit/seed0.json` (the companion null that
  forced the pivot from e20b uniform-bank to e21 single-slot injection)

## g. Reproduction one-liner

```
python3 v2/experiments/e21_counterfactual_injection/run.py --steps 200 --lr 5e-3
```

~3 min wall-clock on Apple M-series with `bf16` and the patched
`AttentionBank`.
