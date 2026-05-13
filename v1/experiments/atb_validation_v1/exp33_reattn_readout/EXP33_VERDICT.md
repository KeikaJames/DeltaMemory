# Exp33 — Re-Attention Readout (Phase B): REJECTED

**Date**: 2026-05-13
**Verdict**: **REJECTED.** Joint-softmax sparse-attention readout reproduces
the Exp27 falsification pattern on the Exp31/32 splits. The bank produces
a large **fact-agnostic** margin uplift (logp(target_new) − logp(target_true)
moves from −3.83 → −0.03) but fails Gate D — shuffling which V belongs to
which fact gives *better* margins than the correct binding.

## Context

The Exp32 double-negative left two architectural questions open:

  - **Phase B** — would sparse-attention readout, where the bank K
    enters the model's own softmax instead of being added to residual,
    finally produce Gate B with fact identity?
  - **Phase C** — would parameter-edit (ROME/MEMIT) baselines confirm
    a known-good architecture *does* exist on this base model?

Phase B was nominally answered already by Exp27 (`EXP27_SPARSE_VERDICT.md`,
exp13_anb_readdressability) which fell from PASS at N=100 to FAIL at
N=200. Exp33 re-runs the joint-softmax sparse-attention bank on the
exact Exp31/32 splits as a clean control before declaring the
post-Exp32 plan closed.

## Setup

- Model: Qwen3-4B-Instruct-2507, MPS / bf16, 36 layers.
- Bank: `AttnNativeBank` with `bank_separate_softmax=False` (default
  joint softmax path, `attn_native_bank.py` lines 857-889).
- Bank size N = 200 (125 test facts + 75 train-set distractors).
- Write path: native `write_fact` — each fact passed through the model's
  own forward, K/V captured at the "period" position (Stage 13A default).
- Read path: each test paraphrase, last-token logits.
- Metric: margin = logp(target_new) − logp(target_true), averaged over
  2 paraphrases per fact, 125 facts.
- Single seed (seed=0); 3-variant comparison per α.

## Results — N = 200, single seed, 125 test facts

```
base (α=0, bank read with mass=0):  margin = −3.833
```

| α     | topk_full | minus_correct | shuffled_factids | Gate A=topk−minus | Gate D=topk−shuf |
|-------|----------:|--------------:|-----------------:|------------------:|-----------------:|
| 0.05  | **−0.034** | −0.056      | −0.044           | **+0.022**        | +0.010           |
| 0.10  | **−0.116** | −0.097      | **−0.008**       | −0.019            | **−0.108** ✗     |
| 0.30  | **−0.075** | −0.084      | −0.097           | +0.009            | +0.022           |

Reading the columns:

- **Absolute Gate B uplift is huge** — from −3.83 to −0.034 at α=0.05.
  Adding the bank pulls target_new and target_true to near-equal mass.
- **But the uplift is fact-agnostic.** At α=0.1 the shuffled-V variant
  (each fact reads someone else's V) gives a *better* margin (−0.008)
  than the correct binding (−0.116) — Gate D inverts by 0.108 nats.
- At α=0.05 the spreads across variants are all within ~0.02 nats —
  smaller than per-fact noise, signalling no real fact-identity binding.
- At α=0.30 the bank V is over-injected and absolute margins regress
  back toward base; variant ordering becomes inconsistent.

Interpretation: the bank lifts probability mass on *the population of
target_new tokens that any fact in the bank stores*. It does not lift
the specific target_new that matches the read query. The mechanism is
uniform steering bias, not routed memory.

## Cross-experiment table (4 independent falsifications + Exp33)

| Attack                    | Readout         | N=100 gates | N=200 gates       |
|---------------------------|-----------------|-------------|-------------------|
| Exp24 K-routing α-add     | additive        | DIRECTIONAL | weak              |
| Exp26 V@object_last       | additive        | A+C+D PASS  | A+C+D FAIL        |
| Exp26b multi-V α-add      | additive        | A+C+D PASS  | A+C+D FAIL        |
| Exp27 joint softmax α=0.05| joint softmax   | C+D PASS    | A weak / C+D FAIL |
| **Exp33 joint softmax**   | joint softmax   | (smoke +)   | **D FAIL**        |

Exp33 confirms the pattern on Exp31/32 splits with no change in shape.

## Combined with Exp31, Exp32 and the LS diagnostic

  - **Exp31 (H_A learned K-adapter)** — REJECTED. Gate B = 0/375.
  - **Exp32 (H_B MLP-side gated)** — REJECTED. Gate B = 0/375; Gate D
    failed by −1.17 logits; Gate E by −1.42.
  - **LS diagnostic** — InfoNCE's 45% shuffled-pair signal is
    dynamics-induced, not geometric. Honest routing ceiling on Q→K is
    76% test top-1 (rank-64 CCA). Closed-form on shuffled labels gives
    chance — data is not poisoned.
  - **Exp33 (Phase B sparse-attention)** — REJECTED. Large absolute
    uplift, but fact-agnostic. Gate D inverts at α=0.1.

The unifying conclusion is now **architectural, not routing-related**
and **not data-scale-related**:

> Bank-style external memory cannot inject fact identity into Qwen3-4B
> regardless of (a) routing protocol (cosine vs softmax-internal),
> (b) readout protocol (additive residual vs joint softmax),
> (c) injection site (attention vs MLP),
> (d) optimisation (InfoNCE vs closed-form LS), or
> (e) data scale (the routing channel works; the readout channel does
>     not extract the routed fact).

## Files

  - Driver: `run_exp33.py`
  - Eval cells: `run_qwen_full/cells.jsonl` (1,250 rows; 125 facts × 3 α
    × 3 variants × 1 seed + 125 base)
  - Summary: `run_qwen_full/summary.json`

## Reproducing

```bash
cd v1/experiments/atb_validation_v1/exp33_reattn_readout
python3 run_exp33.py \
    --model Qwen/Qwen3-4B-Instruct-2507 --device mps --dtype bf16 \
    --n-test 125 --n-distractors 75 \
    --alphas 0.05 0.1 0.3 \
    --out run_qwen_full
```

Runtime: ≈ 2.5 min on M-series MPS (200 write_facts ≈ 60 s, then 3 α ×
3 variants × 250 paraphrases ≈ 90 s).

## Next

Phase C (MEMIT-style parameter edit, Exp34) is the **positive control** —
a known-good architecture that *does* flip target tokens by directly
editing the MLP down_proj. If Exp34 succeeds where Exp31/32/33 failed,
the verdict locks: **for fact insertion on Qwen3-4B, parameter editing
works; external bank memory does not**.
