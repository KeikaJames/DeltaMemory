# Exp24 — Sparse-Attention Readout

**Verdict**: `SPARSE_PASS_DIRECTIONAL` — first positive retrieval signal in the program.

## Setup

The existing `AttnNativePatcher` already implements concat-softmax
`attn(q, [K_seq; M_K], [V_seq; α·M_V])` — bank K participates in native
attention competition. Two routing knobs:

- `bank_topk` — top-K bank slots before softmax (sparse routing).
- `bank_separate_softmax` — bank gets its own softmax, merged additively.

n = 100 facts × 2 seeds × 12 variants × 3 α ∈ {0.005, 0.05, 1.0}.
Bank: dual-site capture (K@relation_last, V@subject_last).

## Mean margin table

| variant | α=0.005 | α=0.05 | α=1.0 |
|---|---:|---:|---:|
| base | −4.87 | −4.87 | −4.87 |
| **full_bank_concat** | **−0.42** | **−0.28** | **−0.26** |
| full_bank_topk3 | −1.00 | −0.85 | −0.64 |
| full_bank_topk1 | −0.66 | −0.68 | −1.07 |
| full_bank_topk1_minus_correct | −0.86 | −0.85 | −1.10 |
| full_bank_topk1_shuffled_factids | −0.61 | −0.76 | −1.14 |
| full_bank_topk1_shuffled_layers | −4.68 | −4.68 | −0.48 |
| bank_separate_softmax_concat | −4.88 | −4.80 | −0.97 |
| bank_separate_softmax_topk1 | −4.86 | −4.74 | −0.49 |
| oracle (single-slot) | −2.29 | −2.46 | −3.97 |
| random_topk1 | −2.46 | −2.56 | −4.04 |

## Paired bootstrap (95% CI)

### Critical: does top-1 routing find the correct slot?

`full_bank_topk1 − full_bank_topk1_minus_correct`:

| α | mean diff | 95% CI | pass |
|---:|---:|---|---|
| 0.005 | **+0.193** | [+0.016, +0.391] | ✓ |
| 0.05 | **+0.175** | [+0.022, +0.345] | ✓ |
| 1.0 | +0.031 | [−0.079, +0.133] | ✗ |

**First time in this program** a correct-fact retrieval signal beats CI=0.
Removing the correct fact from the bank degrades top-1 routing by ~+0.18 nats.

### But: does fact-identity matching K↔V matter?

`full_bank_topk1 − full_bank_topk1_shuffled_factids`:

| α | mean diff | 95% CI |
|---:|---:|---|
| 0.005 | −0.054 | [−0.171, +0.062] |
| 0.05 | +0.083 | [−0.132, +0.293] |
| 1.0 | +0.065 | [−0.219, +0.367] |

**No.** Shuffling K/V identity within slots does not degrade. The retrieval
signal comes from **K-side routing alone** — V content is interchangeable.
This matches Exp17/21 (K-causal, V-causal independently) but tells us native
softmax routing extracts K's structure without ever using V's identity.

### Does sparse top-1 beat dense concat?

`full_bank_concat − full_bank_topk1`:

| α | mean diff | 95% CI |
|---:|---:|---|
| 0.005 | +0.245 | [−0.139, +0.630] |
| 0.05 | +0.400 | [+0.050, +0.748] |
| 1.0 | +0.813 | [+0.511, +1.137] |

**Concat still wins.** Dense softmax over all 100 slots beats sparse top-1.
Bank-presence steering remains the dominant lever; sparse routing only
recovers a small fraction of the steering benefit.

### Sanity: top-1 vs random single-slot

`full_bank_topk1 − random_topk1`: ✓ at all α (+1.80 to +2.97 nats).
Single-slot from a curated bank beats single-slot from random pool.

## Verdict: SPARSE_PASS_DIRECTIONAL

- **K-side fact identity is retrievable** via native softmax routing at small
  α (effect size +0.18 nats, CI lower bound > 0).
- **V-side fact identity is not used** — shuffled K/V pairing does not break
  retrieval, suggesting the bank's V acts as a non-discriminative bias term.
- **Steering still dominates routing** — full concat-softmax bank beats every
  sparse variant. Routing recovers <10% of bank-presence steering benefit.

## What this means for the original Exp24-30 ladder

- **K-causality replicates at bank scale** under sparse-attention readout.
- **Site-stratified V is not the right lever** — V identity is null both in
  Exp23 α-additive and Exp24 sparse-attention. The "subject_last V-causality"
  from Exp15 was V-as-bias, not V-as-content.
- **A new direction emerges**: maximize K-routing signal, treat V as constant
  steering. This is closer to a "fact-keyed bias library" than a memory.

## Next-step proposals (need user direction)

1. **Push K-routing signal**: hard-negative neighbors, alpha re-sweep at
   α∈{0.003, 0.007, 0.01, 0.02}, bank size {32, 64, 200, 807} to find where
   correct-vs-minus_correct gap maximizes.
2. **V-content recovery**: try writing V from larger spans (full prompt, not
   single token) — if V can carry content with more bandwidth, fact-identity
   K↔V matching might emerge.
3. **Accept and publish**: "Sparse-attention readout on native attention bank
   shows weak K-only retrieval (+0.18 nats), dominated by 4-nat steering
   effect. ANB is a steering library with weak retrieval, not memory."

## Artifacts

- `run_mps_exp24_smoke/cells.jsonl` — 7400 cells
- `run_sparse_attention.py` — runner using `bank_topk` + `bank_separate_softmax`
