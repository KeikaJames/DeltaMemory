# Exp27 — V@object_last full validation: VERDICT

**Status:** `EXP27_FAIL — N=100 V@object_last PASS_STRONG was a small-bank artifact.`

## Summary

The PASS_STRONG result from Exp26 (V@object_last, N=100, α=0.010,
A=+0.347, C=+0.200, D=+0.249) **does NOT replicate at N=200**.

## Replication runs

### Exp27a (single α at full validation depth)

- N=200, n=200, α=0.010, 3 seeds.
- 759 s MPS, 3600 cells.

| Gate | diff | CI | Verdict |
|---|---|---|---|
| A: topk1 − minus_correct | −0.052 | [−0.139, +0.037] | overlaps 0 |
| C: topk1 − meanV | −0.053 | [−0.121, +0.016] | overlaps 0 |
| D: topk1 − shuffled_factids | **−0.122** | **[−0.203, −0.043]** | **CI excludes 0, negative direction!** |

### Exp27b (α sweep at N=200)

- N=200, n=200, α ∈ {0.003, 0.005, 0.007, 0.010, 0.015, 0.020}, 3 seeds.
- 18,600 cells.

| α | Gate A | Gate C | Gate D |
|---|---|---|---|
| 0.003 | +0.005 [−0.06, +0.07] | −0.028 [−0.09, +0.03] | −0.039 [−0.10, +0.02] |
| 0.005 | −0.041 [−0.13, +0.05] | −0.046 [−0.13, +0.03] | +0.006 [−0.07, +0.08] |
| 0.007 | −0.070 [−0.17, +0.02] | −0.012 [−0.08, +0.05] | +0.037 [−0.03, +0.11] |
| 0.010 | −0.052 [−0.14, +0.04] | −0.053 [−0.12, +0.02] | **−0.122 [−0.20, −0.04]** |
| 0.015 | +0.092 [+0.02, +0.16] | −0.082 [−0.17, +0.00] | −0.051 [−0.14, +0.04] |
| 0.020 | +0.086 [−0.01, +0.19] | +0.019 [−0.08, +0.12] | +0.054 [−0.06, +0.17] |

retrieval_accuracy collapses to **1.0× chance (0.005 = 1/200)** across all
α (vs 2× chance at N=100). The K-routing signal is **bank-size limited**.

## Interpretation

The N=100 V@object_last PASS_STRONG verdict was an artifact of:

1. **Bank size N=100**: K-routing accuracy was 2× chance (0.020 vs 0.010).
   At N=200 this drops to 1× chance — pure noise.
2. **Sample-size variance** at n=300 paired tests was high enough that
   +0.347 could appear by chance.

The pattern is now consistent across all bank-size and V-site sweeps:

- Retrieval accuracy hovers at 1–2× chance, decaying with N.
- Margin gates (A, C, D) are not replicable across N.
- The signal is "there's something at N=100" — likely a bank-size dependent
  steering aliasing — not "fact retrieval".

## Verdict for the Exp23→27 chain

| Experiment | Original verdict | Falsified by |
|---|---|---|
| Exp23 oracle ceiling | FAIL (α-additive) | (correctly identified) |
| Exp24 sparse-attention | DIRECTIONAL (+0.193 N=100) | Exp25 bank-size sweep |
| Exp25 K-routing | FRAGILE_K_ROUTING | self (showed non-monotone) |
| Exp26 V@object_last | PASS_STRONG (N=100) | **Exp27 N=200 replication** |
| Exp27 full validation | **FAIL** | — |

The site-stratified sparse-attention ANB with native softmax readout does
NOT scale beyond small banks (N≲100). The apparent victory in Exp26 was
a small-N artifact.

## What is robust

1. **K@relation_last has weak K-causality** (Exp17, Exp21) — well-established.
2. **Retrieval accuracy ≈ 2× chance at small banks** — small but real.
3. **V@object_last gives the cleanest signal among V-sites at N=100** — even
   if it doesn't survive N=200, the relative ordering across V-sites is
   informative for future work.

## Next decision points (for the user)

Three honest paths:

1. **Accept negative result for native sparse-attention ANB at any
   useful scale.** Write up "K-causality without scalable retrieval" as
   the final program finding. Pivot architecture.

2. **Try learned key adapter** (Exp31 in old plan): train small linear
   `A: d_q → d_k` so `A(q) · M_K` separates correct from random with
   high confidence. Native attention math, but with one trained matrix.
   The user previously expressed reservations about non-native
   components, but this preserves the attention readout structure.

3. **Multi-token V capture** (Exp26b): aggregate V over a span instead
   of one position. Doesn't change the architecture, only the capture
   protocol. Could lift the retrieval ceiling without learned params.

The 2× chance retrieval at small banks is a small but consistent
substrate signal. It needs either capacity or a learned routing layer
to become a working memory; the data here cannot prove either is
sufficient at scale by current means.
