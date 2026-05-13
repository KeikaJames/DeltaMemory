# Exp26b verdict — Multi-token V capture falsified

## Verdict

**EXP26b_PASS_AT_N100_FAIL_AT_N200** — Same Exp26→Exp27 falsification pattern,
third independent confirmation that the α-additive native injection architecture
has a hard ceiling at N≈100.

## What we changed

V capture site went from a single token (Exp26 V@object_last) to a span
mean over `[subject_first .. object_last]`. Mean V span length: ~8 tokens.
K@relation_last unchanged. No learned parameters. Native attention only.

## Results

### N=100, 3 seeds, α=0.005 — ALL GATES PASS

| Gate | Contrast | diff | CI95 | verdict |
|---|---|---|---|---|
| A | topk1 − minus_correct      | +0.447 | [+0.292, +0.618] | PASS |
| C | topk1 − meanV              | +0.133 | [+0.033, +0.238] | PASS |
| D | topk1 − shuffled_fact_ids  | +0.172 | [+0.086, +0.262] | PASS |

retrieval_accuracy = 1.0% (chance = 1.0%, SE 0.6%). V-identity matters
despite null argmax accuracy — signal lives in the softmax weights, not
top-1 selection.

### N=200 falsification — ALL GATES COLLAPSE

| Gate | α=0.005 | α=0.010 |
|---|---|---|
| A | diff=−0.022 [−0.10, +0.06] FAIL | diff=+0.009 [−0.08, +0.10] FAIL |
| C | diff=−0.064 [−0.14, +0.01] FAIL | diff=+0.036 [−0.02, +0.09] FAIL |
| D | diff=−0.035 [−0.10, +0.03] FAIL | diff=−0.050 [−0.12, +0.02] FAIL |

retrieval_accuracy = 1.0–1.5% (chance = 0.5%). bank_mass ≈ 0.38 (still
heavy steering).

## Pattern across program

Three independent attacks on the K-routing/V-readout problem, three
identical falsification curves:

| Exp | Attack | N=100 gates | N=200 gates |
|---|---|---|---|
| Exp24 sparse | K@relation_last full bank | DIRECTIONAL (+0.193 nat) | weak/null |
| Exp26 single V | K@relation, V@object_last | A+C+D PASS_STRONG | All FAIL (Exp27) |
| Exp26b multi V | K@relation, V@subj_to_obj | A+C+D PASS | All FAIL |

retrieval_accuracy never escapes the 2× chance ceiling at N=100 and falls
to ~1× chance by N=200, irrespective of how V is captured.

## Why this is a ceiling, not a tuning problem

The α-additive native injection equation is:

    h_out ← h_seq + α · σ(q · M_K^T) · M_V

The cosine routing σ(q · M_K^T) is what we tuned. We tried:
- richer V site (Exp26 V@object_last)
- richer V span (Exp26b V@subj_to_obj, 8-token mean)
- K-routing only (Exp24)

None lifts retrieval_accuracy above chance for N≥200, because the bank
is **added** to the residual rather than competing for attention mass
inside the softmax. The bank can steer the residual (Gate A) but cannot
re-address it.

This is the same conclusion the user reached after Exp23 — the bug is
the readout architecture, not the K/V capture sites.

## Forward path

Two genuinely different attack vectors remain:

1. **Sparse-attention readout** — put `M_K` into the actual softmax with
   the sequence keys: `Attn(Q, [K_seq; M_K], [V_seq; M_V])`. This forces
   the bank to compete with sequence keys instead of being added to the
   output. The user's directive "K-routing 做稳，再攻 V-content" was tested
   above and failed; sparse-attention is the unattempted lever.

2. **Learned read-time key adapter** (Exp31 in plan) — train a small
   `A: q_relation → k_bank` linear map so that `A(q) · M_K^T` separates
   correct from random with high margin. The cosine on raw `q · M_K^T`
   has been shown insufficient three times.

Both are open. Path 1 is native; path 2 is minimally parametric.

Recommend doing path 1 next (it is what the original Exp24 sparse-attention
plan called for before we got distracted by the +0.193 K-routing crack).
