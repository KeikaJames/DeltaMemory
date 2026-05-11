# Exp23 — Site-Stratified Oracle Ceiling

**Verdict**: `DUAL_ORACLE_FAIL`

**Hypothesis**: dual-site capture (K@relation_last, V@subject_last) with
oracle slot selection beats every control.

**Result**: falsified. With n=100 × 2 seeds at α=0.005:

| variant | mean margin |
|---|---|
| old_full_bank | **−0.27** |
| minus_correct | −0.33 |
| shuffled_fact_ids | −0.41 |
| relationK_randomV | −2.28 |
| **oracle_relationK_subjectV** | **−2.29** |
| oracle_relationK_relationV | −2.34 |
| randomK_subjectV | −2.44 |
| random_relationK_subjectV | −2.46 |
| oracle_subjectK_relationV | −3.25 |
| oracle_subjectK_subjectV | −3.26 |
| shuffled_layers | −4.32 |
| base | −4.87 |

### Paired bootstrap (95% CI on `oracle_relationK_subjectV − x`)

| contrast | mean diff | 95% CI | pass |
|---|---:|---|---|
| − base | +2.58 | [+2.02, +3.16] | ✓ (generic steering) |
| − old_full_bank | −2.02 | [−2.61, −1.46] | ✗ |
| − random_relationK_subjectV | +0.17 | [−0.22, +0.55] | ✗ |
| − relationK_randomV | −0.02 | [−0.08, +0.05] | ✗ |
| − randomK_subjectV | +0.14 | [−0.24, +0.52] | ✗ |
| − minus_correct | −1.97 | [−2.52, −1.43] | ✗ |

### Interpretation

1. **Bank presence steers margin** by ~2.6 nats vs base. Any bank works.
2. **Bank size dominates** fact identity: full N≈100 bank beats single-slot
   oracle by 2 nats. Removing the correct slot (`minus_correct`) does not
   degrade the win — i.e. the correct slot is not pulling weight.
3. **K identity is null**: `oracle_relationK_subjectV ≈ randomK_subjectV`
   (CI crosses 0).
4. **V identity is null**: `oracle_relationK_subjectV ≈ relationK_randomV`
   (CI tight around 0).
5. `shuffled_fact_ids ≈ old_full_bank` — mixing fact identities at K/V level
   does not break the steering, confirming **the bank is acting as a global
   activation drift, not a routed memory**.

### Why the prior K-causality (Exp17/21) does not transfer

Exp17/21 showed `relation_last` K is causal **in isolation** at small banks
(k=1 with controlled controls). Here the K signal is drowned by the
generic activation steering that any non-empty bank produces. The contrast
that worked at k=1 vanishes at single-slot oracle vs full-bank because the
full-bank steering term is a much larger lever than the routing term.

### Decision: pivot to **Exp31 (learned read-time key adapter)**

The natural Exp24 routing (next on the original ladder) cannot recover from
this. If oracle (perfect routing!) loses to random selection by 0.0–0.2 nats,
natural QK routing will be at best indistinguishable. The contingency tree
sends us to Exp31: learn `A: d_q → d_k` so that `A(q_rel) · M_K[*, rel_last]^T`
forms a discriminative score. If even with a learned adapter we cannot
separate correct from random, ATB-native memory is empirically false and we
exit to RAG-hybrid territory (Exp36).

### Artifacts

- `run_mps_exp23_smoke/cells.jsonl` — 2400 cells (100 facts × 12 var × 2 seeds)
- `run_mps_exp23_smoke/analysis.json` — bootstrap CI table
- `run_mps_exp23_smoke/verdict.txt` — `DUAL_ORACLE_FAIL`
- `run_mps_exp23_smoke/manifest.json`, `env.json`
