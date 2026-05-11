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

### α-sweep confirms structural failure

Tested α ∈ {0.0005, 0.001, 0.005, 0.02} on n=50 × 2 seeds:

| α | base | old_full_bank | oracle_relationK_subjectV | minus_correct | oracle − full |
|---:|---:|---:|---:|---:|---:|
| 0.0005 | −5.76 | −0.20 | −2.26 | −0.15 | **−2.06** |
| 0.0010 | −5.76 | −0.08 | −2.23 | +0.04 | **−2.15** |
| 0.0050 | −5.76 | −0.14 | −2.22 | −0.20 | **−2.08** |
| 0.0200 | −5.76 | −0.00 | −2.25 | −0.02 | **−2.25** |

**The 2-nat oracle deficit is α-invariant**. This is not an injection-strength
problem — it is a **substrate-capacity** problem. A single slot's K/V cannot
match the activation steering of a 100-slot full bank at any α.

### Decision: site-stratified KV direction is structurally falsified

Originally the contingency tree pointed to **Exp31 (learned adapter)** as the
fallback. But the α-sweep shows the gap is downstream of K matching — even
**perfect routing** (oracle) can't close it, because the issue is that a single
slot's V is too small a lever vs. the bank-presence steering term. A learned
adapter (which only improves K matching) cannot fix V-side capacity. Exp31
would therefore be a known-null run.

**Implications for the rest of Exp23–40**:

- **Exp24** (natural routing) — strictly weaker than oracle, already dead.
- **Exp25** (swap matrix) — every cell would test a single-slot variant; all dead.
- **Exp27** (α stability) — already done above; no α range exists where the
  method works.
- **Exp31** (adapter) — solves wrong problem; deferred unless V-side fix found.
- **Exp36** (RAG comparison) — user has explicitly rejected RAG-hybrid as a
  pivot direction (ATB philosophy is native memory).

**Scientific conclusion** (locked):

> The K-causality observed in Exp17/21 (relation_last is K-causal in single-fact
> banks) does **not** compose to bank-level retrieval. At full bank scale,
> activation-steering term dominates routing term by >2 nats independent of α.
> Native single-slot retrieval in a 100-fact bank is empirically infeasible on
> Qwen3-4B with this capture/inject architecture.

**Remaining non-dead directions** (require user input):

1. **V-side amplification** — make per-slot V contribution match full-bank
   energy. Equivalent to learning a per-slot scale at write time. New experiment
   (call it Exp23v: V-side scaling). Borderline native.
2. **Sparse attention bank** — instead of α-scaled additive injection, use
   a softmax over slots that competes with native KV. Closer to in-context
   attention. Architectural change. (Was implicit in Exp24 spec but with α-add
   readout — needs full attention-replace readout.)
3. **Pivot to a different substrate** — Gemma/Llama may have different
   activation-steering vs routing balance. Was Exp29.
4. **Accept the negative result** — publish "site-stratified native ANB does
   not scale beyond k=1" as a clean negative.

### Artifacts

- `run_mps_exp23_smoke/cells.jsonl` — 2400 cells (100 facts × 12 var × 2 seeds)
- `run_mps_exp23_smoke/analysis.json` — bootstrap CI table
- `run_mps_exp23_smoke/verdict.txt` — `DUAL_ORACLE_FAIL`
- `run_mps_exp23_smoke/manifest.json`, `env.json`
