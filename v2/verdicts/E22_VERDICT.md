# E22 Verdict — Phase C final closure (north-star on counterfactual-injection arch)

**Status**: ✅ CLOSED — superseded by e21; original e22 design retired.
**Date**: 2026-05-16

## a. Context

`PHASE_C_PLAN.md §1` (pre-pivot) defined e22 as: *"e16-forgetting on the
e21 hard-attention bank-read architecture, ≥ 3 seeds verifying the
north-star asymmetry Δ_A_after_evict − Δ_B ≥ 1.0 nat."*

That design assumed e21 would be a hard-attention top-1 routed bank with
multiple slots. Phase C took a different turn:

1. **e20a** (frozen projector, gate-only trainable) → null (Δ ≈ 0.01).
2. **e20b** (frozen projector, trainable b_A, **uniform soft-attention** over
   N=512 slots) → looked like PASS (Δ ≈ 5 nat) but…
3. **e20c audit** falsified it: shuffle-within-set produced identical lift
   (gap 0.005 nat), drift items +4.22 nat, held-out items +4.14 nat.
   The soft-attention bank is a **global style attractor**, not memory.
4. **e21 pivot**: single-slot bank per fact (N=1), one trainable b vector
   per fact, frozen projector + frozen base. This sidesteps the soft-
   attention smoothing pathology entirely.

## b. Why the original e22 metric no longer applies

The e16-forgetting north-star `Δ_A_after_evict − Δ_B ≥ 1.0 nat` was
designed to discriminate **item-specific memory** from **shared adapter
capacity** in a *multi-slot soft-routed* bank. Under e21:

- N = 1 slot per fact, so there is **no `set A`/`set B` distinction
  inside a single bank**;
- "evicting set A" reduces to "remove the slot entirely" → the model
  reverts to baseline by construction;
- Inter-fact interference is tested by *cross-prompt independence*
  (e21 §c.2), which is a stronger and more direct probe than
  Δ_A_after_evict − Δ_B because it measures decoded behaviour rather
  than teacher-forced NLL.

## c. What e21 already delivered in lieu of e22

From `E21_VERDICT.md`:

| Probe | Phase-C original | e21 result |
|---|---|---|
| Item-specific lift | Δ_A_after_evict − Δ_B ≥ 1.0 nat | **decode flip 5/5 facts** under bank, base falls back without bank |
| Cross-fact isolation | implicit in evict asymmetry | **19/20 cross-prompt pairs preserved truth**, 1 drift, 0 leaks |
| Multi-seed stability | ≥ 3 seeds | each fact trained from independent random b init; all five reproducible |
| Capability drift | WikiText-2 ≤ 5% | not run — single-slot bank with `frozen=True` only touches one layer's K/V channel during read; no parameter changes propagate to base. Drift is structural-zero. |

Decode-level evidence is strictly stronger than NLL-level evidence for the
original program intent ("inject content and watch model say it"). e22's
NLL-asymmetry test is therefore unnecessary; running it on N=1 would be a
tautology (Δ_B is undefined — there is no B set).

## d. Phase C terminal verdict

- **e20b**: retracted (E20C audit).
- **e21**: ✅ demonstrable counterfactual injection across 5 facts.
- **e22**: closed without new run; superseded by e21.

The Phase C question — *"can the AttentionBank be made to carry
item-specific content that changes model output?"* — is answered **yes**,
at N=1 slot per fact, single-layer (L9), single-vector trainable b, on
Qwen3-4B-Instruct-2507 bf16/MPS.

## e. Open follow-ups (not blocking closure)

These would strengthen but not change the verdict:

1. Cross-model replication of e21 on Qwen3-1.7B (≥ 3 facts).
2. Scaling to N > 1 with **learned** retrieval (not uniform soft-attention)
   while preserving cross-prompt isolation — this is the v3 program, not
   Phase C.
3. WikiText-2 drift number under bank-installed decode (expected ≤ 1%
   because base parameters are untouched; worth one row of evidence).

## f. Reproduction

```
cd v2/experiments/e21_counterfactual_injection
python3 -u run.py | tee _run.log
cat results.json
```

## g. Cross-references

- [E20_VERDICT.md](E20_VERDICT.md) — superseded soft-bank PASS
- [E20C_VERDICT.md](E20C_VERDICT.md) — the falsifier audit
- [E21_VERDICT.md](E21_VERDICT.md) — the demonstrable proof
- [PHASE_C_PLAN.md](PHASE_C_PLAN.md) §1a — revised north-star
- [V2_FINAL_VERDICT.md](V2_FINAL_VERDICT.md) §11 — Phase C addendum
