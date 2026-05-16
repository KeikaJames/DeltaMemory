# E06 — Relation-disjoint OOD split

**Status**: **PASS** on driver criterion (Δ_OOD ≤ −1.0) but the result must be read alongside the adapter-not-memory interpretation that emerged from e10/e11/e16/e17.

**Headline**: With a strict 70/30 relation split (train relations ∩ test relations = ∅), Δ on held-out relations is **−4.37 nat** vs Δ on train relations **−5.32 nat**. The 18% gap suggests modest relation-specificity but the bulk of the effect transfers.

---

## a. Reproduction command

```bash
python3 v2/experiments/e06_relation_disjoint_ood/run.py --seed 0 \
    --bank_layer 9 --rank 64 --steps 200 \
    --n_train 120 --n_eval 120 --n_preload 512 --train_frac 0.7
```

## b. Seeds & sample size

seed 0; relation split train_frac=0.7 (train/test relation sets disjoint); n_train=120, n_test_ood=120; n_preload=512; steps=200.

## c. Raw data paths

`v2/experiments/e06_relation_disjoint_ood/e06_seed0.json`

## d. Numbers

| split | Δ NLL |
|---|---:|
| train relations | −5.322 |
| **test relations (relation-OOD)** | **−4.370** |
| gap (OOD − train) | +0.952 (~18%) |

Verdict object: `{pass: True, rule: 'Δ NLL on OOD test_rel ≤ -1.0'}`

## e. Verdict

- **Hypothesis ("the projector does not just memorize train relations")**: Δ_OOD = −4.37 ≪ −1.0 threshold. ✅
- **Pass rate**: 1/1.
- **Not a falsifier**; consistent with the adapter-as-distributional-shift reading: most of the gain is template/grammar-driven, which is the part that transfers across relations.

## f. Caveat

- Single seed.
- E16-forgetting (A/B symmetry across 4 runs) shows the projector applies a similar lift to *both* trained and arbitrary-untrained banks. The 18% Δ gap here is consistent with that picture: relations the projector hasn't specifically seen still get most of the boost because the boost is largely template-conditional, not relation-conditional.
- E17 (negation/wrong-target) shows the same template-conditional behavior — wrong targets still get −2.66 nat lift on the canonical template.

## g. Implications

- E06 isolated *would* support the memory thesis. Read in context with e10/e11/e16/e17, it instead supports the adapter-with-template-prior interpretation.
- Useful as a *non-falsifier* control: confirms the mechanism is not catastrophically train-relation-specific.
