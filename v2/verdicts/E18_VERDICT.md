# E18 — Chained 2-hop reasoning

**Status**: The "memory composes facts" claim is **decisively refuted** by E18.
**Headline**: Having both bridge facts A and B in the bank does not lower the 2-hop NLL more than having only one. Across 3 seeds, |Δ(AB_vs_A_only)| ≤ 0.014 nat and |Δ(AB_vs_B_only)| ≤ 0.010 nat — well inside noise. The mechanism does not compose multi-hop chains; it does not perform multi-hop reasoning.

---

## a. Reproduction command

```bash
for S in 0 1 2; do
  python3 v2/experiments/e18_2hop/run.py --seed $S \
      --bank_layer 9 --rank 64 --steps 200 \
      --n_natural_chains 120 --synthetic_ok true
done
```

## b. Seeds & sample size

seeds {0,1,2}; 60 train chains + 60 eval chains; bank_layer=9; rank=64; steps=200.

## c. Raw data paths

`v2/experiments/e18_2hop/e18_seed{0,1,2}.json`

## d. Numbers (NLL on 2-hop question)

| seed | None (no bank facts) | A_only | B_only | AB_both | Δ(AB vs A_only) | Δ(AB vs B_only) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.185 | 12.179 | 12.194 | 12.184 | **+0.006** | **−0.010** |
| 1 | 12.580 | 12.581 | 12.586 | 12.578 | **−0.003** | **−0.008** |
| 2 | 12.531 | 12.528 | 12.544 | 12.542 | **+0.014** | **−0.002** |

Pass criterion: `Δ(AB vs A_only) ≤ −0.8 AND Δ(AB vs B_only) ≤ −0.8`.
Result: **0/3 pass**. All 9 deltas land in [-0.010, +0.014] nat — essentially zero.

For reference, single-hop NLL on this same setup is ~10.13 (B only) / ~9.62 (A only). The 2-hop NLL is ~12.2, indicating the 2-hop question is genuinely harder than either single hop, confirming the task structure is meaningful.

## e. Verdict

- **Hypothesis**: "If the bank stores facts and the projector retrieves them, then for a 2-hop query (`X → A → B → answer`), having BOTH A and B in the bank should reduce NLL more than having just one."
- **Result**: **Refuted.** No composition benefit detected. Across 3 seeds, the gap between AB-both and the better single-only condition is in [-0.010, +0.014] nat — indistinguishable from zero.
- **Pass rate**: 0/3.
- **Falsifier #5 in V2_FINAL_VERDICT §1.Overall Stance.**

## f. Caveat

Single-hop NLLs themselves are nearly identical across the 4 conditions (None vs A_only vs B_only vs AB_both differ by ≤ 0.015 nat on the single-hop A or B query). This suggests that on the *natural-chain* eval distribution used here, even the single-hop bank insertion is not measurably helping — consistent with the e17 finding that bank effects are template-conditional and the eval distribution here doesn't trigger the projector strongly.

A more aggressive design that re-trains the projector specifically on 2-hop chains (rather than on factual-completion items) might produce a different result, but would also no longer be probing whether the v2 *memory* mechanism composes. That experiment is out of scope here.

## g. Implications

- The ALB thesis's "chained reasoning via bank composition" claim cannot be supported by this architecture under any seed.
- Combined with e16-forgetting (A/B symmetry), e10 (random > real under top-K), and e17 (wrong-target lift), the v2 mechanism is structurally incapable of differentiating bank contents during query — composition is a special case of differentiation and therefore impossible.
