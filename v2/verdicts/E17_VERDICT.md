# E17 — Negation robustness (content-blindness probe)

**Status**: The "memory content matters" claim is **decisively refuted** by E17.
**Headline**: On standard prompts with **random (incorrect) targets**, the trained bank+projector still lowers NLL by **−2.66 ± 0.14 nat across 3 seeds**. The mechanism cannot tell a correct continuation from a wrong continuation; it pulls every continuation down. This is the cleanest direct probe of content-blindness in the v2 suite.

---

## a. Reproduction command

```bash
for S in 0 1 2; do
  python3 v2/experiments/e17_negation_robustness/run.py --seed $S \
      --bank_layer 9 --rank 64 --steps 200 --n_train 120 --n_test 80
done
```

## b. Seeds & sample size

seeds {0,1,2}; n_train=120, n_test=80; bank_layer=9; rank=64; steps=200.

## c. Raw data paths

`v2/experiments/e17_negation_robustness/e17_seed{0,1,2}.json`

## d. Numbers (signed; negative = improvement)

| condition | seed 0 | seed 1 | seed 2 | mean | semantics |
|---|---:|---:|---:|---:|---|
| (a) standard + standard target | −4.94 | −4.08 | −5.31 | **−4.78** | canonical task |
| (b) standard + **random** target | −2.77 | −2.50 | −2.71 | **−2.66 ± 0.14** | **same template, wrong target — should NOT improve under retrieval** |
| (c) negated + random target | −0.66 | −0.91 | −0.43 | −0.67 | sanity check (negation flips meaning) |
| (d) negated + true target | −0.79 | −0.95 | −0.72 | −0.82 | sanity check (negated correct) |

**Key contrast: (a) vs (b)**. Condition (b) uses the standard MEMIT template (e.g., "Paris is the capital of __") but pairs it with a random wrong target ("Tokyo", "banana", etc.) for the NLL computation. A content-based memory must NOT lower the NLL of a wrong continuation; doing so means the projector is shifting the *entire* output distribution toward some learned attractor, irrespective of whether that attractor matches the true target.

Across 3 seeds, the wrong-target NLL falls by **2.66 nat on average**. The projector lifts the wrong-target continuation by more than half as much as the right-target continuation (2.66 / 4.78 ≈ 56 %). Mean over 3 seeds: −2.66, std 0.14.

Sanity checks (c, d) confirm the negated template breaks the mechanism (Δ shrinks to −0.67 / −0.82), ruling out a trivial pure-vocab-bias explanation: the projector is template-sensitive, just not target-sensitive.

## e. Verdict

- **Hypothesis**: "the bank stores facts; querying for a fact should preferentially lift its associated target, not arbitrary wrong targets"
- **Result**: **Refuted.** A wrong target on the same standard template gets ~56 % of the lift the correct target receives. The projector is template-aware but content-blind.
- **Pass rate**: 0/3 (all 3 seeds show large wrong-target lift; replicates tightly with std=0.14).
- **Falsifier #4 in V2_FINAL_VERDICT §1.Overall Stance.**

## f. Caveat

Cells (c) and (d) — negated templates — do show much smaller Δ (|Δ| ≤ 0.95). This means the projector is conditioning on *template* (it broke negated prompts) but not on *target*. A version of this experiment that varies prompt syntactic structure more finely would help separate "template-aware adapter" from "completion-style attractor."

## g. Implications

- The strongest single signature of content-based retrieval would be "right-target gets large lift, wrong-target gets ~0 lift." E17 shows the opposite of that signature with very tight replication (3-seed std 0.14 on a Δ of 2.66).
- Combined with e16-forgetting (A/B symmetry) and e11 (any non-zero bank works), this places the v2 mechanism firmly in adapter territory.
