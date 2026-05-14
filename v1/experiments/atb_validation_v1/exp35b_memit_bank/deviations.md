# Exp35b Deviations Log

Append-only record of any deviation from `preregister.json` discovered during
execution. Each entry is committed at the moment of discovery, before any
downstream metric is computed.

---

## D2 subject-collision target (2026-XX-XX, corpus build step 00)

**Preregistered**: `subject_collision_target_frac: 0.30`

**Discovered**: The full `azhx/counterfact` dataset has only 901 multi-fact
subjects out of 18,466 unique subjects (4.9%). Even sampling 10000/19728 to
maximise multi-fact subjects, the upper bound is ~11% subject-collision frac.
At our actual sampled 10k, the realised collision frac is **6.78%**
(`corpus_meta.json`).

**Decision**: The D2 audit will run on the natural collision subset (~680
test facts whose subject collides with another bank fact). The N=10k corpus
construction is honest. We do NOT synthesise additional collisions because
that would itself be a cheat (artificial collisions != naturally collided
subjects).

**Impact on verdict**: D2 statistical power is reduced. If the natural
collision subset is too small for the pre-registered top-1 ≥ 25% threshold
to be meaningfully measurable (N < 300), we will report D2 as INCONCLUSIVE
rather than PASS/FAIL.

---
