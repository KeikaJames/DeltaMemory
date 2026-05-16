# E19 — Seed replication (stability)

**Status**: **PASS** — the projector effect is seed-stable across 5 seeds at two layers.

**Headline**: At layer 21, mean Δ_real = **−6.77 nat**, std = 0.247, |cv| = **0.037**. At layer 9, mean Δ_real = **−4.87 nat**, std = 0.327, |cv| = **0.067**. Both far below the 0.15 cv threshold. The effect size is real and reproducible; it is not seed-noise.

---

## a. Reproduction command

```bash
for L in 9 21; do
  for S in 0 1 2 3 4; do
    python3 v2/experiments/e19_seed_replication/run.py --seed $S --layer $L \
        --rank 64 --steps 200 --n_train 120 --n_test 120 --n_preload 512
  done
done
```

## b. Seeds & sample size

10 runs total: seeds {0, 1, 2, 3, 4} × layers {9, 21}; n_train=120, n_test=120, n_preload=512; rank=64; steps=200.

## c. Raw data paths

`v2/experiments/e19_seed_replication/cells/L{9,21}_s{0..4}.json`

## d. Numbers

**Layer 21**

| seed | Δ_real |
|---:|---:|
| 0 | −7.207 |
| 1 | −6.646 |
| 2 | −6.774 |
| 3 | −6.611 |
| 4 | −6.588 |
| **mean** | **−6.765** |
| std | 0.247 |
| **|cv|** | **0.037** |

**Layer 9**

| seed | Δ_real |
|---:|---:|
| 0 | −4.795 |
| 1 | −5.165 |
| 2 | −4.781 |
| 3 | −4.397 |
| 4 | −5.208 |
| **mean** | **−4.869** |
| std | 0.327 |
| **|cv|** | **0.067** |

## e. Verdict

- **Hypothesis ("Δ is reproducible across seeds; std/|mean| ≤ 0.15")**: Confirmed at both layers, with margin.
- **Pass rate**: 2/2.
- Layer 21 ≈ 39% stronger than layer 9 — a robust layer-choice signal worth carrying into any follow-up.

## f. Caveat

- Same model, same data subset, single rank — seed variance only.
- Does not say anything about *what* the effect is (the adapter-vs-memory question is settled by e10/e11/e16/e17). E19 only certifies that whatever-it-is is not a seed artifact.

## g. Implications

- E19 is the **statistical-credibility floor** under the entire v2 numerical record: ten independent runs converge to mean Δ ∈ {−4.87, −6.77} with cv ≤ 0.07.
- For any future result that disagrees with the adapter interpretation, the bar is now: it has to clear the seed-noise band documented here (≈ ±0.5 nat at L9, ±0.3 nat at L21).
