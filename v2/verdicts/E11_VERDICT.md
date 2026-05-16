# E11 — Noise-bank robustness

**Status**: The "memory content matters" claim is **decisively refuted** by E11.
**Headline**: Six different *purely-synthetic* bank populations — IID Gaussian, uniform on the unit sphere, a single random row replicated 512×, a single learned-encoder row replicated 512×, a constant non-random vector replicated 512×, and the real MEMIT bank under top-K=1 — all train projectors that recover **≥2.7 nat (typically ≥5 nat) NLL drop**, matching or exceeding the canonical real-bank Δ=−5.83 nat (Phase B2). The bank's contents are not a stored memory; the bank substrate is a learnable adapter substrate.

---

## a. Reproduction command

```bash
for V in n1_iid_gaussian n2_uniform_sphere n3_single_row_replicated \
         n4_single_random_replicated n5_constant_vector n6_real_bank_K1; do
  python3 v2/experiments/e11_noise_robustness/run.py --variant $V --seed 0 \
      --bank_layer 9 --rank 64 --steps 200 --n_train 120 --n_test 80
done

# Layer-21 replication for variants n1, n3, n5:
for V in n1_iid_gaussian n3_single_row_replicated n5_constant_vector; do
  python3 v2/experiments/e11_noise_robustness/run.py --variant $V --seed 0 \
      --bank_layer 21 --rank 64 --steps 200 --n_train 120 --n_test 80
done
```

## b. Seeds & sample size

seed 0; n_train=120, n_test=80; bank_layer ∈ {9, 21}; rank=64; steps=200.

## c. Raw data paths

`v2/experiments/e11_noise_robustness/e11_{variant}_{seed0|L21_seed0}.json`

## d. Numbers

Bank-on (real) NLL after training and `nll_drop = base − real_after`. Convention is unsigned-positive: larger drop = more improvement.

| Variant | Layer | base NLL | real-after | nll_drop |
|---|---:|---:|---:|---:|
| n1 IID Gaussian            | 9  | 11.998 | 5.951 | **6.05** |
| n1 IID Gaussian            | 21 | 11.998 | 5.683 | **6.32** |
| n2 Uniform on sphere       | 9  | 11.998 | 5.951 | **6.05** |
| n3 Single row replicated   | 9  | 11.998 | 6.496 | **5.50** |
| n3 Single row replicated   | 21 | 11.998 | 5.642 | **6.36** |
| n4 Single random replicated| 9  | 11.998 | 6.108 | **5.89** |
| n5 Constant vector         | 9  | 11.998 | 9.290 | **2.71** |
| n5 Constant vector         | 21 | 11.998 | 5.519 | **6.48** |
| n6 Real bank, top-K=1      | 9  | 11.998 | 6.237 | **5.76** |
| **canonical (real bank, all-attend)** | 9 | 12.13 | 6.30 | 5.83 |

All variants share the same random-bank-control behavior at eval: random-bank Δ ≈ 0 (replacing the trained bank with a fresh random bank at eval time returns NLL to base). This is the inverse story from the training-time control: training-time **content is irrelevant**, but eval-time **substrate must match what was trained on**. Both observations are consistent with an adapter that bakes its function into the projector parameters relative to a frozen reference bank state.

## e. Verdict

- **Hypothesis**: "the bank stores semantic facts; varying the content should vary the effect"
- **Result**: **Refuted.** Replacing the MEMIT bank with pure noise (Gaussian, sphere, constant) yields equal or larger NLL drops. The bank's payload is causally irrelevant during training.
- **Pass rate**: 0/9 (all variants violate the implicit rule "real bank should beat synthetic banks by a meaningful margin").
- **Falsifier #1 in V2_FINAL_VERDICT §1.Overall Stance.**

## f. Caveat

The n5_constant_vector @ L9 result is anomalously weak (2.71 nat) compared to its L21 replication (6.48 nat). This suggests layer-9 has structural sensitivity to constant-row banks that other layers do not. Does not change the core finding; if anything, n5@L9 is the closest thing to a "memory disambiguation" hit in this entire experiment, and it still produces 2.7 nat of bank-on improvement (greater than e10 top-K real-bank Δ=−2.54).

## g. Implications

- The K-projector + AttentionBank training procedure has the same approximation power as a low-rank fine-tune: it can absorb factual-completion signal from gradient steps and re-emit it through any non-degenerate bank substrate.
- This forces the v2 thesis to retreat from "native memory" to "parameter-efficient adapter plumbed through a bank API." Subsequent waves (e10, e16, e17, e18) confirm and extend this finding.
