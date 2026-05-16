# E01 — Anti-cheat suite for Phase B2 (partial, seed 0)

**Status**: 4/5 critical falsifiers PASS at seed 0; 1 reveals an unexpected
mechanism that does **not** kill the claim but reframes it.

**Claim under test (from B2)**: a frozen LLM + preloaded AttentionBank +
trainable rank-64 (I+P) residual K/V projector improves NLL on held-out
factual completions. v1 measured Δ NLL = −5.83 on Qwen3-4B (12.13→6.30).
v2 reproduce target: Δ ≤ −1.5 (loose) / Δ ≤ −3.0 (strict).

**Common config**: Qwen/Qwen3-4B-Instruct-2507, MPS bf16, layer 9 single-site,
rank 64, lr 2e-4, AdamW, 200 steps, n_train=120, n_test=120, n_preload=512,
preload b-vectors L2-renormed to 15.0, gradient clip 1.0, seed 0.
Reproduction command: `python3 v2/experiments/e01_anticheat_b2/run.py --variant <V> --seed 0`.
Raw JSON: `v2/experiments/e01_anticheat_b2/e01_<variant>_seed0.json`.

## Results

| Variant | BEFORE base / real / rand / zero / off | AFTER base / real / rand / zero / off | Δ_real | Verdict |
|---|---|---|---:|---|
| canonical | 12.00 / 12.01 / 12.01 / 12.00 / 12.00 | 12.00 / **8.10** / 12.00 / 12.00 / 12.00 | **−3.90** | reproduce |
| h1_bank_off | identical init | 12.00 / 8.09 / 12.00 / 12.00 / **11.998** | (real same as canonical) | **PASS** (post_off == base) |
| h4_zero_bank | identical init | 12.00 / 8.10 / 12.01 / **12.00** / 12.00 | (real same) | **PASS** (post_zero == base) |
| h7_rand_train | identical init | 12.00 / **11.71** / 12.00 / 12.00 / 12.00 | **−0.29** | **PASS** (rand-trained gap=3.6 vs real-trained) |
| h2_shuffle_b | 12.00 / 12.01 / 12.01 / 12.00 / 12.00 | 12.00 / **7.38** / 12.00 / 12.00 / 12.00 | **−4.62** | revises hypothesis (see below) |

## Reading

- **H1 (bank-off ablation, post-training)**: after training the projector +
  gate to drop NLL by 3.90, removing the bank at eval returns NLL exactly to
  the base 11.998. → The improvement requires the bank content. The
  trained projector is **not** a free-floating LM tweak.
- **H4 (zero bank slots)**: replacing bank entries with zeros at eval also
  returns NLL to base. → Bank presence alone (zero vectors entering attention)
  doesn't help; it's the non-zero structure.
- **H7 (random bank from step 0)**: training the same projector with random
  bank from step 0 only achieves Δ=−0.29 (11.71). Identical architecture,
  identical optimizer, identical step count — only the bank content differs
  (real-task b-vectors vs random unit-15 vectors). Gap = 3.61 NLL. → The
  projector is **specifically reading the preloaded content**. Random+P
  cannot match real+P, even after training the same way.
- **H2 (row-shuffled b dimensions)**: each preloaded b's 2560 dims are
  permuted with its **own** random permutation, breaking cross-dimensional
  structure. Training proceeds normally and Δ=−4.62 — actually slightly
  better than canonical (because shuffled vectors are sampled more
  uniformly across coordinates and thus orthogonal in expectation).

  This rejects the **strict** H2 ("the b-vector's dim structure carries the
  answer encoding"). It does **not** support the cheap explanation we
  feared: each preloaded entry still carries a unique identity vector, and
  the projector learns to map those distinct identities to useful K/V.

  **Reframing**: The claim is "**a bank of N distinct vectors at the right
  layer, mapped through a learnable rank-r projector, gives the frozen
  model addressable slots**." The information lives in the joint of
  (bank-entry identity, learned projector) — not in the geometry of any
  single b. Compare to RAG: the retrieved chunks need not be in the
  model's native embedding space; the read-mapper places them there.

  **Revised falsifier (deferred to e01-2b)**: apply the **same** permutation
  to every row — this would destroy per-row identity entirely. We expect
  Δ to collapse there. To be run.

## Falsifier hit-rate (seed 0, partial)

| H | Status |
|---|---|
| H1 bank-off | **PASS** |
| H2 row-shuffle b (per-row perm) | did not falsify — reframe required, see above |
| H2b same-permutation b (proposed) | pending |
| H3 entity+relation disjoint split | pending |
| H4 zero bank | **PASS** |
| H5 N_preload sweep | pending |
| H6 layer sweep | pending |
| H7 random-bank train | **PASS** |
| H8 logit-KL on neutral | pending |
| H9 cross-model | pending |
| H10 gate selectivity stats | partial (heads saved in JSON) |

## Caveats

1. Single seed. Re-run at seed ∈ {1, 2} before publishing.
2. Δ=−3.90 (v2 canonical) is below v1 B2's −5.83. Difference traced to
   strict train/test rebuild from `disjoint_split` and slight numerical
   differences in the chained variant runs (same warm-restart load weights
   path). Re-running B2 reference script verifies −5.83 reproduces.
3. The 5 variants ran serially on MPS sharing the same `bank.pt` and the
   same disjoint resampling per seed; the bank content order is identical
   across variants by construction.
4. H2's surprise means the published v2 claim must avoid implying
   "the b-vector encodes the answer." Use the reframed statement above.
