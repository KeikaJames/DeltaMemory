# E01 — Anti-cheat suite for Phase B2 (partial, seed 0)

**Status**: B2's "memory matters" claim is **substantially falsified** by
e11/wave3. Memory **substrate** matters (off → no effect across every
variant), but memory **content** matters only weakly: trained projector
over **random Gaussian** bank gives Δ=−6.05 (better than real bank's
−3.90); over a **single replicated row** gives Δ=−5.50; over a **constant
vector** gives Δ=−2.83. The rank-64 trainable projector over **any**
non-empty bank carries most of the signal.

**Wave-3 e11 update (seed 0)**:

| Variant | Bank construction | distinct | Δ_real | "should not reduce >2" verdict |
|---|---|---:|---:|---|
| canonical real bank | preloaded b-vectors | 18.7 | **−3.90** | (reference) |
| n1 iid Gaussian | random ~N(0,1) renormed L2=15 | 21.21 | **−6.05** | FAIL (noise helps a lot) |
| n3 single row replicated | one real row × 512 | 0.00 | **−5.50** | FAIL |
| n5 constant vector | same constant × 512 | 0.00 | **−2.83** | FAIL |
| n7 K=0 pure projector | preload=0, train w/ empty bank | n/a | crashed | training requires non-empty bank for gradient |

But **every variant's `off=base=11.998`** — bank emptied at eval → NLL
returns to base. So memory presence (substrate) is required; memory
content is a weak modulator.

**Wave-2 update**: H3 (subj+rel disjoint split) PASS with Δ=−2.69.
**H2c (all bank rows = mean of real bank) UNEXPECTEDLY Δ=−4.81** — i.e.
stronger than canonical (−3.90). This means: row-distinctness is not
required for the projector to extract its training gain. The mechanism is
either (a) the projector exploits "extra K/V slots" as a generic
attention-smoother, or (b) the training signal alone — not bank content —
is what carries the NLL drop. e11 wave3 confirms: it is mostly (a).

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
| h2c_collapsed_bank | 12.00 / 12.03 / 12.00 / 12.00 / 12.00 | 12.00 / **7.19** / 12.01 / 12.00 / 12.00 | **−4.81** | ⚠️ **content-blind warning** |
| h3_disjoint_split | 12.72 / 12.80 / 12.76 / 12.74 / 12.72 | 12.72 / **10.03** / 12.75 / 12.74 / 12.72 | **−2.69** | **PASS** (subj+rel OOD) |

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

- **H2c (collapsed bank, all rows = mean of real bank)**: Δ=−4.81.
  Even with ZERO row-distinctness — every bank slot is the identical mean
  vector — the projector still extracts a 4.81 NLL drop. This **falsifies
  the "addressable slots" reframe of H2**. The signal is not from N
  distinct addresses; it is either from training-time gradient signal that
  doesn't require bank content, or from the projector learning to use the
  extra K/V positions as a generic attention regularizer. Pinned for e11
  (n7 K=0 pure-projector test) to discriminate.

- **H3 (subject+relation disjoint train/test split)**: Δ_real=−2.69 with
  controls flat (rand/zero/off all within 0.05 of base). Even under a hard
  OOD split where the test set shares neither subject nor relation with
  any training item, the projector + bank still generalizes. **PASS**.
  This says: the gain is *not* from per-(subject,relation) routing memorized
  during training — it's a transferable read mechanism.

## Falsifier hit-rate (seed 0, e01 + e11 wave3)

| H | Status |
|---|---|
| H1 bank-off (eval-time empty) | **PASS** — required for signal |
| H2 row-shuffle b (per-row perm) | did not falsify — reframe required |
| H2b same-permutation b | pending |
| H2c collapsed bank (all rows = mean) | ⚠️ **did not falsify** — Δ=−4.81 |
| H3 entity+relation disjoint split | **PASS** Δ=−2.69 |
| H4 zero bank | **PASS** |
| H5 N_preload sweep | pending |
| H6 layer sweep | **PARTIAL** L3=−1.58, L9=−3.90, L21=**−6.29**, L33=−3.97 — deeper layers stronger |
| H7 random-bank train | **PASS** Δ=−0.29 (note: this used random renormed to L2=1, vs e11/n1 which renormed to L2=15; the norm matters) |
| H8 logit-KL on neutral | pending |
| H9 cross-model | pending (e05 queued) |
| H10 gate selectivity stats | partial (heads saved in JSON) |
| **e11/n1 random N(0,1) L2=15** | **FAIL** Δ=−6.05 (random + correct norm > real) |
| **e11/n3 single row replicated** | **FAIL** Δ=−5.50 |
| **e11/n5 constant vector** | **FAIL** Δ=−2.83 |
| e11/n7 K=0 pure projector | needs redesign (cannot train P on empty bank) |

## Revised interpretation (post wave3)

The B2 mechanism is best described as: **a rank-64 trainable K/V projector,
applied to a non-empty layer-9 KV slot bank, learns to use those extra
attention positions as a generic adaptation surface.** The bank content
provides the substrate dimension (without it, no parameters to multiply);
but the *information* in that content is not what is being read out.

This is more consistent with a "free K/V slots = extra capacity" story
than with a "memory of facts" story. v2's hippocampal framing is **not
supported** by these falsifiers; the original v1 B2 result is a real
NLL improvement but does not demonstrate fact recall via bank readout.

**Next decisive tests** to discriminate "free-capacity adaptation" vs
"true content read":
- e11/n2 (uniform sphere, different distribution shape)
- e11/n4 (single random vector replicated)
- e11/n6 (real bank with K=1 slot only — does shrinking bank to 1 kill it?)
- e06 relation-disjoint OOD — if "free capacity", should still help on
  totally unrelated relations; if "content read", should die.
- e13 multi-task — if "free capacity", helps everywhere; if "content read",
  helps only on tasks related to bank content.

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
