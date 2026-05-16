# E10 — Top-K retrieval mechanism

**Status**: The "retrieval rehabilitation" path is **refuted** by E10.
**Headline**: Replacing the canonical `all-attend` bank read with `top-K cosine` retrieval — the cleanest way to test whether the projector is doing content-based lookup — does NOT recover a memory-like signature. Under `top-K cosine, K=8`, a **random bank beats the real bank by 1.89 nat** (Δ_random=−4.43 vs Δ_real=−2.54). Under `all-attend`, a **random renormalized bank beats the real bank by 1.67 nat** (Δ_random=−5.71 vs Δ_real=−4.05). Random banks consistently give the projector *more* headroom to fit, not less.

---

## a. Reproduction command

```bash
for V in all_attend_real all_attend_random_renorm15 \
         topk_cosine_real_K1 topk_cosine_real_K8 topk_cosine_real_K64 \
         topk_cosine_random_K8 topk_random_indices_K8; do
  python3 v2/experiments/e10_topk_retrieval/run.py --variant $V --seed 0 \
      --bank_layer 9 --rank 64 --steps 200 --n_train 120 --n_eval 80
done
```

## b. Seeds & sample size

seed 0; n_train=120, n_test=80; bank_layer=9; rank=64; steps=200.

## c. Raw data paths

`v2/experiments/e10_topk_retrieval/e10_{variant}_seed0.json`

## d. Numbers

| Variant | delta_real (signed; negative = improvement) |
|---|---:|
| `all_attend_real` (canonical) | **−4.05** |
| `all_attend_random_renorm15` | **−5.71** |
| `topk_cosine_real_K1`  | −0.69 |
| `topk_cosine_real_K8`  | **−2.54** |
| `topk_cosine_real_K64` | −3.21 |
| `topk_cosine_random_K8`  | **−4.43** |
| `topk_random_indices_K8` | −1.49 |

**Key contrasts**:
- `topk_cosine_random_K8` Δ=−4.43 vs `topk_cosine_real_K8` Δ=−2.54 → **random bank beats real bank under cosine top-K by 1.89 nat**. The K=8 nearest-neighbors of the query in a random-Gaussian bank steer the projector to a *lower* eval NLL than the K=8 nearest-neighbors of the query in the real MEMIT bank.
- `all_attend_random_renorm15` Δ=−5.71 vs `all_attend_real` Δ=−4.05 → **random bank also beats real bank under all-attend by 1.67 nat**. Renormalized to match real-bank L2 stats (renorm15), random still wins.
- `topk_cosine_real_K1` Δ=−0.69 collapses to near-baseline. Single-NN retrieval kills the effect, but this is *because* a single bank entry is a thin substrate, not because retrieval was useful.
- `topk_random_indices_K8` Δ=−1.49: random indices (no similarity computation) underperform cosine top-K. The cosine head learns *something*, but what it learns is not content-aligned.

## e. Verdict

- **Hypothesis**: "if the bank stores facts, then top-K *cosine* retrieval (which preferentially selects facts similar to the query) should outperform top-K *random-indices* retrieval and all-attend retrieval"
- **Result**: **Refuted.** Cosine top-K does outperform random-indices top-K (−2.54 vs −1.49), but a random-bank cosine top-K outperforms a real-bank cosine top-K (−4.43 vs −2.54). The cosine head learns to navigate the bank as substrate, not as content. There is no retrieval-of-facts; there is only learned routing through a learnable substrate.
- **Pass rate**: 0/1 (the prediction-divergent contrast `random_K8 vs real_K8` lands on the wrong side by a wide margin).
- **Falsifier #6 in V2_FINAL_VERDICT §1.Overall Stance.**

## f. Caveat

The single best-performing cell of all of e10 is `all_attend_random_renorm15` Δ=−5.71, which is the closest to Phase B2's canonical Δ=−5.83. This means **the best v2 retrieval configuration is one that has no retrieval at all and no real content**. We do not currently understand the renorm15 magic; possibly it's matching internal-statistics regularization of the projector training, similar to BatchNorm running stats.

## g. Implications

- "Top-K cosine retrieval" is not a path forward for the memory thesis on this architecture.
- If a future variant wants to claim content-based behavior, it must engineer an explicit, *learnable* address mechanism that operates orthogonally to the bank-as-substrate dynamics demonstrated here. e16-forgetting (A/B symmetry across 3 seeds) further constrains any such future design: the address head must encode information that survives bank eviction, or the symmetry will reproduce.
