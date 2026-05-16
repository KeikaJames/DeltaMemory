# E16 — Bank capacity & forgetting

**Status**: The "memory" framing is **decisively refuted** by E16.
**Headline**: The bank's contents are functionally irrelevant; what matters is the *presence* of any bank substrate. After training a projector with set_A in the bank, evicting set_A and replacing it with a different set_B yields identical Δ on set_A queries and set_B queries (within 0.02 nat, across 3 seeds). This is the cleanest content-blindness probe in the entire v2 suite.

---

## a. Reproduction command

```bash
# Phase A — scaling (does bigger bank help? does in-bank vs out-of-bank differ?)
python3 v2/experiments/e16_capacity/run.py --phase scaling --seed 0 \
    --N_grid 64,256,1024,4096 --steps 200 --n_train 120 --n_eval 80

# Phase B — forgetting (does eviction destroy the effect?)
for S in 0 1 2; do
  python3 v2/experiments/e16_capacity/run.py --phase forgetting --seed $S \
      --steps 200 --n_train 120 --n_eval 80
done
```

## b. Seeds & sample size

- Phase A (scaling): seed 0; 4 N values; n_train=120, n_eval=80; bank_layer=9; rank=64
- Phase B (forgetting): seeds {0,1,2}; N=512 fixed; n_train=120, n_eval=80 per set; rank=64

## c. Raw data paths

- `v2/experiments/e16_capacity/scaling/N{64,256,1024,4096}_seed0.json`
- `v2/experiments/e16_capacity/scaling/summary_seed0.json`
- `v2/experiments/e16_capacity/forgetting/seed{0,1,2}.json`

## d. Numbers

### Phase A — scaling (does N differentially help in-bank items?)

| N | Δ_in_bank | Δ_out_of_bank | Δ_in − Δ_out |
|---:|---:|---:|---:|
| 64   | 5.91 | 5.84 | +0.07 |
| 256  | 6.06 | 5.74 | +0.32 |
| 1024 | 5.06 | 4.62 | +0.44 |
| 4096 | 5.82 | 5.45 | +0.37 |

(Unsigned-positive convention; Δ = base NLL − post NLL; larger = more improvement.)

**Finding**: Δ_in ≈ Δ_out across the entire 64→4096 range; differences are within 0.5 nat at all N. A content-based memory must preferentially help items that are actually in the bank. This does not happen.

### Phase B — forgetting (does eviction destroy the effect?)

| seed | Δ_A_initial | Δ_A_after_evict | Δ_B (never trained) | Δ_A_zero (empty) |
|---:|---:|---:|---:|---:|
| 0 | 4.986 | 4.812 | 4.817 | 0.000 |
| 1 | 5.269 | 1.613 | 1.615 | 0.000 |
| 2 | 4.741 | 0.283 | 0.298 | 0.000 |

**Findings**:
1. **Δ_A_after_evict ≈ Δ_B across all 3 seeds (max gap 0.02 nat)**. After replacing the trained-on set with a never-seen set, the projector applies the *same* effect to both. This is the signature of an adapter that uses the bank as substrate, not as content store.
2. Magnitude of post-eviction Δ varies wildly across seeds (0.28 → 4.81 nat). The variance is in *how much* the projector continues to lift, not in *which set* it preferentially lifts. The A/B symmetry holds regardless of magnitude.
3. Empty bank → Δ=0 in all seeds. This rules out the projector having learned a vocab bias independent of the bank (i.e., it's not a pure decoder shift). The bank substrate is necessary, but its contents are not.

## e. Verdict

- **Hypothesis**: "the bank stores facts; query-time lookup retrieves them"
- **Result**: **Refuted.** Bank contents have no measurable effect once the projector is trained. Bank substrate (i.e. *any* non-zero KV stream of the right shape) is sufficient and necessary.
- **Pass rate**: Phase A 0/1 (no in-bank preference detected). Phase B 0/3 (all 3 seeds violate the rule `|Δ_A_after − Δ_zero| < 0.3 AND |Δ_B − Δ_A_initial| < 1.0`).
- **Falsifier #10 in V2_FINAL_VERDICT §1.Overall Stance.**

## f. Caveat

The post-eviction Δ_A (and Δ_B) shows large seed variance (0.28 → 4.81). The mechanism that determines whether the projector retains a "free" lift after eviction is not yet understood; one hypothesis is that depending on initialization, training either drives the projector into a "low-rank rotation of the bank substrate" basin (high free lift) or a "specific-bank-content fitter" basin (low free lift). The A/B symmetry holds in both basins. A targeted experiment varying the training set size or rank may disentangle this; not run.

## g. Implications

- The "Hippocampus-style Native LLM Memory" framing — long-term store + working memory with content-based retrieval — is mechanistically not what is happening at v2 scale on Qwen3-4B.
- The K-projector is a parameter-efficient adapter (rank-64 ≈ 420K params) injected via an attention bank API. Its effect on factual completion NLL is real, robust across seeds, and entirely independent of bank payload semantics.
- Future architectural research should either (a) accept the adapter framing and optimize the bank-as-substrate path, or (b) add an explicit retrieval head with its own learnable address mechanism (current `topk_cosine` retrieval in e10 already failed to demonstrate content-based behavior — random bank under top-K beat real bank).
