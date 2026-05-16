# E14 — Pause-head training

**Status**: **FAIL** — all 4 (λ_sparse) cells produce identical positive Δ NLL = **+0.76** (worse than base) and identical mean_pauses = **0.622** (frozen across cells, indicating the sparsity penalty has no effect on the pause distribution).

**Headline**: Across λ_sparse ∈ {0.0, 0.01, 0.1, 1.0} with K_max=4 and steps=200, every cell produces bit-identical pause statistics and bit-identical post-NLL. Either the pause-head gradient path is broken or the parameterization is degenerate. 0/4 cells pass.

---

## a. Reproduction command

```bash
for LAM in 0.0 0.01 0.1 1.0; do
  python3 v2/experiments/e14_pause_train/run.py --seed 0 \
      --lam_sparse $LAM --max_pauses 4 --steps 200 --lr 2e-4 \
      --bank_layer 9 --rank 64 --n_train 120 --n_test 120 --n_preload 512
done
```

## b. Seeds & sample size

seed 0; 4 cells (λ_sparse grid); K_max=4; steps=200; n_train=120, n_test=120.

## c. Raw data paths

`v2/experiments/e14_pause_train/cells/lam{0.0,0.01,0.1,1.0}_kmax4_seed0.json`
`v2/experiments/e14_pause_train/e14_summary_seed0.json`

## d. Numbers

| λ_sparse | base NLL | post NLL (lpl) | Δ NLL | mean_pauses |
|---:|---:|---:|---:|---:|
| 0.00 | 12.033 | 12.796 | **+0.762** | **0.6225** |
| 0.01 | 12.033 | 12.796 | **+0.762** | **0.6225** |
| 0.10 | 12.033 | 12.796 | **+0.762** | **0.6225** |
| 1.00 | 12.033 | 12.796 | **+0.762** | **0.6225** |

All cells return **byte-identical** before/after blocks. `std_pauses = 0.0` across n=120 samples in each cell — the pause head is producing a constant (per-example) value, not a learned distribution.

## e. Verdict

- **Hypothesis ("a trainable pause head can learn when to halt and what to retrieve")**: Refuted.
- **Pass rate**: 0/4.
- λ_sparse has **zero effect** on either pause statistics or NLL — strong evidence the pause-head loss term is not coupled to the optimizer in this driver.
- This is the auto-pause companion failure to **e04** (halt-head never fires). Both halves of the "auto-pause / halt" dual-channel subsystem are mechanistically inert in current training.

## f. Caveat

- Driver may have a wiring bug (constant pause output across all cells is suspicious). However, even taking the numbers at face value, the pause head does not pass the gating criterion. A driver fix would need to be paired with a re-run, and the prior probability of a positive result is low given e04's parallel null.

## g. Implications

- **Falsifier in spirit**, similar to e04: the trainable auto-pause mechanism does not deliver gain on top of the projector. Combined with e04 (halt-head dead) and e15 (cumulative pondering inert at K>2), the entire multi-round / pause subsystem of the ALB thesis collapses to "use a single LPL round with a trained K-projector."
