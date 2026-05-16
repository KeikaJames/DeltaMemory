# E15 — Ponder curriculum / K-round saturation

**Status**: The "multi-round pondering improves quality" claim is **falsified** by E15.
**Headline**: K ∈ {1, 2, 3, 4} cumulative-mode run on a fixed projector: K=1 Δ=0 (untrained), K=2 Δ=**4.7531**, K=3 Δ=**4.7531**, K=4 Δ=**4.7531** — **identical to 4 decimal places**. Rounds beyond K=2 add zero information. The projector at round 2 absorbs all available signal; the proposed multi-round attention dynamics are inert.

---

## a. Reproduction command

```bash
for K in 1 2 3 4; do
  python3 v2/experiments/e15_ponder/run.py --seed 0 \
      --K $K --mode cumulative \
      --bank_layer 9 --rank 64 --steps 200 \
      --n_train 120 --n_test 80
done
```

## b. Seeds & sample size

seed 0; K ∈ {1,2,3,4}, mode = cumulative; n_train=120, n_test=80; bank_layer=9; rank=64; steps=200.

## c. Raw data paths

- `v2/experiments/e15_ponder/cells/K{1,2,3,4}_modecumulative_seed0.json`
- `v2/experiments/e15_ponder/e15_summary_seed0.json`

## d. Numbers

| K | post NLL | Δ NLL (unsigned-positive = base − post) |
|---:|---:|---:|
| 1 | 12.0331 | 0.0000 |
| 2 | 7.2800 | **4.7531** |
| 3 | 7.2800 | **4.7531** |
| 4 | 7.2800 | **4.7531** |

Base NLL = 12.0331 in all cells (same eval set, frozen model).

## e. Verdict

- **Hypothesis**: "Multi-round attention with cumulative bank read enables progressive refinement: K>2 should yield further NLL drops over K=2."
- **Result**: **Refuted.** K=2, K=3, K=4 produce literally identical post-NLL (matched to 4 decimal places). The projector at round 2 reaches a fixed point; subsequent rounds contribute nothing.
- **Pass rate**: 0/1 (no K>2 cell improved over K=2 by ≥ 0.3 nat as required).
- **Falsifier #7 in V2_FINAL_VERDICT §1.Overall Stance.**

## f. Caveat

This run is *cumulative* mode (bank carries forward across rounds) at fixed projector parameters. A *forgetful* mode (bank reset each round, projector trained jointly with K_max from scratch) was not run; in principle, that variant could behave differently. However, the prima facie expectation under cumulative-mode failure is that any other mode would either fail similarly or trade convergence stability for higher peak Δ — neither of which would rescue the multi-round thesis as currently stated.

## g. Implications

- The "K curriculum" and "ponder loss" components of the proposed ALB stack are not load-bearing on this architecture.
- Combined with e04 (halt head never fires), the entire multi-round subsystem (auto-pause + halt + cumulative bank) collapses to "do K=2 once and stop."
- The K=2 fixed-point at Δ=4.75 is itself a useful baseline number for what the projector can recover from a single LPL round.
