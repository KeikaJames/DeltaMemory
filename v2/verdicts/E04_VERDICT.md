# E04 — ACT halt + K_max sweep

**Status**: The "adaptive computation via halt-head" claim is **falsified** by E04.
**Headline**: Across all 4 (λ_ponder, K_max) cells in a 2×2 pilot, the trainable halt head **never fires** (mean halts = 0.00 in every cell), yet the experiment "passes" its Δ-NLL criterion with Δ = −4.57 to −5.25. The gain is delivered entirely by the K-projector under default K=2 (i.e., the canonical e01 path). The ACT mechanism is **mechanistically inert** in current training.

---

## a. Reproduction command

```bash
for LAM in 0.01 0.10; do
  for KMAX in 4 16; do
    python3 v2/experiments/e04_act_halt/run.py --seed 0 \
        --lam_ponder $LAM --k_max $KMAX \
        --bank_layer 9 --rank 64 --steps 200 \
        --n_train 120 --n_test 80
  done
done
```

## b. Seeds & sample size

seed 0; 2×2 grid (λ_ponder ∈ {0.01, 0.10}, K_max ∈ {4, 16}); n_train=120, n_test=80; bank_layer=9; rank=64; steps=200.

## c. Raw data paths

- `v2/experiments/e04_act_halt/cells/lam{0.010,0.100}_kmax{4,16}_seed0.json`
- `v2/experiments/e04_act_halt/e04_summary_seed0.json`

## d. Numbers

| λ_ponder | K_max | base NLL | post NLL | Δ NLL | mean post-halts |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 4  | 12.033 | 6.814 | **−5.220** | **0.000** |
| 0.01 | 16 | 12.033 | 6.993 | **−5.040** | **0.000** |
| 0.10 | 4  | 12.033 | 7.458 | **−4.575** | **0.000** |
| 0.10 | 16 | 12.033 | 6.784 | **−5.250** | **0.000** |

(`post-halts` is the average number of halt events per eval example across the full K_max budget. Range observed: 0.00 in all 4 cells.)

## e. Verdict

- **Hypothesis**: "A learned halt head can perform adaptive computation: easier examples halt early, harder examples use more rounds."
- **Result**: **Refuted.** Halt head never fires (mean halts = 0.00 in every cell), regardless of λ_ponder or K_max. The NLL drop comes entirely from the projector + default K=2 inference, which is just the canonical e01 path under a different driver. Adaptive computation is not happening.
- **Pass rate**: 0/4 on the structural-mechanism test (halt head must produce non-trivial halt distribution).
- **Falsifier #9 in V2_FINAL_VERDICT §1.Overall Stance.**

## f. Caveat

This is a **pilot** — only 200 training steps, single seed, 2×2 grid. A more aggressive training (5000+ steps), entropy-bonus annealing, or different λ schedule might activate the halt head. **However**: even if a longer schedule could push halts > 0, the projector already saturates its NLL drop at K=2 (e15 shows K=3 and K=4 add zero gain over K=2 in cumulative mode). So even if the halt head learned to fire, there is no quality headroom for it to access by allocating more rounds — making the ACT machinery structurally redundant on this architecture.

## g. Implications

- The "dual-channel HNM" thesis's *auto-pause* channel — meant to be implemented through trainable halt heads — is mechanistically inert with the current projector + training recipe.
- Any future architectural revision needs to either (a) make K=2 produce *insufficient* improvement so additional rounds matter, or (b) provide the halt head with an orthogonal training signal (e.g., explicit difficulty labels) so it has something to learn beyond what the projector already absorbs.
