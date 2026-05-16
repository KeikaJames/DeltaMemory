# E07 — Per-layer K-projector

**Status**: **PASS** — multi-layer projectors meaningfully beat single-layer.

**Headline**: Single layer [9] gives Δ = **−2.52**; triple [3,9,21] gives Δ = **−5.35** (+112% gain); six-layer [3,9,15,21,27,33] gives Δ = **−4.79** (slightly worse than triple, suggesting saturation). Multi-layer ≫ single-layer, but more layers ≠ more better.

---

## a. Reproduction command

```bash
python3 v2/experiments/e07_perlayer_kproj/run.py --seed 0 \
    --layer_sets "[9],[3,9,21],[3,9,15,21,27,33]" \
    --rank 64 --steps 200 --n_train 120 --n_eval 120 --n_preload 512
```

## b. Seeds & sample size

seed 0; 3 layer configurations; n_train=120, n_eval=120, n_preload=512; rank=64 per projector; steps=200.

## c. Raw data paths

`v2/experiments/e07_perlayer_kproj/e07_seed0.json`

## d. Numbers

| layers | n_params total | Δ_real | Δ_zero | Δ_off |
|---|---:|---:|---:|---:|
| [9] | 330,241 | −2.518 | +0.018 | 0.000 |
| **[3, 9, 21]** | 990,723 | **−5.352** | +0.022 | 0.000 |
| [3, 9, 15, 21, 27, 33] | 1,981,446 | −4.791 | +0.016 | 0.000 |

Pass criterion: `triple − single ≤ −0.5` → −2.83 ≤ −0.5 ✅.

## e. Verdict

- **Hypothesis ("multiple layers add gain")**: Confirmed at triple ([3,9,21]) — 2.83 nat improvement over single. ✅
- **Saturation observed**: six-layer underperforms triple by 0.56 nat despite 2× params. The marginal layer cost (params + training noise) exceeds the marginal layer benefit beyond ~3 well-spaced layers.
- **Pass rate**: 1/1 on the stated rule.

## f. Caveat

- Single seed; rank held constant across configs (parameter count grows with number of layers — a rank-matched control would isolate "more layers" from "more capacity").
- Layer choice {3, 9, 21} is heuristic; no per-layer importance scan.

## g. Implications

- The "multi-layer projection" is the single largest absolute improvement over single-layer projection in the v2 program. If the projector is fundamentally an adapter (per e16-forgetting), then this finding amounts to: a 3-layer LoRA-like fine-tune beats a 1-layer one, with diminishing returns past 3 layers.
- Useful starting config for future runs: triple-layer [3,9,21] at rank ≥ 64.
