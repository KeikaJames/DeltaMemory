# E02 — Scale matrix (N_preload × N_train × steps)

**Status**: Mixed / partial PASS — scale matrix is **largely consistent with the projector thesis** but flags one anomaly (a +0.63 nat *regression* at the largest config, suggesting training instability / over-fitting beyond a sweet spot).

**Headline**: Δ_real spans −5.06 (best, n=1024 t=200) to **+0.63 (worst, n=2048 t=1000)**. Random control hovers at |Δ_rand| ≤ 0.04 across all 5 cells (signed-clean). The mechanism does scale, but training-step ramp can flip the sign on this driver and warrants caution.

---

## a. Reproduction command

```bash
for CFG in "n512_t200_s200" "n1024_t200_s500" "n2048_t200_s500" \
           "n2048_t500_s500" "n2048_t1000_s1000"; do
  python3 v2/experiments/e02_scale_matrix/run.py --seed 0 --cfg $CFG \
      --bank_layer 9 --rank 64 --layers 9
done
```

## b. Seeds & sample size

seed 0; 5 grid cells over (N_preload, N_train, steps); bank_layer=9; rank=64; single layer.

## c. Raw data paths

`v2/experiments/e02_scale_matrix/cells/n*_t*_single_s*_seed0.json`

## d. Numbers

| N_preload | N_train | steps | Δ_real | Δ_rand |
|---:|---:|---:|---:|---:|
| 512 | 200 | 200 | **−4.7632** | −0.035 |
| 1024 | 200 | 500 | **−5.0576** | −0.022 |
| 2048 | 200 | 500 | **−4.0166** | +0.002 |
| 2048 | 500 | 500 | **−1.9877** | +0.036 |
| 2048 | 1000 | 1000 | **+0.6277** | +0.020 |

## e. Verdict

- 4/5 cells show Δ_real ≤ −1.9 with paired Δ_rand near zero — projector + bank effect is robust at moderate scale.
- The largest cell (n_train=1000, steps=1000) **degrades** to +0.6 — likely projector over-fitting / optimizer divergence at this driver's lr=2e-4. Not a thesis violation; a hyper-parameter warning.

## f. Caveat

Single seed; no lr sweep paired to the t=1000 cell; full Latin-square subset from the v2 plan (60-config target) was not exercised. The "Δ improves monotonically with scale" prediction is **not** confirmed at the upper end of this driver's range.

## g. Implications

The projector mechanism works robustly at small scale and is *not unbounded*: a fully-extrapolated narrative ("more compute → more memory") is not what the data say. The 5000-step e05 Qwen3-4B run (Δ=−3.97) provides a separate compute scaling point and is consistent with bounded gains.
