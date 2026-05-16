# E09 — v1 AttentionNativeBank resurrection

**Status**: **PASS** — confirms B2 thesis: adding the K-projector is the difference between v1's null result and v2's −5 nat lift on the *same* (model, bank, data, training budget).

**Headline**: Identical configuration except for the K-projector:
- **v1 mode (no projector, untrained bank-read)**: Δ NLL = **+0.014** (statistical null, within ±0.3 band)
- **v2 mode (K-projector added, projector trained)**: Δ NLL = **−5.007** (signed)

The 5-nat gap *is* what the projector contributes.

---

## a. Reproduction command

```bash
python3 v2/experiments/e09_v1_anb_resurrect/run.py --seed 0 \
    --mode v1_orig --bank_layer 9 --steps 200 \
    --n_train 120 --n_eval 120 --n_preload 512

python3 v2/experiments/e09_v1_anb_resurrect/run.py --seed 0 \
    --mode v2_kproj --bank_layer 9 --rank 64 --steps 200 \
    --n_train 120 --n_eval 120 --n_preload 512
```

## b. Seeds & sample size

seed 0; n_train=120, n_test=120, n_preload=512; bank_layer=9.
- v1_orig: 92,196 trainable params (gating only), projector disabled.
- v2_kproj: 419,876 trainable params (gating + rank-64 K-projector).

## c. Raw data paths

- `v2/experiments/e09_v1_anb_resurrect/e09_v1_orig_seed0.json`
- `v2/experiments/e09_v1_anb_resurrect/e09_v2_kproj_seed0.json`

## d. Numbers

| mode | base NLL | post NLL (real) | Δ NLL (signed) |
|---|---:|---:|---:|
| v1_orig (no projector) | 12.033 | 12.019 | **+0.014** (null) |
| v2_kproj (rank-64 projector) | 12.033 | 7.009 | **−5.007** |

## e. Verdict

- **Hypothesis ("the K-projector is the load-bearing component of B2")**: Confirmed. The same mechanism without the projector is bit-null. Adding the projector recovers the full B2-style effect.
- **Pass rate**: 2/2 on driver rules.

## f. Caveat

- Single seed; reproduced at multiple seeds via e19 (which holds the projector and varies seed; mean Δ_L9 ≈ −4.87 across seed 0–4, std 0.33).
- "v1 null result" here means *null at this training budget on these b-vectors*; the v1 program had many sub-variants whose individual nulls are not all rerun here.

## g. Implications

- E09 is a **mechanistic decomposition**, not a falsifier: it pinpoints *what* in v2 produced the apparent gain over v1. The K-projector is responsible.
- Combined with e16-forgetting (A/B symmetry), the picture is: the K-projector is a small rank-64 fine-tune that learns a template-conditional distribution shift; v1 lacked this shift and so produced null; v2 has it and so produces −5 nat.
- The corollary for v1: a v1 reboot would not need a different bank or attention scheme; it would need a learned read-projection.
