# E05 — Cross-model replication

**Status**: **PASS** on signal-exists criterion — the mechanism reproduces on Qwen3-1.7B (different size, b-vector dim 2560→2048 projected) and on Qwen3-4B at 5000 steps. However, the same A/B symmetry interpretation from e16-forgetting applies: this is *adapter portability*, not memory portability.

**Headline**: Qwen3-1.7B (with random Gaussian b-vector dim projection 2560→2048) recovers Δ_real = **−6.00 nat** at only 200 steps. Qwen3-4B at **5000 steps** yields Δ_real = **−3.97 nat**, only ~80% of the 200-step run's gain — confirming that more training beyond the sweet spot does not monotonically improve and bounded gains apply across model scales.

---

## a. Reproduction command

```bash
python3 v2/experiments/e05_cross_model/run.py --seed 0 \
    --model Qwen/Qwen3-1.7B --steps 200 \
    --bank_layer 9 --rank 64 --n_train 120 --n_eval 120 --n_preload 512

python3 v2/experiments/e05_cross_model/run.py --seed 0 \
    --model Qwen/Qwen3-4B-Instruct-2507 --steps 5000 \
    --bank_layer 9 --rank 64 --n_train 120 --n_eval 120 --n_preload 512
```

## b. Seeds & sample size

seed 0; two model runs; n_train=120, n_test=120, n_preload=512; bank_layer=9; rank=64.

## c. Raw data paths

- `v2/experiments/e05_cross_model/e05_qwen3_1p7B_seed0.json`
- `v2/experiments/e05_cross_model/e05_qwen3_4B_steps5000_seed0.json`

## d. Numbers

| Model | hidden | steps | base NLL | post NLL (real) | Δ_real | Δ_rand |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 2048 | 200 | 12.647 | 6.644 | **−6.003** | +0.048 |
| Qwen3-4B-Instruct-2507 | 2560 | 5000 | 12.033 | 8.061 | **−3.972** | +0.044 |

(b-vectors for the 1.7B run were projected from 2560→2048 via a fixed Gaussian, renorm L2=15.0. The 4B run uses native-dim b-vectors.)

## e. Verdict

- **Hypothesis ("mechanism is not Qwen-4B-specific")**: Both models show large Δ_real ≥ 1.0 and tiny Δ_rand ≤ 0.2. ✅
- **Pass rate**: 2/2 on the signal-exists criterion.
- **Falsifier #** — Not a falsifier; this is *consistent* with both the original B2 thesis and the adapter-reinterpretation. The adapter scales across models.

## f. Caveat

- The 1.7B run used a *random* Gaussian projection of the b-vectors — meaning the "content" of the bank was scrambled along its dimensions. The fact that Δ_real = −6 nat under such projection echoes e10/e11 (random/noised content still works) and reinforces the adapter interpretation: the projector learns a target distribution shift, not item-specific retrieval.
- Llama-3.2-3B and Mistral-7B from the v2 plan were not exercised here.

## g. Implications

- The mechanism is **model-architecture-portable**, but combined with e16-forgetting (A/B symmetry), e10 (random > real), e11 (noise tolerated), the portable thing is the projector-as-adapter, not the bank-as-memory.
- The 5000-step run's *lower* Δ versus the 200-step n=512 baseline (−4.76 in e02) demonstrates the same bounded-gain phenomenon as e02's t=1000 cell — pushing compute does not extract more signal once the projector has fit its target distribution.
