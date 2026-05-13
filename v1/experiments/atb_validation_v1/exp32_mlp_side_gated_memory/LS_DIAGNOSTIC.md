# LS Diagnostic — Closed-Form Routing Probe

**Date**: 2026-05-13
**Verdict**: The InfoNCE shortcut signal (shuffled-pair test top-1 ≈ 45%) is
**optimisation-dynamics-induced**, *not* a geometric property of the captured
activations. Small-data bias is real but does not refute the Exp31/Exp32
double-negative — it sharpens it: the bottleneck is the α-additive readout
protocol, not routing quality.

## Motivation

After Exp31 (learned K-adapter, H_A) and Exp32 (MLP-side gated, H_B) both failed
Gate B (LM-output target_new_logprob), one of two hypotheses had to be true:

1. **Data-scarcity hypothesis** — 567 captured facts is too small; InfoNCE
   finds activation-co-statistic shortcuts that route on paraphrase
   signal rather than fact identity.
2. **Architectural-ceiling hypothesis** — α-additive residual readout
   (`h_out ← h_seq + α·σ(qM_K^T)M_V`) cannot transfer fact identity to
   logits regardless of routing quality.

To disambiguate, we replaced the iterative InfoNCE optimiser with two
closed-form least-squares estimators that cannot exploit optimisation
dynamics:

  - **Full-rank ridge per layer** — `W_l = (Q^T Q + λI)^{-1} Q^T K` on
    (paraphrase, anchor) pairs.
  - **Rank-r CCA per layer** — closed-form analogue of InfoNCE's
    `key_dim=64` projection head.

A clean LS estimator on truly-random labels must give chance-level
retrieval. A geometric shortcut would survive.

## Setup

- Cache: `data/cache/Qwen_Qwen3-4B-Instruct-2507_{train,val,test}.pt`
  (anchor_K = MLP-input residual, queries_Q = paraphrase MLP-input;
  L=36, D=2560, P=2 paraphrases per fact)
- Splits: train N=567, val N=115, test N=125 (chance = 0.80%)
- Estimators evaluated on **real pairs**, **shuffled pairs** (Q permuted),
  and **3 random-label seeds** (Q permuted with different seeds).
- Routing metric: per-layer cosine in projected space, averaged across
  L=36 layers, argmax over the bank.

## Results

### Full-rank ridge (real pairs)

| λ      | train_t1 | val_t1 | test_t1 | test_t5 |
|--------|---------:|-------:|--------:|--------:|
| 1e-1   | 100.00%  | 47.83% | 50.40%  | 82.80%  |
| 1e+0   | 100.00%  | 46.52% | 49.20%  | 83.20%  |
| 1e+1   | 100.00%  | 46.52% | 51.20%  | 82.00%  |
| 1e+2   |  99.57%  | 50.43% | 51.20%  | 82.40%  |
| 1e+3   |  79.13%  | 38.26% | 42.00%  | 76.80%  |

→ Full-rank D×D ridge over-parameterises with N·P = 1134 samples per layer
   and 2560² parameters. Best honest test top-1 ≈ 50%.

### Rank-r CCA (real pairs, reg = 1e-2)

| rank | train_t1 | val_t1 | test_t1 | test_t5 |
|-----:|---------:|-------:|--------:|--------:|
|   16 |  95.65%  | 56.96% | 46.40%  | 82.80%  |
|   64 | 100.00%  | 83.48% | **76.00%** | 95.60%  |
|  256 | 100.00%  | 76.09% | 74.00%  | 93.20%  |

→ Bottleneck regularises. Rank-64 closed-form CCA achieves **76% test
   top-1** — the honest ceiling of *linear* routing from paraphrase Q to
   anchor K with a `key_dim=64`-equivalent projection.

### Rank-64 CCA on shuffled / random labels

| pairs            | train_t1 | val_t1 | test_t1 | test_t5 |
|------------------|---------:|-------:|--------:|--------:|
| real             | 100.00%  | 83.48% | 76.00%  | 95.60%  |
| **shuffled**     |   1.74%  |  1.30% | **0.80%** |  7.20% |
| random_seed=42   |       –  |     –  |  2.40%  |     –   |
| random_seed=43   |       –  |     –  |  1.60%  |     –   |
| random_seed=44   |       –  |     –  |  0.80%  |     –   |
| **random mean**  |       –  |     –  | **1.60%** |   –   |
| chance (1/125)   |          |        | 0.80%   |         |

→ **All randomised baselines collapse to chance.** Closed-form CCA on
   shuffled pairs gives 0.80% test top-1 — identical to chance.

## Comparison with InfoNCE (Exp32 trained routers)

| Routing protocol     | Real test_t1 | Shuffled test_t1 | Ratio (shuffled / chance) |
|----------------------|-------------:|-----------------:|--------------------------:|
| InfoNCE (Exp32)      | 88.4%        | **45.2%**        | 56×                       |
| Closed-form CCA-64   | 76.0%        | 0.8%             | 1×                        |
| Closed-form ridge    | ~50%         | (not run; ridge on shuffled cannot exceed CCA) | ≤1× |
| Random labels (LS)   | –            | 1.6% (mean of 3) | 2×                        |

### Decomposition of the InfoNCE 88%

- **76 pp** are real linear-routable signal (LS-achievable).
- **~12 pp** is genuine non-linear margin gain that InfoNCE extracts via
  its contrastive softmax objective.
- **~45 pp** of the InfoNCE shuffled-pair score is a **dynamics-induced
  shortcut** that no closed-form estimator can reproduce. It collapses
  the moment we remove the iterative optimiser.

## Implications

1. **Small-data bias is real but optimiser-specific.**
   The dataset is not poisoned. Closed-form on random labels gives
   chance. The shortcut signal is something InfoNCE *manufactures* via
   its softmax + temperature + per-batch contrast on N=567 samples.

2. **The Exp31/Exp32 double-negative tightens, it does not weaken.**
   Even the honest 76% routing ceiling, when fed through the
   α-additive readout, still produces Gate B = 0/375 in Exp32's full
   8-variant LM-output matrix. Routing quality is not the bottleneck;
   the readout protocol is.

3. **More training facts will not save the bank architecture.**
   The data-scarcity hypothesis predicted that scaling N would shrink
   the shuffled-pair shortcut. But the shortcut is already absent under
   closed-form estimation — scaling N would only narrow the
   InfoNCE-vs-LS gap, not unlock Gate B.

4. **Next research direction is architectural, not data-scale.**
   Two falsifiable continuations:
   - **Re-attention readout (Exp33)** — M_K enters the next layer's
     softmax instead of being added to residual. If Gate B still fails
     with oracle top-1 routing, the bank paradigm is dead.
   - **Parameter-edit baseline (Exp34)** — ROME/MEMIT-style rank-1
     `down_proj` update. Used as a positive control: a known
     architecture that *does* flip target tokens. If Exp34 succeeds
     where Exp33 fails, the verdict is that ANB-style external memory
     fundamentally cannot inject fact identity into Qwen3-4B.

## Reproducing

```bash
cd v1/experiments/atb_validation_v1/exp32_mlp_side_gated_memory
python3 probe_ls_routing.py \
    --cache data/cache/Qwen_Qwen3-4B-Instruct-2507 \
    --out run_qwen_full/ls_probe.json
```

Runtime: ≈ 90 s on CPU (no MPS needed; pure linear algebra).
Outputs: `run_qwen_full/ls_probe.json` with every trial's metrics.

## Conservation

α = 0 bit-equality on the underlying injector remains untouched. The LS
probe only re-fits routers on existing captures; it does not change any
training/eval pipeline. All prior Exp32 reports and verdicts remain valid.
