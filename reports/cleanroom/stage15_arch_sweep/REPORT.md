# Phase L4-rev — v3.1 architectural sweep (Mac MPS, Gemma-4-E2B)

**Split**: `eval/splits_v31/dev_v31.jsonl` (N=41, sha-locked).
**Projector**: `reports/cleanroom/stage15_kproj_v31/k_projector_gemma4.pt`
  (sha256 `2532bdc5…`).
**Device**: Apple Metal / PyTorch MPS, bfloat16. Seed 0.
**LLM weights**: frozen.

## Recall@1 across architecture knobs

| Configuration | recall@1 | Δ vs baseline |
|---|---:|---:|
| baseline (joint softmax, dot, τ=1)   | **0.5585** | — |
| bank_topk = 1                         | 0.4146 | −14.4 |
| bank_topk = 2                         | 0.4610 |  −9.8 |
| bank_topk = 4                         | 0.4854 |  −7.3 |
| bank_topk = 8                         | 0.5366 |  −2.2 |
| τ = 0.25                              | 0.0585 | −50.0 |
| τ = 0.5                               | 0.3317 | −22.7 |
| τ = 0.7                               | 0.5049 |  −5.4 |
| τ = 1.5                               | 0.5415 |  −1.7 |
| cosine, τ = 0.07                      | 0.5415 |  −1.7 |
| cosine, τ = 0.1                       | 0.5390 |  −2.0 |
| cosine, τ = 0.2                       | 0.5195 |  −3.9 |
| **separate-softmax, β = 1.0**         | **0.0878** | **−47.1** ⚠️ |
| separate-softmax, β = 0.5             | 0.5244 |  −3.4 |
| separate-softmax + cosine, β = 1.0    | 0.1024 | −45.6 |

Reference points (preserved):
- B0 no-memory:      0.3512
- B1 prompt-insert:  0.6366  ← target
- B2 RAG oracle:     0.6561

## Findings

### Finding 1 — softmax dilution is NOT the bottleneck (at N=41)

Stage 14 REPORT hypothesised that for N≥30 the bank's share of the joint softmax
mass collapses, motivating top-k gating. Data here invalidate that hypothesis
for *this* projector at N=41:

  - Full softmax (top-k=0)  → 0.5585
  - Top-k=1                 → 0.4146

Hard top-k drops the soft-attention spread that v3.1's projector relies on. The
projector emits *partially* aligned scores; the joint softmax averages over
"close-enough" candidates, which acts as a denoiser. Hard top-k destroys that
averaging and hands the model a wrong slot whenever the argmax is wrong.

### Finding 2 — bank-only separate softmax with β=1 destroys the model

Independent softmaxes over (sequence) and (bank) followed by additive merge
collapses the model (recall 0.09). The reason: the bank's V vectors have
roughly head-norm scale, while the sequence's softmax-weighted V is much
smaller per token. Adding them at β=1 makes hidden state ≈ bank slot, and the
LM head decodes garbage. β=0.5 recovers most of the loss but still loses to
baseline by 3.4 pp. So the joint softmax serves a second function we did not
appreciate before: it provides a *gain control* between sequence and bank
contributions that any separate-softmax design has to recreate by hand.

### Finding 3 — cosine scoring loses at all τ

Cosine (L2-normalised) scoring loses 1.7–3.9 pp. The deployed projector was
trained with raw dot-product InfoNCE, so its outputs are not norm-canonical.
A fair comparison requires retraining the projector with cosine InfoNCE.
However, the small magnitude of the loss (~2 pp) suggests the *scale*
issue is not the dominant bottleneck — alignment is.

### Finding 4 — τ = 1 is already near-optimal

Sharpening (τ < 1) consistently hurts. Slightly softer (τ = 1.5) loses
1.7 pp. The projector's score distribution is already correctly calibrated
to the joint softmax.

## Strategic conclusion

All architectural knobs (top-k, τ, cosine, separate softmax) either tie or
lose. **The only remaining lever is the K-projector itself.** Per the user's
methodology review, the path forward is:

1. **Low-rank projector** (`rank ≪ d_h`): regularise against under-training
   on 1830 pairs (current default is full d×d).
2. **Layer-shared / layer-grouped projector**: tie weights across nearby
   layers to reduce parameters by 4–8×.
3. **Cross-relation hard negatives**: present the projector with paraphrases
   of *other* relations as negatives during InfoNCE.
4. **Joint K- and Q-projector**: project both sides to a shared sub-space.
   (Risk: changes the bit-equal regression — would need a guarded toggle.)

We do NOT pursue (5) the option that would beat B1 by relaxing the red line
(LoRA, weight editing, or attention-adapter modules outside the bank): those
are weight-editing baselines, not DeltaMemory.

## Action items

- [ ] L4-train: retrain projector with low-rank + cross-rel hard negatives on GB10
- [ ] L5: gate against `val2_v31.jsonl` once a config beats baseline by ≥3 pp
- [ ] Diagnose GB10 CUDA bf16 collapse (separate from projector) — needed for M
- [ ] Commit: do NOT promote any of the 14 sweep variants — defaults preserved

## Reproducer

```bash
python scripts/run_v31_topk_sweep.py \
  --topk-list 0 --tau-list 1.0 \
  --out reports/cleanroom/stage15_arch_sweep/baseline
# add --cosine for cosine; --separate --beta X for bank-only softmax
```
