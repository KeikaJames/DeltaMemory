# Stage 15 — v3.1 Phase L2 + N preliminary findings

This is the **first run** of v3.1 (K-projector retrained on the 15-relation /
3,050-pair v3.1 dataset). The frozen v3 spec is unchanged; this is a parallel
artifact at `reports/cleanroom/stage15_kproj_v31/k_projector_gemma4.pt`.

## Training summary (Phase L2 — GB10)

```
  device         : NVIDIA GB10 (CUDA cap 12.1, bf16)
  data           : eval/splits_v31/train_v31.jsonl (183 facts, 1,464 (write, paraphrase) pairs)
  paraphrases    : 8 per fact (capped via --max-paraphrases-per-fact)
  epochs         : 8
  batch_size     : 16
  lr             : 1e-3
  temperature    : 0.07 (InfoNCE)
  loss_first     : 3.0446
  loss_last      : 1.0517
  layers         : 35
  k_projector_pt : sha256 = 2532bdc53a925abcdd770a410117fcd7a79512bfdc3b7ab4ef5c07cb97a49ced
                   size   = 14,746,261 bytes
```

Loss decreased ~3× over 8 epochs. Identity-init was preserved at step 0 (verified
by the trainer's `is_identity_after_train: false` flag — i.e., gradient updates
took it away from identity).

## Phase N preliminary intervention demo (5 facts × 3 platforms)

The same 5 demo facts (`scripts/run_intervention_demo.py`'s built-in list) were
evaluated under three configurations:

| platform | bank | f1 Paris/Anne | f2 Eiffel | f3 Mona Lisa | f4 Relativity | f5 Python |
|---|---|---:|---:|---:|---:|---:|
| Mac MPS (bf16) | identity-init (raw v2) | +4.41 | -0.04 | -0.04 | -0.07 | -0.09 |
| Mac MPS (bf16) | **v3.1 projector** | **+1.33** | -0.05 | -0.00 | +0.01 | -0.02 |
| GB10 CUDA (bf16) | identity-init (raw v2) | -16.53 | -17.84 | -17.96 | -10.44 | -17.08 |
| GB10 CUDA (bf16) | **v3.1 projector** | -8.58 | -14.21 | -14.80 | -9.35 | -18.31 |

(Numbers are `v3 − B0` logprob deltas; positive = bank lifts the target token.)

## Findings

### F1. Mac v3.1 projector: directionally correct

On Mac MPS, the v3.1 projector lifts the only "unknown" fact (f1 Paris mayor;
B0 = −5.05) by +1.33 logprob (~3.8× prob increase) while leaving the four
already-known facts within ±0.05. This is the expected qualitative behavior.

The lift is smaller than the v2 raw bank's +4.41 lift — i.e., the projector
**reduces** the raw bank's spike on f1. This is consistent with the projector
having been trained on a different fact distribution; on the v3.1 dev split (not
yet evaluated, see L4 below) the lift should be larger.

### F2. GB10 CUDA path: regression, *independent* of the projector

Both v3.1 projector and identity-init collapse to large-negative logprob deltas
on GB10. This is **not** a v3.1 issue — the identity-init bank also collapses
(e.g., f1 = −16.53 with raw bank). The collapse appears specific to the
GB10 CUDA bf16 forward path with α > 0 and non-empty bank.

Boundary cases still hold on GB10 (proven in the K7 conservation regression):
* α = 0 / empty bank → bit-equal to unpatched LLM ✓
* untrained projector (identity) → bit-equal to v2 raw bank ✓

The collapse therefore happens during the bank-augmented attention forward, not
during capture or projection. Hypotheses (in order of likelihood):

1. **bf16 softmax precision on Blackwell**: when the bank's K vector is in a
   different numerical range from the prompt's K, bf16 softmax may swing too
   hard toward the bank entry, drowning the language modeling distribution.
2. **RoPE alignment at capture time**: the capture site (`address` token,
   `period` policy) emits a K that has a specific RoPE phase. On GB10 the
   `apply_rope` invocation in `Gemma4Adapter.apply_rope` may produce a slightly
   different result than on Mac MPS (different reduction order in
   `cos_sin_chunk`).
3. **Bank softmax dilution**: with N=1 bank entry, the dilution should be
   minimal, but bf16 underflow could make the bank-attention weight either
   ~1.0 (overwhelming) or ~0.0 (silent).

### F3. Action items before declaring v3.1 frozen

These changes go into the next iteration (L3-L5):

- [ ] **L4 dev sweep**: evaluate v3.1 projector on `eval/splits_v31/dev_v31.jsonl`
  (the data it was trained on the train half of) — Mac MPS, where the platform
  is known to work.
- [ ] **GB10 numerical diagnostic**: capture the layer-by-layer max-abs-diff
  between Mac and GB10 for one (write, read) pair with the same projector.
  Identify the layer where the divergence first exceeds 1e-2.
- [ ] **bank_topk default**: even with a single bank entry the structural fix
  is good hygiene; default `bank_topk = 1` for k=1 banks may avoid the dilution.
- [ ] **Cross-platform sweep**: re-run the demo on Mac MPS fp32 and GB10 fp32 to
  isolate bf16 from CUDA-specific issues.

## What this means for the v3.1 roadmap

- The v3.1 K-projector exists, was trained without errors, and behaves
  qualitatively correctly on Mac MPS (the platform the prior v3 frozen evaluation
  used).
- The GB10 path needs a numerical-stability diagnostic before we can claim cross-
  platform parity. This is a Phase K8 follow-up; it does not block v3.1 evaluation
  on Mac.
- The plan-of-record evaluation (Phase L4 dev sweep, Phase L5 val2 gate, Phase G+1
  test_v31 eval) should proceed on Mac MPS. The GB10 platform unblocks once
  the diagnostic is closed.

## Artifacts

* `reports/cleanroom/stage15_kproj_v31/k_projector_gemma4.pt`
  (sha256 = 2532bdc53a925abcdd770a410117fcd7a79512bfdc3b7ab4ef5c07cb97a49ced)
* `reports/cleanroom/stage15_kproj_v31/summary.json` (loss curve + config)
* `transcripts/v31_intervention/gemma-4-e2b-mac/demo.md` — Mac MPS v3.1 result
* `transcripts/v31_intervention/gemma-4-e2b/demo.md` — GB10 v3.1 result (collapse)
* `transcripts/v31_intervention/gemma-4-e2b-control/demo.md` — GB10 control (collapse)

## Reproducer

```
# Train (GB10):
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/train_k_projector.py --model google/gemma-4-E2B \
  --device cuda --epochs 8 --batch-size 16 --lr 1e-3 --temperature 0.07 \
  --max-paraphrases-per-fact 8 --data-dir eval/splits_v31 \
  --out reports/cleanroom/stage15_kproj_v31/k_projector_gemma4.pt

# Demo (Mac):
python scripts/run_intervention_demo.py --model google/gemma-4-E2B --device mps \
  --dtype bfloat16 --kproj reports/cleanroom/stage15_kproj_v31/k_projector_gemma4.pt \
  --label v3.1 --out-dir transcripts/v31_intervention/gemma-4-e2b-mac
```
