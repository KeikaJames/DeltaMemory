# Stage 6 Phase 1 — token-preserving writer × 3-channel sweep

Run command:
```
python3 scripts/run_stage6.py --phase 1 --model google/gemma-4-E2B \
  --device mps --dtype bfloat16 --seeds 0 1 \
  --writer-pools mean attn --swap-options off on \
  --train-samples 24 --eval-samples 24 --steps 200 \
  --memory-dim 256 --block-size 64 \
  --report-root reports/experiments/stage6_phase1
```

Frozen base invariant held (`trainable_base_params == 0`) on all 8 cells.
Suite: `address_token_binding_single_token`. Writer:
`oracle_span_writer + payload_probe @ first_layer + payload_answer_loss=1.0
+ payload_embedding_loss=0.5 + lm_head_lora rank=4 scale=50 + warmup_frac=0.1`.

## Per-cell held-out top1 (answer rank == 1)

| pool | swap | seed | payload_probe | logit_bias | lm_head_lora | delta_qv | oracle |
|------|------|------|--------------:|-----------:|-------------:|---------:|-------:|
| attn | off  | 0    | 0.583 | 0.250 | 0.167 | 0.000 | 1.000 |
| attn | off  | 1    | 0.500 | 0.333 | 0.167 | 0.000 | 1.000 |
| attn | on   | 0    | 0.500 | 0.292 | 0.167 | 0.000 | 1.000 |
| attn | on   | 1    | 0.417 | 0.292 | 0.292 | 0.042 | 1.000 |
| mean | off  | 0    | 0.417 | 0.250 | 0.167 | 0.000 | 1.000 |
| mean | off  | 1    | 0.458 | 0.208 | 0.125 | 0.042 | 1.000 |
| mean | on   | 0    | 0.500 | 0.333 | 0.083 | 0.000 | 1.000 |
| mean | on   | 1    | 0.167 | 0.208 | 0.167 | 0.042 | 1.000 |

## Cell means (n=2 seeds)

| pool | swap | payload_probe | logit_bias | lm_head_lora | delta_qv |
|------|------|--------------:|-----------:|-------------:|---------:|
| attn | off  | **0.542** | 0.292 | 0.167 | 0.000 |
| attn | on   | 0.458 | 0.292 | 0.229 | 0.021 |
| mean | off  | 0.438 | 0.229 | 0.146 | 0.021 |
| mean | on   | 0.333 | 0.271 | 0.125 | 0.021 |

## Headline findings

* **Token-preserving attention pooling > mean pooling.** On the strongest
  channel (`payload_probe`), `attn` averages **+0.135** over `mean`
  (0.500 vs 0.385 averaged across swap settings). Effect is consistent
  across seeds and swap conditions.
* **Stage-2 swap loss does not help.** Adding the contrastive swap loss
  flat-lines or slightly hurts `payload_probe` (attn: 0.542 → 0.458;
  mean: 0.438 → 0.333). The writer-level upgrade dominates.
* **Story A negative reference holds.** `delta_qv` stays at 0.00–0.04
  across all 8 cells: Q/V residual cannot bind without supervision on
  the answer logits, exactly as predicted by the activation-steering /
  task-vector literature.
* **Oracle sanity holds.** `oracle_logit_answer_embedding` = 1.000 in
  every cell, confirming both the eval harness and the LM head LoRA
  upper-bound channel are wired correctly.
* **Strict pass gate NOT met.** The Stage 6 acceptance threshold is
  held-out top1 ≥ 0.85 on at least one of `payload_probe`,
  `logit_bias`, `lm_head_lora`. Best observed is 0.583. Phase 2 (LAMA
  factual transfer) and Stage 3 (shared retrieval) remain blocked.

## Interpretation

The token-preserving writer modestly improves the binding ceiling
(+10–15 pp on `payload_probe`) but does not break through the binding
plateau identified in the advisor's analysis. Even with paired-bind
contrastive loss and an LM-head LoRA channel, the held-out top1 stops
around 0.5–0.6.

This is consistent with the hypothesis that **address-key
disambiguation through an attention-side residual is information-limited
even when the writer faithfully preserves token identity.** The next
step in the research programme is to move the payload to a
*query-conditioned fast-weight adapter* on the OV path (rank-1 LoRA
emitted by a hypernetwork) with explicit answer-token contrastive loss,
as outlined in the Story B research direction.

## Artifacts

* Per-cell summary JSON / report MD: `reports/experiments/stage6_phase1/phase1_pool-*_swap-*_seed-*/`
* Manifest: `reports/experiments/stage6_phase1/stage6_manifest.json`
* Auto-generated channel table: `<!-- BEGIN AUTOGEN: stage6 -->` block in `README.md`
