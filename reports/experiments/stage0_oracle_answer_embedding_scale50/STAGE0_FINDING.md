# Stage 0 finding: LM-head readout is healthy; writer is the bottleneck

## Setup
- Model: `google/gemma-4-E2B`, MPS, bfloat16
- Suite: `address_token_binding_single_token` (32 verified single-Gemma-token color codes)
- Diagnostic: feed `output_embeddings.weight[answer_first_token]` directly as a
  logit bias to the final logits, bypassing the trained writer and projector.
- `--logit-bias-scale 50.0` (Gemma's output embeddings are normalized so
  `||W[answer]||^2 ≈ 1`, requiring an explicit scale to match base logit magnitudes).

## Result

| mode                              | answer_nll | answer_rank | top10 |
| --------------------------------- | ---------: | ----------: | ----: |
| no_memory                         |    18.6584 |    72719.62 |  0.00 |
| delta_qv (trained)                |    12.2010 |    15348.12 |  0.00 |
| logit_bias (trained projector)    |    18.6584 |    72719.62 |  0.00 |
| **oracle_logit_answer_embedding** | **0.0015** |    **1.00** | **1.00** |

Per-sample under oracle: every example reaches `rank=1`, `top10=1`, `nll < 0.01`.

## Interpretation

The LM-head readout path can recover the answer token from its own output
embedding with essentially zero loss. Therefore:

- The metric / readout pipeline (`compute_answer_metrics`, `_apply_logit_bias`,
  `answer_start - 1`, single-token answer set) is correct.
- The negative payload-probe and logit-bias results are **not** caused by a
  pipeline bug.
- The bottleneck is the writer's `raw_value` representation: it does not encode
  the answer token's identity in a form that `LayerNorm + Linear → LM head` can
  decode. This is consistent with the advisor's hypothesis and with the probing
  literature (token identity decays through depth; mean-pooling across layers
  destroys what is left).

## Implication for Stage 1

Stage 1 (writer identity-carrier capacity sweep) is now the critical path. Any
writer variant whose `raw_value`, after the existing `LayerNorm + Linear`,
approximates `output_embeddings.weight[answer]` (cosine alignment) will bind
trivially through the `logit_bias` channel.

The upper-bound representation we are aiming at is therefore explicit:

```text
target_payload  ≈  W_proj^{-1}(output_embeddings.weight[answer])
```

Stage 1 must show a writer + supervision combination that places `raw_value`
close to that target on held-out examples.
