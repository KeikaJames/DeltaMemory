# Stage 2C Finding: LM-head fast-weight LoRA is stable, payload-specific, but below held-out gate

## Setup

Model: `google/gemma-4-E2B` on MPS/Metal, frozen base, oracle-span writer,
`address_token_binding_single_token`, `payload_probe_layer_strategy=first_layer`,
`train_samples=16`, `eval_samples=16`, `steps=256`, `memory_dim=256`.

Stage 2C implements a temporary LM-head low-rank fast-weight channel:

```text
u, v = H(payload)
logits' = logits + scale * sum_r (hidden @ u_r) * (W_out @ v_r)
```

The frozen LM head is not mutated. The generated update exists only for the
current forward pass.

## Implementation note

The first rank-1 attempt without direction normalization produced very large
updates (`lm_head_lora_update_norm ≈ 36k`) and unstable evaluation. The final
implementation normalizes generated `u` and `v` directions and divides
`hidden @ u` by `sqrt(hidden_size)`, then uses `lm_head_lora_scale=50`.

## Results

| run | rank | scale | train loss | held-out NLL | held-out top1 | held-out top10 | binding margin | paired flip top1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unnormalized | 1 | 1 | unstable | 35.6559 | 0.0000 | 0.0625 | -2.4326 | 0.0000 |
| normalized | 1 | 50 | 0.0020 | 7.5877 | 0.3750 | 0.3750 | 8.0273 | 0.3750 |
| normalized | 4 | 50 | 0.0831 | 6.0782 | 0.3750 | 0.4375 | 3.2969 | 0.3750 |

Positive upper bound remains healthy:

| mode | held-out NLL | rank | top1 |
|---|---:|---:|---:|
| `oracle_logit_answer_embedding` | 0.0079 | 1.0000 | 1.0000 |

## Interpretation

LM-head LoRA is a **real improvement over the broken Q/V path for binding
diagnostics**:

- it can train-fit answer-token likelihood;
- correct payload produces positive answer-token binding margin;
- paired payload flips the sign of the margin;
- held-out paired-payload flip occurs for the same subset solved by correct
  payload.

However, it still **does not pass** the strict held-out Stage 2 gate:

- required top1 correct: `>= 0.85`;
- best top1 correct: `0.3750`;
- required paired flip: `>= 0.8`;
- best paired flip: `0.3750`.

The blocker is therefore not simply "Q/V residual cannot transmit identity."
It is now a more precise issue: the current writer/readout learns
payload-specific train facts and partial held-out structure, but not enough
generalizable answer identity to support shared retrieval.

## Decision

Do **not** proceed to Stage 3 shared retrieval yet. Shared retrieval would only
compound the unsolved channel/generalization error. The next rigorous branch is
one of:

1. larger-data output/LoRA generalization sweep (`train=64`, `eval=64`, lower LR
   or longer training), or
2. redesign the identity carrier/readout to use a more token-preserving writer,
   then re-run Stage 2.

