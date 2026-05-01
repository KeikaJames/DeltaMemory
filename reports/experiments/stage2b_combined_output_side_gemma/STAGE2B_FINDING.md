# Stage 2B Finding: output-side channels train-fit but do not yet pass held-out binding

## Setup

Model: `google/gemma-4-E2B` on MPS/Metal, frozen base, oracle-span writer,
single-token answer binding suite, `payload_probe_layer_strategy=first_layer`,
`train_samples=16`, `eval_samples=16`, `steps=256`, `memory_dim=256`.

Stage 2B tests whether the Stage 1 identity carrier can be delivered through
short output-side channels before implementing fast-weight LoRA.

## Runs

| run | objective | held-out payload_probe top1 | held-out logit_bias top1 | key result |
|---|---|---:|---:|---|
| `stage2b_payload_probe_embedding_swap_gemma` | payload CE + embedding + payload swap | 0.3750 | 0.0000 | payload probe flips, logit bias untrained |
| `stage2b_logit_bias_firstlayer_gemma` | logit bias CE + logit-bias swap | 0.0000 | 0.2500 | logit bias train-fits and swaps, held-out weak |
| `stage2b_combined_output_side_gemma` | payload CE + embedding + logit bias CE + logit-bias swap | 0.3750 | 0.3125 | strongest held-out logit-bias result so far |

Positive oracle control remains healthy:

| mode | held-out NLL | rank | top1 |
|---|---:|---:|---:|
| `oracle_logit_answer_embedding` | 0.0079 | 1.0000 | 1.0000 |

## Best combined run details

Final train signals:

| metric | value |
|---|---:|
| `payload_answer_loss` | 0.7064 |
| `payload_embedding_loss` | 0.5802 |
| `logit_bias_loss` | 0.0001 |
| `stage2_swap_loss` | 0.0000 |
| `stage2_binding_margin` | 34.5625 |
| `stage2_swap_margin` | 31.3750 |

Held-out Stage 2 summary:

| mode | answer_nll | mean rank | top1 | top10 |
|---|---:|---:|---:|---:|
| `payload_probe` | 4.7193 | 24.7500 | 0.3750 | 0.3750 |
| `logit_bias` | 23.2488 | 90035.3750 | 0.3125 | 0.3750 |
| `oracle_logit_answer_embedding` | 0.0079 | 1.0000 | 1.0000 | 1.0000 |

Payload-swap controls in the best combined run:

| mode | binding margin | top1 correct | top1 paired |
|---|---:|---:|---:|
| `payload_probe_oracle_correct` | 2.5117 | 0.3750 | 0.0625 |
| `logit_bias_oracle_correct` | 6.9160 | 0.3125 | 0.0000 |
| `logit_bias_oracle_paired` | -7.3516 | 0.0000 | 0.3125 |

## Interpretation

Stage 2B establishes **train-fit output-side binding capacity**: both
`payload_probe` and `logit_bias` can learn large train-time binding and swap
margins, and paired payloads flip answer preference on held-out examples for
the subset they solve.

However, it does **not** pass the strict held-out binding gate:

- required top1 correct: `>= 0.85`;
- best held-out `payload_probe` top1: `0.3750`;
- best held-out `logit_bias` top1: `0.3125`;
- best held-out logit-bias margin is positive, but accuracy is too low.

This is not a Q/V transmission failure anymore. The failure has moved to
**generalizable payload/readout identity**, because direct output-side channels
can memorize the train set but do not yet generalize enough from 16 training
facts.

## Implementation note

During Stage 2B, `logit_bias` was corrected to use the same
`payload_probe_layer_strategy` as `payload_probe`. Before this fix, logit bias
silently averaged raw values across all layers and reintroduced the Stage 1
identity-destroying `mean_all` bug.

## Next decision

Before implementing LM-head rank-1 LoRA, run one of:

1. a larger-data output-side generalization sweep (`train_samples=64`,
   `eval_samples=64`, lower LR/longer training), or
2. implement LM-head rank-1 LoRA but treat it as a channel-capacity test, not a
   full binding success, unless held-out top1 and swap gates pass.

