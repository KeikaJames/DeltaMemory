# Mneme Paired-Conflict Pilot

## Question

Can a paired conflict suite isolate query-specific factual binding?

This suite creates pairs of examples with the same unit but different ledger IDs
and different answer codes. The foreign-memory control therefore injects a
plausible competing answer for the same unit under a different ledger.

The run uses the corrected protocol:

- randomized answer codes by seed;
- question-only retrieval queries;
- no prompt insertion;
- frozen Gemma base;
- all-layer Q/V Delta injection;
- conflict-margin scoring.

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `backend`: `Apple Metal / PyTorch MPS`
- `dtype`: `bfloat16`
- `steps`: `12`
- `train_samples`: `16`
- `eval_samples`: `16`
- `seed`: `6`
- `task_suite`: `paired_conflict_binding`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `layers`: `all`
- `conflict_margins`: `true`
- `base_frozen`: `true`
- `prompt_insertion_used`: `false`
- `retrieval_query_uses_answer`: `false`

## Result

| metric | value |
| --- | ---: |
| eval_examples | 16 |
| delta_qv_nll | 4.6836 |
| no_memory_nll | 12.3341 |
| wrong_query_nll | 4.6609 |
| zero_nll | 12.2860 |
| random_nll | 12.1671 |
| shuffled_nll | 13.1904 |
| wrong_layer_nll | 5.2569 |
| delta_drop_vs_no_memory | 7.6505 |
| wrong_query_minus_delta_nll | -0.0226 |
| delta_margin | -0.2162 |
| wrong_query_margin | -0.2256 |
| no_memory_margin | -0.2168 |
| delta_margin_advantage_vs_wrong_query | 0.0094 |

## Interpretation

The in-attention Delta channel remains very strong: `delta_qv` lowers answer NLL
from `12.3341` to `4.6836`.

But the paired-conflict alignment test is still not passed. Correct Delta and
wrong-query Delta are effectively tied on answer NLL, and correct Delta only
improves the correct-vs-foreign margin over wrong-query Delta by `0.0094` NLL.
That is too small to support a query-specific binding claim.

Current evidence should therefore stay narrow:

```text
Delta Q/V injection inside attention creates a large, reproducible memory-channel
effect over ordinary frozen attention, but current synthetic suites still do not
prove that query-specific retrieval of the correct memory is the causal source.
```

The next research move should not be larger repetitions of this setup. It should
change the learning objective or architecture so the adapter is forced to use
the retrieved key/value identity:

1. train with a contrastive margin loss between correct and paired-foreign
   answers;
2. retrieve from a shared memory store containing all paired conflicts instead
   of evaluating each sample with its own isolated store;
3. add a hidden/KV retrieval baseline and matched adapter baseline before
   claiming memory-specific superiority.

