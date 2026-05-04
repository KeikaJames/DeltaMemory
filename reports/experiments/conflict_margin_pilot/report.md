# Mneme Conflict-Margin Pilot

## Question

Does correct Mneme improve the relative likelihood of the correct answer
over a plausible foreign answer more than foreign-memory Delta does?

This pilot was added after `delta_qv_wrong_query` remained tied with correct
`delta_qv` on answer NLL. It scores two candidates for each question:

- the correct answer for the current sample;
- the answer from a foreign eval sample.

The key margin is:

```text
foreign_answer_nll - correct_answer_nll
```

Positive is good: it means the model assigns lower NLL to the correct answer
than to the foreign answer. A query-specific memory effect should make this
margin larger under correct `delta_qv` than under `delta_qv_wrong_query`.

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `backend`: `Apple Metal / PyTorch MPS`
- `dtype`: `bfloat16`
- `steps`: `12`
- `train_samples`: `16`
- `eval_samples`: `16`
- `seed`: `5`
- `task_suites`: `adversarial_negative`, `paraphrase_nolima_style`
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
| tasks | 2 |
| total_eval_examples | 32 |
| mean_delta_qv_nll | 5.1661 |
| mean_no_memory_nll | 12.8438 |
| mean_wrong_query_nll | 5.1504 |
| mean_delta_margin | -0.0841 |
| mean_wrong_query_margin | -0.0771 |
| mean_no_memory_margin | -0.0345 |
| mean_delta_margin_advantage_vs_wrong_query | -0.0070 |

## Per Task

| task | delta_qv_nll | no_memory_nll | wrong_query_nll | delta_margin | wrong_query_margin | no_memory_margin | delta_margin_advantage_vs_wrong_query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial_negative | 5.2364 | 12.5307 | 5.2268 | -0.0807 | -0.1203 | -0.1532 | 0.0396 |
| paraphrase_nolima_style | 5.0957 | 13.1569 | 5.0740 | -0.0876 | -0.0339 | 0.0841 | -0.0537 |

## Interpretation

This is a useful negative result. Delta injection still strongly improves
answer NLL over ordinary frozen Gemma attention, but the margin test does not
show query-specific factual binding. Correct Delta does not make the correct
answer more preferred over a foreign answer than foreign-memory Delta does.

Current evidence should therefore be phrased narrowly:

```text
All-layer Delta Q/V injection is a powerful in-attention memory channel on the
current synthetic tasks, but the current task family has not yet isolated
query-specific retrieval as the causal mechanism.
```

The next dataset must be redesigned around explicit paired conflicts rather
than using a foreign answer sampled from another independent example. Each
question should have multiple plausible answers in the same local format, and
the foreign memory should encode a competing answer for the same queried key or
a near-collision key. The pass gate is a positive correct-vs-foreign margin
advantage for correct Delta over foreign-memory Delta.

