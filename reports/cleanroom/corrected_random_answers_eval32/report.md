# Delta Memory Corrected Random-Answer Eval32

## Question

Does layerwise Delta Memory injection still beat ordinary frozen Gemma
attention after removing the deterministic train/eval answer-pattern shortcut?

This corrected run changes the synthetic datasets so each seed samples answer
codes randomly. The train-like and eval-like answer sequences are no longer the
same. The run also includes a stronger `delta_qv_wrong_query` control that
injects a foreign sample's Delta Memory into the current question.

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `backend`: `Apple Metal / PyTorch MPS`
- `dtype`: `bfloat16`
- `steps`: `12`
- `train_samples`: `16`
- `eval_samples`: `32`
- `seed`: `4`
- `task_suites`: `single_fact_late_reference`, `multi_hop_binding`, `temporal_overwrite`, `paraphrase_nolima_style`, `adversarial_negative`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `layers`: `all`
- `injected_layers`: `15`
- `base_frozen`: `true`
- `prompt_insertion_used`: `false`

## Global Result

| metric | value |
| --- | ---: |
| tasks | 5 |
| total_eval_examples | 160 |
| mean_delta_qv_nll | 5.4738 |
| mean_no_memory_nll | 12.7923 |
| mean_raw_memory_nll | 15.4011 |
| mean_zero_nll | 12.7705 |
| mean_random_nll | 12.6584 |
| mean_shuffled_nll | 8.2889 |
| mean_wrong_layer_nll | 6.4812 |
| mean_wrong_query_nll | 5.4568 |
| mean_delta_drop_vs_no_memory | 7.3185 |
| mean_wrong_query_drop_vs_no_memory | 7.3356 |
| mean_wrong_query_minus_delta_nll | -0.0170 |
| tasks_delta_beats_wrong_query | 1 |
| per_example_delta_beats_no_memory_rate | 1.0000 |
| per_example_delta_beats_wrong_query_rate | 0.4688 |

## Per Task

| task | delta_qv | no_memory | wrong_query | wrong_layer | shuffled | raw_memory | delta_drop_vs_no_memory | wrong_query_minus_delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single_fact_late_reference | 5.4290 | 12.6443 | 5.4467 | 6.4449 | 5.9196 | 15.2993 | 7.2154 | 0.0177 |
| multi_hop_binding | 5.2328 | 12.2376 | 5.2133 | 5.3551 | 10.9378 | 14.3114 | 7.0048 | -0.0196 |
| temporal_overwrite | 5.5855 | 13.5177 | 5.5569 | 6.3121 | 8.0575 | 15.1122 | 7.9322 | -0.0286 |
| paraphrase_nolima_style | 5.5354 | 12.9481 | 5.5022 | 8.9723 | 9.0428 | 16.7036 | 7.4127 | -0.0332 |
| adversarial_negative | 5.5861 | 12.6137 | 5.5647 | 5.3217 | 7.4869 | 15.5790 | 7.0276 | -0.0214 |

## Interpretation

The original strong Delta-vs-ordinary-attention result survives the answer
randomization fix: all-layer `delta_qv` beats ordinary frozen Gemma attention on
every held-out example in this 160-example run, with mean NLL `5.4738` vs
ordinary attention `12.7923`.

However, the foreign-memory `delta_qv_wrong_query` control is almost identical
to correct `delta_qv` and is slightly better on mean NLL. This means the current
benchmark does **not** yet prove query-specific retrieval alignment. The current
evidence supports a narrower claim: trained Delta Q/V injection inside attention
creates a large memory-channel gain over ordinary attention, but the synthetic
tasks still allow a task-level or answer-format activation effect.

The next rigorous step is not more seeds on this same suite. It is a conflict
suite where foreign memories contain plausible but different answer values and
where injecting a foreign Delta should push the model toward the wrong answer if
query-memory alignment is not enforced.

