# Delta Memory Question-Only Query Eval32

## Question

Does layerwise Delta Memory injection still beat ordinary frozen Gemma attention
after removing both identified evaluation shortcuts?

This run uses:

- randomized answer codes by seed, so train/eval answer sequences no longer
  match;
- retrieval queries computed only from `Question: ...\nAnswer:`, without answer
  tokens;
- all-layer in-attention Delta Q/V injection;
- no source-text prompt insertion;
- frozen Gemma base weights.

It also includes a stronger `delta_qv_wrong_query` control that injects a
foreign eval sample's Delta Memory into the current question.

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
- `retrieval_query_uses_answer`: `false`

## Global Result

| metric | value |
| --- | ---: |
| tasks | 5 |
| total_eval_examples | 160 |
| mean_delta_qv_nll | 5.5366 |
| mean_no_memory_nll | 12.7923 |
| mean_raw_memory_nll | 15.4011 |
| mean_zero_nll | 12.7336 |
| mean_random_nll | 12.6596 |
| mean_shuffled_nll | 8.4315 |
| mean_wrong_layer_nll | 6.7908 |
| mean_wrong_query_nll | 5.5399 |
| mean_delta_drop_vs_no_memory | 7.2557 |
| mean_wrong_query_drop_vs_no_memory | 7.2524 |
| mean_wrong_query_minus_delta_nll | 0.0032 |
| tasks_delta_beats_wrong_query | 3 |
| per_example_delta_beats_no_memory_rate | 1.0000 |
| per_example_delta_beats_wrong_query_rate | 0.5000 |

## Per Task

| task | delta_qv | no_memory | wrong_query | wrong_layer | shuffled | raw_memory | delta_drop_vs_no_memory | wrong_query_minus_delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single_fact_late_reference | 5.8407 | 12.6443 | 5.8563 | 7.0635 | 7.1811 | 15.2992 | 6.8037 | 0.0156 |
| multi_hop_binding | 5.1609 | 12.2376 | 5.1475 | 5.9970 | 7.4018 | 14.3114 | 7.0767 | -0.0134 |
| temporal_overwrite | 6.5907 | 13.5177 | 6.6166 | 7.4614 | 8.9149 | 15.1122 | 6.9271 | 0.0259 |
| paraphrase_nolima_style | 5.4536 | 12.9481 | 5.4254 | 8.4956 | 10.5831 | 16.7036 | 7.4946 | -0.0282 |
| adversarial_negative | 4.6374 | 12.6137 | 4.6536 | 4.9365 | 8.0768 | 15.5790 | 7.9763 | 0.0163 |

## Interpretation

The robust part of the result is now clear: with answer-pattern and retrieval
query leakage removed, all-layer Delta Q/V injection still beats ordinary
frozen Gemma attention on every held-out example in this eval32 run. Mean answer
NLL drops from `12.7923` to `5.5366`.

The mechanism-specific claim is not yet strong enough. `delta_qv_wrong_query`
is effectively tied with correct `delta_qv` (`+0.0032` NLL worse on average, 50%
per-example win rate). That means the present benchmark proves a powerful
in-attention Delta channel, but it does not yet prove that question-specific
retrieval is the causal source of the gain.

Wrong-layer and shuffled controls remain weaker than correct Delta, so there is
some evidence that layer/content alignment matters. But the next experiment
must force foreign memories to carry conflicting answer evidence and measure
answer-margin effects, not just answer NLL for the correct label.

## Dynamic next step

Do not spend more compute repeating this suite with more seeds yet. The next
priority is a conflict-control suite:

1. Build paired examples where two or more units share the same question format
   but have different answer codes.
2. Evaluate both correct-answer NLL and foreign-answer NLL under correct Delta
   vs foreign-memory Delta.
3. Require correct Delta to improve the margin
   `foreign_answer_nll - correct_answer_nll`, while foreign-memory Delta should
   shrink or reverse that margin.
4. Only after this passes should larger seeds/eval sizes be run.

