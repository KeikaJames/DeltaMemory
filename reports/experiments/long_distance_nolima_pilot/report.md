# Delta Memory Long-Distance NoLiMa-Style Pilot

## Question

Does Delta Memory survive a longer non-literal retrieval setting?

This suite places a unit-to-alias crosswalk, long unrelated filler, and an
answer-bearing policy memorandum far apart. The question uses the hardware
identifier, while the answer is attached to the alias. Retrieval queries are
question-only and source text is not inserted into the prompt.

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `backend`: `Apple Metal / PyTorch MPS`
- `dtype`: `bfloat16`
- `steps`: `12`
- `train_samples`: `8`
- `eval_samples`: `8`
- `seed`: `8`
- `task_suite`: `long_distance_nolima_style`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `layers`: `all`
- `shared_memory_retrieval`: `true`
- `base_frozen`: `true`
- `prompt_insertion_used`: `false`
- `retrieval_query_uses_answer`: `false`

## Result

| mode | answer_nll |
| --- | ---: |
| `delta_qv` | 4.9367 |
| `no_memory` | 11.8879 |
| `hidden_retrieval` | 15.3602 |
| `delta_qv_zero` | 11.8698 |
| `delta_qv_random` | 11.7319 |
| `delta_qv_shuffled` | 4.8210 |
| `delta_qv_wrong_query` | 4.9496 |
| `delta_qv_wrong_layer` | 8.4190 |

| metric | value |
| --- | ---: |
| delta_drop_vs_no_memory | 6.9512 |
| delta_minus_shuffled_nll | 0.1157 |
| delta_margin_advantage_vs_wrong_query | -0.0001 |
| mechanism_supported_on_eval | false |

## Interpretation

Delta Q/V injection still greatly improves answer NLL over ordinary attention,
zero/random controls, and the lightweight hidden retrieval baseline. But it
does not beat shuffled memory on this pilot, and its correct-vs-wrong-query
margin advantage is effectively zero.

This again supports the narrow channel claim, not a query-specific retrieval
claim. The long-distance suite is now available for future runs, but this pilot
does not justify larger confirmation until the retrieval-alignment mechanism is
fixed.

