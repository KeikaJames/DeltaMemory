# Delta Memory Hidden Retrieval Baseline Pilot

## Question

How does Delta Q/V injection compare against a lightweight hidden-retrieval
baseline?

This baseline uses retrieved hidden/raw memory values as a late hidden-state
fusion before the output head. It is not a full RetrievalAttention or
Memorizing-Transformer implementation, but it is a non-prompt hidden retrieval
control that is closer to frontier retrieval-memory baselines than ordinary
no-memory attention.

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `backend`: `Apple Metal / PyTorch MPS`
- `dtype`: `bfloat16`
- `steps`: `12`
- `train_samples`: `16`
- `eval_samples`: `16`
- `seed`: `7`
- `task_suite`: `paired_conflict_binding`
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
| `delta_qv` | 5.8246 |
| `hidden_retrieval` | 14.5274 |
| `raw_memory` | 14.5274 |
| `no_memory` | 12.2118 |
| `delta_qv_wrong_query` | 5.8563 |
| `delta_qv_shuffled` | 5.8438 |

| metric | value |
| --- | ---: |
| delta_drop_vs_no_memory | 6.3872 |
| hidden_retrieval_drop_vs_no_memory | -2.3156 |
| delta_minus_hidden_retrieval_nll | -8.7028 |
| delta_margin_advantage_vs_wrong_query | 0.0405 |

## Interpretation

The lightweight hidden-retrieval baseline is not competitive in this setup. It
is worse than ordinary no-memory attention, while Delta Q/V injection remains
strong.

This does **not** mean Delta Memory has beaten full RetrievalAttention or
Memorizing Transformer baselines. It only rules out the current raw/hidden
late-fusion approximation as a strong baseline. A full baseline would need
retrieved KV/hidden states to participate inside attention rather than being
added after the model's final hidden state.

