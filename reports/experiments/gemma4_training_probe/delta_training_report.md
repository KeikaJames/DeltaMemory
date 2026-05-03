# DeltaMemory Delta Q/V Adapter Training Report

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `auto`
- `dtype`: `bfloat16`
- `steps`: `1`
- `lr`: `0.001`
- `block_size`: `128`
- `memory_dim`: `512`
- `top_k`: `2`
- `alpha_scale`: `0.2`
- `gate_bias`: `-1.0`
- `layer_id`: `14`
- `trainable_base_params`: `0`
- `training_scope`: `writer_and_qkv_projector_only`
- `prompt_insertion_used`: `False`

## Before / After

| mode | initial_nll | final_nll | final_rank | q_delta | v_delta | gate_v |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 11.7211 | 11.7211 | 37901.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 11.8004 | 11.3691 | 34723.5000 | 2.9507 | 0.5332 | 0.1904 |
| delta_qv_zero | 11.7211 | 11.7781 | 37659.7500 | 0.0038 | 0.0017 | 0.2695 |
| delta_qv_random | 11.7159 | 11.7448 | 37739.0000 | 1.5432 | 0.5536 | 0.2695 |
| delta_qv_shuffled | 11.7220 | 11.7223 | 38155.5000 | 1.5398 | 0.5307 | 0.2695 |
| delta_qv_force_gate | 11.7070 | 10.1297 | 24659.7500 | 8.0666 | 2.8030 | 1.0000 |

## Diagnosis

- `delta_qv_nll_drop`: `0.4312628507614136`
- `trained_delta_beats_zero`: `True`
- `trained_delta_beats_random`: `True`
- `trained_delta_beats_shuffled`: `True`
- `q_delta_nonzero`: `True`
- `v_delta_nonzero`: `True`
- `adapter_learned_to_use_delta`: `True`

## Interpretation

The frozen base model is not trained. This run trains only the DeltaMemory writer and Q/V intervention adapter, showing whether the external Delta path can be optimized.
A stronger scientific claim still requires trained Delta to beat zero, random, and shuffled controls on held-out examples.
