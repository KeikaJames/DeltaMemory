# DeltaMemory Delta Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `8`
- `lr`: `0.001`
- `train_samples`: `8`
- `eval_samples`: `8`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `layer_id`: `14`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 12.8407 | 12.8407 | 42341.6562 | 0.2500 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 12.8390 | 9.2081 | 15457.5625 | 0.3750 | 12.5424 | 5.5615 | 0.6367 |
| delta_qv_zero | 12.8407 | 12.8525 | 42336.7500 | 0.2500 | 0.0205 | 0.0221 | 0.2715 |
| delta_qv_random | 12.8605 | 12.8249 | 42392.1562 | 0.2500 | 1.5489 | 0.5611 | 0.2687 |
| delta_qv_shuffled | 12.8296 | 10.0721 | 21775.0312 | 0.4062 | 9.6518 | 4.4216 | 0.5840 |
| delta_qv_force_gate | 12.8105 | 8.1538 | 11464.3750 | 0.3750 | 18.6382 | 8.5876 | 1.0000 |

## Diagnosis

- `eval_delta_nll_drop`: `3.630860261619091`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `True`
- `mechanism_supported_on_eval`: `True`

## Interpretation

This experiment trains only the DeltaMemory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
