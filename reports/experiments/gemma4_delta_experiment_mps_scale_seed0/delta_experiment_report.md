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
| no_memory | 11.9019 | 11.9019 | 37832.6875 | 0.3125 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 11.9206 | 8.2896 | 13196.6875 | 0.3750 | 18.1356 | 7.1005 | 0.7126 |
| delta_qv_zero | 11.9019 | 11.9196 | 38012.7500 | 0.2812 | 0.0242 | 0.0242 | 0.2715 |
| delta_qv_random | 11.9082 | 11.8848 | 38111.1875 | 0.2500 | 1.6234 | 0.5826 | 0.2778 |
| delta_qv_shuffled | 11.9446 | 10.2355 | 25451.9375 | 0.3438 | 5.8425 | 2.7522 | 0.4719 |
| delta_qv_force_gate | 11.9392 | 7.6877 | 10637.0312 | 0.4062 | 23.5930 | 9.7017 | 1.0000 |

## Diagnosis

- `eval_delta_nll_drop`: `3.630987249314785`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `True`
- `mechanism_supported_on_eval`: `True`

## Interpretation

This experiment trains only the DeltaMemory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
