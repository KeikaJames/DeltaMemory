# DeltaMemory Delta Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `auto`
- `dtype`: `bfloat16`
- `steps`: `2`
- `lr`: `0.001`
- `train_samples`: `4`
- `eval_samples`: `4`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `layer_id`: `14`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 13.0605 | 13.0605 | 40484.6250 | 0.1875 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 13.0520 | 11.9027 | 29453.6250 | 0.2500 | 3.1491 | 1.6065 | 0.3828 |
| delta_qv_zero | 13.0605 | 13.0450 | 41249.2500 | 0.1875 | 0.0061 | 0.0035 | 0.2695 |
| delta_qv_random | 13.0718 | 13.0245 | 40911.3125 | 0.1875 | 1.5344 | 0.5620 | 0.2720 |
| delta_qv_shuffled | 13.0436 | 12.6590 | 36725.4375 | 0.1875 | 1.9789 | 0.8309 | 0.3066 |
| delta_qv_force_gate | 13.0333 | 10.3388 | 16877.8750 | 0.4375 | 8.3855 | 4.1965 | 1.0000 |

## Diagnosis

- `eval_delta_nll_drop`: `1.1492252349853516`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `True`
- `mechanism_supported_on_eval`: `True`

## Interpretation

This experiment trains only the DeltaMemory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
