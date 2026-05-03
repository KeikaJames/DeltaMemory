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
| no_memory | 12.3142 | 12.3142 | 40031.9375 | 0.1875 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 12.3284 | 11.4234 | 30372.7500 | 0.1875 | 3.2554 | 1.1013 | 0.3027 |
| delta_qv_zero | 12.3142 | 12.2936 | 39786.6250 | 0.1875 | 0.0067 | 0.0033 | 0.2695 |
| delta_qv_random | 12.3504 | 12.3196 | 39650.6875 | 0.1875 | 1.5376 | 0.5576 | 0.2695 |
| delta_qv_shuffled | 12.3130 | 12.0658 | 36474.0625 | 0.1875 | 1.9879 | 0.6930 | 0.2764 |
| delta_qv_force_gate | 12.4011 | 9.8100 | 18005.1875 | 0.3125 | 8.4495 | 3.6412 | 1.0000 |

## Diagnosis

- `eval_delta_nll_drop`: `0.9049676060676575`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `True`
- `mechanism_supported_on_eval`: `True`

## Interpretation

This experiment trains only the DeltaMemory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
