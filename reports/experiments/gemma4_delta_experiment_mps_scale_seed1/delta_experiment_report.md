# Mneme Delta Q/V Multi-Example Experiment

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
| no_memory | 12.2138 | 12.2138 | 40166.6250 | 0.2500 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 12.2562 | 7.4858 | 10599.6250 | 0.4688 | 23.0794 | 9.4779 | 0.8320 |
| delta_qv_zero | 12.2138 | 12.2266 | 40163.5938 | 0.2188 | 0.0243 | 0.0262 | 0.2715 |
| delta_qv_random | 12.2482 | 12.2238 | 40149.4688 | 0.2500 | 1.6955 | 0.5485 | 0.2629 |
| delta_qv_shuffled | 12.2065 | 10.7552 | 30093.9688 | 0.3438 | 7.2794 | 3.1570 | 0.4775 |
| delta_qv_force_gate | 12.2880 | 7.1823 | 9557.9062 | 0.4375 | 27.3579 | 11.4357 | 1.0000 |

## Diagnosis

- `eval_delta_nll_drop`: `4.770357571542263`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `True`
- `mechanism_supported_on_eval`: `True`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
