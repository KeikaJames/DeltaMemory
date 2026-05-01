# RCV-HC Delta Q/V Multi-Example Experiment

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
| no_memory | 12.2436 | 12.2436 | 37260.0625 | 0.1875 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 12.2486 | 11.4252 | 28175.3750 | 0.3750 | 3.3498 | 1.1009 | 0.2959 |
| delta_qv_zero | 12.2436 | 12.2309 | 36648.5625 | 0.1875 | 0.0065 | 0.0034 | 0.2695 |
| delta_qv_random | 12.2798 | 12.1931 | 36770.9375 | 0.1875 | 1.5509 | 0.5675 | 0.2690 |
| delta_qv_shuffled | 12.2640 | 12.1303 | 36290.0625 | 0.1875 | 1.5955 | 0.5334 | 0.2559 |
| delta_qv_force_gate | 12.2740 | 9.9021 | 16078.0625 | 0.3750 | 8.6939 | 3.7232 | 1.0000 |

## Diagnosis

- `eval_delta_nll_drop`: `0.8233859241008759`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `True`
- `mechanism_supported_on_eval`: `True`

## Interpretation

This experiment trains only the RCV-HC writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
