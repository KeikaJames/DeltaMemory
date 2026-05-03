# DeltaMemory Delta Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `auto`
- `dtype`: `bfloat16`
- `steps`: `1`
- `lr`: `0.001`
- `train_samples`: `2`
- `eval_samples`: `2`
- `block_size`: `128`
- `memory_dim`: `512`
- `top_k`: `2`
- `layer_id`: `14`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 12.0813 | 12.0813 | 34809.0000 | 0.1250 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 12.0615 | 11.9160 | 33201.8750 | 0.2500 | 2.2522 | 0.5415 | 0.2251 |
| delta_qv_zero | 12.0813 | 12.0984 | 34574.5000 | 0.1250 | 0.0040 | 0.0017 | 0.2695 |
| delta_qv_random | 12.1066 | 12.0936 | 35531.2500 | 0.1250 | 1.5391 | 0.5289 | 0.2676 |
| delta_qv_shuffled | 12.0622 | 11.9414 | 34215.3750 | 0.2500 | 2.2508 | 0.5434 | 0.2261 |
| delta_qv_force_gate | 12.1094 | 11.3594 | 28544.5000 | 0.2500 | 6.9400 | 2.4656 | 1.0000 |

## Diagnosis

- `eval_delta_nll_drop`: `0.1454954743385315`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `True`
- `mechanism_supported_on_eval`: `True`

## Interpretation

This experiment trains only the DeltaMemory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
