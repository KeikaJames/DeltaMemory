# Delta Memory Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `16`
- `lr`: `0.001`
- `task_suite`: `address_token_binding`
- `train_samples`: `8`
- `eval_samples`: `8`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `address_margin_weight`: `5.0`
- `address_margin`: `0.05`
- `identity_gate_beta`: `64.0`
- `identity_gate_tau`: `0.01`
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 12.1616 | 12.1616 | 23742.0312 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 14.2103 | 14.2109 | 23838.0625 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 14.2103 | 14.2109 | 23838.0625 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 12.1193 | 5.0610 | 2690.4375 | 0.7500 | 120.3402 | 46.9011 | 0.3124 | 1.0000 |
| delta_qv_zero | 12.1616 | 12.2563 | 25757.8750 | 0.2500 | 0.2266 | 0.1089 | 0.2695 | 1.0000 |
| delta_qv_random | 12.1489 | 12.4424 | 23492.2188 | 0.2188 | 18.4045 | 6.5168 | 0.2691 | 1.0000 |
| delta_qv_shuffled | 12.0432 | 5.0574 | 2691.1875 | 0.7500 | 120.3362 | 46.8838 | 0.3124 | 1.0000 |
| delta_qv_wrong_layer | 12.0606 | 6.1377 | 4000.9688 | 0.6875 | 120.5978 | 46.3727 | 0.3124 | 1.0000 |
| delta_qv_wrong_query | 12.1126 | 5.0610 | 2690.4375 | 0.7500 | 120.3402 | 46.9011 | 0.3124 | 1.0000 |
| delta_qv_identity_gate | 12.1140 | 8.2896 | 8819.5938 | 0.3750 | 46.0804 | 18.0161 | 0.3124 | 0.3829 |
| delta_qv_force_gate | 11.9615 | 6.0353 | 7494.2812 | 0.6250 | 285.4252 | 114.8653 | 1.0000 | 1.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `7.058250606060028`
- `eval_delta_zero_nll_gap`: `7.195303313434124`
- `eval_delta_random_nll_gap`: `7.381342254579067`
- `eval_delta_shuffled_nll_gap`: `-0.0035820715129375458`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_shuffled`
- `no_memory`: mean_delta=`7.1006`, ci95=`[6.6528, 7.6368]`, win_rate=`1.0000`, permutation_p=`0.0070`
- `raw_memory`: mean_delta=`9.1499`, ci95=`[8.6594, 9.6993]`, win_rate=`1.0000`, permutation_p=`0.0070`
- `hidden_retrieval`: mean_delta=`9.1499`, ci95=`[8.6594, 9.6993]`, win_rate=`1.0000`, permutation_p=`0.0070`
- `delta_qv_zero`: mean_delta=`7.1953`, ci95=`[6.7844, 7.6666]`, win_rate=`1.0000`, permutation_p=`0.0070`
- `delta_qv_random`: mean_delta=`7.3813`, ci95=`[6.9703, 7.7277]`, win_rate=`1.0000`, permutation_p=`0.0070`
- `delta_qv_shuffled`: mean_delta=`-0.0036`, ci95=`[-0.0133, 0.0064]`, win_rate=`0.3750`, permutation_p=`0.5407`
- `delta_qv_wrong_layer`: mean_delta=`1.0767`, ci95=`[0.8842, 1.3014]`, win_rate=`1.0000`, permutation_p=`0.0070`
- `delta_qv_wrong_query`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `delta_qv_identity_gate`: mean_delta=`3.2286`, ci95=`[2.9254, 3.4996]`, win_rate=`1.0000`, permutation_p=`0.0070`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | 0.0150 |
| delta_qv | -0.0062 |
| delta_qv_identity_gate | -0.0565 |
| delta_qv_wrong_query | -0.0062 |

## Address Diagnostics

- `correct_address_rank`: `4.5000`
- `paired_negative_rank`: `4.5000`
- `address_margin`: `0.0005`
- `correct_vs_paired_score_margin`: `-0.0000`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0000`

## Interpretation

This experiment trains only the Delta Memory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
