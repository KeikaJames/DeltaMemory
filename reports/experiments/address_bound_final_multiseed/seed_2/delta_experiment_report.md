# Mneme Q/V Multi-Example Experiment

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
| no_memory | 12.0347 | 12.0347 | 25711.7188 | 0.2812 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 14.1934 | 14.1962 | 25688.0000 | 0.2812 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 14.1934 | 14.1962 | 25688.0000 | 0.2812 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 12.3479 | 4.1443 | 199.1562 | 0.7188 | 179.1881 | 11.0150 | 0.2182 | 1.0000 |
| delta_qv_zero | 12.0347 | 12.1948 | 26608.3750 | 0.2812 | 0.2242 | 0.1018 | 0.2695 | 1.0000 |
| delta_qv_random | 11.9974 | 11.7460 | 26355.2188 | 0.3125 | 18.2891 | 6.4142 | 0.2689 | 1.0000 |
| delta_qv_shuffled | 12.4821 | 4.1567 | 199.7188 | 0.7188 | 179.1850 | 11.0146 | 0.2181 | 1.0000 |
| delta_qv_wrong_layer | 12.4184 | 5.2488 | 2369.2812 | 0.7500 | 184.6616 | 11.5831 | 0.2182 | 1.0000 |
| delta_qv_wrong_query | 12.3334 | 4.1443 | 199.1562 | 0.7188 | 179.1881 | 11.0150 | 0.2182 | 1.0000 |
| delta_qv_identity_gate | 12.3284 | 5.3926 | 3504.8125 | 0.6250 | 73.7737 | 4.4476 | 0.2182 | 0.4064 |
| delta_qv_force_gate | 12.7530 | 5.3612 | 3246.6875 | 0.5938 | 253.1802 | 84.8059 | 1.0000 | 1.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `8.20357554871589`
- `eval_delta_zero_nll_gap`: `8.050551352091134`
- `eval_delta_random_nll_gap`: `7.601724130101502`
- `eval_delta_shuffled_nll_gap`: `0.012445335276424885`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_wrong_query`
- `no_memory`: mean_delta=`7.8904`, ci95=`[7.4664, 8.3651]`, win_rate=`1.0000`, permutation_p=`0.0060`
- `raw_memory`: mean_delta=`10.0519`, ci95=`[9.4165, 10.7182]`, win_rate=`1.0000`, permutation_p=`0.0060`
- `hidden_retrieval`: mean_delta=`10.0519`, ci95=`[9.4165, 10.7182]`, win_rate=`1.0000`, permutation_p=`0.0060`
- `delta_qv_zero`: mean_delta=`8.0506`, ci95=`[7.7574, 8.3699]`, win_rate=`1.0000`, permutation_p=`0.0060`
- `delta_qv_random`: mean_delta=`7.6017`, ci95=`[7.0801, 8.1165]`, win_rate=`1.0000`, permutation_p=`0.0060`
- `delta_qv_shuffled`: mean_delta=`0.0124`, ci95=`[-0.0058, 0.0342]`, win_rate=`0.6250`, permutation_p=`0.3133`
- `delta_qv_wrong_layer`: mean_delta=`1.1045`, ci95=`[0.9375, 1.2520]`, win_rate=`1.0000`, permutation_p=`0.0060`
- `delta_qv_wrong_query`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `delta_qv_identity_gate`: mean_delta=`1.2484`, ci95=`[0.9141, 1.6408]`, win_rate=`1.0000`, permutation_p=`0.0060`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | -0.0001 |
| delta_qv | 0.0243 |
| delta_qv_identity_gate | 0.0843 |
| delta_qv_wrong_query | 0.0243 |

## Address Diagnostics

- `correct_address_rank`: `5.7500`
- `paired_negative_rank`: `5.2500`
- `address_margin`: `0.0036`
- `correct_vs_paired_score_margin`: `-0.0011`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0000`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
