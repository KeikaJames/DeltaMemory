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
| no_memory | 12.1370 | 12.1370 | 31464.2812 | 0.3438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 14.2840 | 14.2868 | 31654.0938 | 0.3438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 14.2840 | 14.2868 | 31654.0938 | 0.3438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 12.0138 | 3.0843 | 121.2500 | 0.8125 | 58.5068 | 32.1710 | 0.3590 | 1.0000 |
| delta_qv_zero | 12.1370 | 12.3009 | 32775.2500 | 0.3125 | 0.2556 | 0.1720 | 0.2715 | 1.0000 |
| delta_qv_random | 12.3191 | 12.2144 | 31959.4688 | 0.3125 | 18.3900 | 6.6559 | 0.2720 | 1.0000 |
| delta_qv_shuffled | 11.9218 | 3.0847 | 119.7812 | 0.7812 | 58.3587 | 32.0172 | 0.3586 | 1.0000 |
| delta_qv_wrong_layer | 12.0302 | 4.7955 | 1873.2500 | 0.7500 | 58.1205 | 32.3318 | 0.3590 | 1.0000 |
| delta_qv_wrong_query | 12.0122 | 3.0924 | 128.0938 | 0.7812 | 58.5068 | 32.1710 | 0.3590 | 1.0000 |
| delta_qv_identity_gate | 12.1419 | 4.8481 | 2011.8438 | 0.7500 | 21.6595 | 11.9335 | 0.3590 | 0.3710 |
| delta_qv_force_gate | 11.3346 | 2.9617 | 33.2500 | 0.7812 | 144.2544 | 60.1464 | 1.0000 | 1.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `8.929549999716983`
- `eval_delta_zero_nll_gap`: `9.216586776259646`
- `eval_delta_random_nll_gap`: `9.130089916470752`
- `eval_delta_shuffled_nll_gap`: `0.000388237000152003`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_shuffled`
- `delta_qv_identity_gate`: mean_delta=`1.7638`, ci95=`[1.3290, 2.2243]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_random`: mean_delta=`9.1301`, ci95=`[8.1276, 10.0607]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_shuffled`: mean_delta=`0.0004`, ci95=`[-0.0168, 0.0175]`, win_rate=`0.5000`, permutation_p=`0.9450`
- `delta_qv_wrong_layer`: mean_delta=`1.7112`, ci95=`[1.3008, 2.1539]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_wrong_query`: mean_delta=`0.0081`, ci95=`[-0.0113, 0.0356]`, win_rate=`0.1250`, permutation_p=`1.0000`
- `delta_qv_zero`: mean_delta=`9.2166`, ci95=`[8.3317, 10.0909]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `hidden_retrieval`: mean_delta=`11.2025`, ci95=`[10.0816, 12.2333]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `no_memory`: mean_delta=`9.0527`, ci95=`[8.1227, 9.9434]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `raw_memory`: mean_delta=`11.2025`, ci95=`[10.0816, 12.2333]`, win_rate=`1.0000`, permutation_p=`0.0065`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| delta_qv | -0.0198 |
| delta_qv_identity_gate | 0.1873 |
| delta_qv_wrong_query | -0.0159 |
| no_memory | -0.1068 |

## Address Diagnostics

- `correct_address_rank`: `4.5000`
- `paired_negative_rank`: `4.5000`
- `address_margin`: `0.0017`
- `correct_vs_paired_score_margin`: `0.0001`

- `delta_qv_margin_advantage_vs_wrong_query`: `-0.0039`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
