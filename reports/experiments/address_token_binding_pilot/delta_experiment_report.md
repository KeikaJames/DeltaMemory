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
| delta_qv | 12.0138 | 3.9559 | 297.1875 | 0.7188 | 54.0077 | 28.6612 | 0.3324 | 1.0000 |
| delta_qv_zero | 12.1370 | 12.1521 | 31669.2812 | 0.3125 | 0.2614 | 0.1549 | 0.2695 | 1.0000 |
| delta_qv_random | 12.3191 | 12.1461 | 31372.6875 | 0.3125 | 18.3421 | 6.5884 | 0.2708 | 1.0000 |
| delta_qv_shuffled | 11.9218 | 3.9591 | 297.3750 | 0.7188 | 53.9726 | 28.5814 | 0.3324 | 1.0000 |
| delta_qv_wrong_layer | 12.0302 | 4.7591 | 1698.7188 | 0.6250 | 53.0485 | 27.7820 | 0.3324 | 1.0000 |
| delta_qv_wrong_query | 12.0122 | 3.9606 | 298.1250 | 0.7188 | 54.0077 | 28.6612 | 0.3324 | 1.0000 |
| delta_qv_identity_gate | 12.1419 | 6.2864 | 8496.9688 | 0.5938 | 20.1218 | 10.7402 | 0.3324 | 0.3710 |
| delta_qv_force_gate | 11.3346 | 4.5822 | 1185.7188 | 0.6875 | 144.7698 | 62.4884 | 1.0000 | 1.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `8.057901141932234`
- `eval_delta_zero_nll_gap`: `8.196124909212813`
- `eval_delta_random_nll_gap`: `8.190131230046973`
- `eval_delta_shuffled_nll_gap`: `0.0031854440458118916`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_shuffled`
- `delta_qv_identity_gate`: mean_delta=`2.3305`, ci95=`[1.8959, 2.7582]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_random`: mean_delta=`8.1901`, ci95=`[7.4151, 8.8721]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_shuffled`: mean_delta=`0.0032`, ci95=`[-0.0340, 0.0408]`, win_rate=`0.5000`, permutation_p=`0.8661`
- `delta_qv_wrong_layer`: mean_delta=`0.8031`, ci95=`[0.6769, 0.9435]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_wrong_query`: mean_delta=`0.0047`, ci95=`[0.0000, 0.0114]`, win_rate=`0.2500`, permutation_p=`0.4943`
- `delta_qv_zero`: mean_delta=`8.1961`, ci95=`[7.5038, 8.8868]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `hidden_retrieval`: mean_delta=`10.3309`, ci95=`[9.4296, 11.1518]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `no_memory`: mean_delta=`8.1810`, ci95=`[7.4988, 8.8529]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `raw_memory`: mean_delta=`10.3309`, ci95=`[9.4296, 11.1518]`, win_rate=`1.0000`, permutation_p=`0.0065`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| delta_qv | 0.0889 |
| delta_qv_identity_gate | -0.0129 |
| delta_qv_wrong_query | 0.0889 |
| no_memory | -0.1068 |

## Address Diagnostics

- `correct_address_rank`: `4.5000`
- `paired_negative_rank`: `4.5000`
- `address_margin`: `0.0017`
- `correct_vs_paired_score_margin`: `0.0001`

- `delta_qv_margin_advantage_vs_wrong_query`: `-0.0000`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
