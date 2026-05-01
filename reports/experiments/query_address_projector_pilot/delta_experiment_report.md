# Delta Memory Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `32`
- `lr`: `0.001`
- `task_suite`: `address_token_binding`
- `train_samples`: `8`
- `eval_samples`: `8`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `address_margin_weight`: `5.0`
- `address_margin`: `0.05`
- `address_score_scale`: `32.0`
- `identity_gate_beta`: `64.0`
- `identity_gate_tau`: `0.01`
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_query_to_address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 12.1370 | 12.1370 | 31464.2812 | 0.3438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 14.2806 | 14.2796 | 31615.4375 | 0.3438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 14.2806 | 14.2796 | 31615.4375 | 0.3438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 11.6696 | 2.5695 | 12.1562 | 0.7812 | 115.1097 | 49.3721 | 0.3393 | 1.0000 |
| delta_qv_zero | 12.1370 | 12.1696 | 31126.4688 | 0.3125 | 0.3480 | 0.1509 | 0.2695 | 1.0000 |
| delta_qv_random | 12.0234 | 12.3156 | 31771.9375 | 0.2812 | 18.9320 | 6.7216 | 0.2715 | 1.0000 |
| delta_qv_shuffled | 11.6971 | 2.5528 | 12.0625 | 0.7812 | 115.1705 | 49.3359 | 0.3392 | 1.0000 |
| delta_qv_wrong_layer | 11.7645 | 2.9521 | 25.4688 | 0.7812 | 116.8510 | 50.8147 | 0.3393 | 1.0000 |
| delta_qv_wrong_query | 11.7166 | 2.5693 | 12.0938 | 0.7812 | 115.1097 | 49.3721 | 0.3393 | 1.0000 |
| delta_qv_identity_gate | 12.0162 | 4.7129 | 1036.3750 | 0.6250 | 42.3380 | 18.3306 | 0.3393 | 0.3652 |
| delta_qv_force_gate | 10.4265 | 5.0719 | 2344.3438 | 0.7188 | 265.1760 | 94.1515 | 1.0000 | 1.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `9.100135892804246`
- `eval_delta_zero_nll_gap`: `9.600095421250444`
- `eval_delta_random_nll_gap`: `9.74604341405211`
- `eval_delta_shuffled_nll_gap`: `-0.016695582424290478`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_shuffled`
- `no_memory`: mean_delta=`9.5675`, ci95=`[8.6837, 10.3047]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `raw_memory`: mean_delta=`11.7101`, ci95=`[10.5895, 12.6149]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `hidden_retrieval`: mean_delta=`11.7101`, ci95=`[10.5895, 12.6149]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_zero`: mean_delta=`9.6001`, ci95=`[8.6666, 10.3895]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_random`: mean_delta=`9.7460`, ci95=`[9.1303, 10.2546]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_shuffled`: mean_delta=`-0.0167`, ci95=`[-0.0266, -0.0084]`, win_rate=`0.1250`, permutation_p=`0.0135`
- `delta_qv_wrong_layer`: mean_delta=`0.3825`, ci95=`[0.2576, 0.5011]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_wrong_query`: mean_delta=`-0.0003`, ci95=`[-0.0182, 0.0172]`, win_rate=`0.1250`, permutation_p=`1.0000`
- `delta_qv_identity_gate`: mean_delta=`2.1434`, ci95=`[1.7009, 2.6572]`, win_rate=`1.0000`, permutation_p=`0.0065`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | -0.1068 |
| delta_qv | -0.0449 |
| delta_qv_identity_gate | 0.0936 |
| delta_qv_wrong_query | -0.0432 |

## Address Diagnostics

- `correct_address_rank`: `4.3750`
- `paired_negative_rank`: `4.5000`
- `address_margin`: `0.0000`
- `correct_vs_paired_score_margin`: `0.0000`

- `delta_qv_margin_advantage_vs_wrong_query`: `-0.0017`

## Interpretation

This experiment trains only the Delta Memory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
