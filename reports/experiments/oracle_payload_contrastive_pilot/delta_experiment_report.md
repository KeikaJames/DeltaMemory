# Mneme Q/V Multi-Example Experiment

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
- `oracle_contrastive_weight`: `2.0`
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
| delta_qv | 11.6696 | 3.9151 | 700.9375 | 0.7500 | 91.7577 | 72.5556 | 0.4261 | 1.0000 |
| delta_qv_zero | 12.1370 | 12.1914 | 31386.1250 | 0.3125 | 0.3108 | 0.2699 | 0.2715 | 1.0000 |
| delta_qv_random | 12.0234 | 12.3226 | 31586.7188 | 0.3438 | 18.7023 | 6.6520 | 0.2704 | 1.0000 |
| delta_qv_shuffled | 11.6971 | 3.9190 | 698.0000 | 0.7500 | 91.8465 | 72.5720 | 0.4261 | 1.0000 |
| delta_qv_wrong_layer | 11.7645 | 4.6091 | 2897.7500 | 0.7188 | 92.6923 | 74.4270 | 0.4261 | 1.0000 |
| delta_qv_wrong_query | 11.7166 | 3.9295 | 708.6250 | 0.7500 | 91.7577 | 72.5556 | 0.4261 | 1.0000 |
| delta_qv_identity_gate | 12.0162 | 5.1511 | 3428.1250 | 0.5938 | 33.4229 | 26.2475 | 0.4261 | 0.3652 |
| delta_qv_force_gate | 10.4265 | 3.9100 | 647.4062 | 0.7500 | 211.2544 | 102.6122 | 1.0000 | 1.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `7.754508805461228`
- `eval_delta_zero_nll_gap`: `8.276226398535073`
- `eval_delta_random_nll_gap`: `8.407489997334778`
- `eval_delta_shuffled_nll_gap`: `0.003888762556016445`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_shuffled`
- `no_memory`: mean_delta=`8.2218`, ci95=`[7.4989, 8.8972]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `raw_memory`: mean_delta=`10.3644`, ci95=`[9.4216, 11.2023]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `hidden_retrieval`: mean_delta=`10.3644`, ci95=`[9.4216, 11.2023]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_zero`: mean_delta=`8.2762`, ci95=`[7.4734, 8.9955]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_random`: mean_delta=`8.4075`, ci95=`[7.6534, 9.1378]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_shuffled`: mean_delta=`0.0039`, ci95=`[-0.0079, 0.0151]`, win_rate=`0.5000`, permutation_p=`0.5477`
- `delta_qv_wrong_layer`: mean_delta=`0.6939`, ci95=`[0.5877, 0.8297]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_wrong_query`: mean_delta=`0.0144`, ci95=`[0.0000, 0.0361]`, win_rate=`0.2500`, permutation_p=`0.4823`
- `delta_qv_identity_gate`: mean_delta=`1.2360`, ci95=`[0.9608, 1.5797]`, win_rate=`1.0000`, permutation_p=`0.0065`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | -0.1068 |
| delta_qv | 0.1099 |
| delta_qv_identity_gate | 0.1303 |
| delta_qv_wrong_query | 0.0983 |
| delta_qv_oracle_correct | 0.1082 |
| delta_qv_oracle_paired | 0.1157 |

## Address Diagnostics

- `correct_address_rank`: `4.3750`
- `paired_negative_rank`: `4.5000`
- `address_margin`: `0.0000`
- `correct_vs_paired_score_margin`: `0.0000`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0117`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
