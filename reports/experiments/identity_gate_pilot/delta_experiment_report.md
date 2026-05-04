# Mneme Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `8`
- `lr`: `0.001`
- `task_suite`: `paired_conflict_binding`
- `train_samples`: `8`
- `eval_samples`: `8`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `address_margin_weight`: `0.0`
- `address_margin`: `0.1`
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
| no_memory | 11.6111 | 11.6111 | 56127.8438 | 0.2812 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 13.9044 | 13.9044 | 56189.8125 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 13.9044 | 13.9044 | 56189.8125 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 12.3239 | 5.3501 | 3156.4375 | 0.7500 | 50.3369 | 19.5041 | 0.3111 | 1.0000 |
| delta_qv_zero | 11.6111 | 11.7512 | 56147.8125 | 0.2500 | 0.1745 | 0.0910 | 0.2695 | 1.0000 |
| delta_qv_random | 11.7563 | 11.9493 | 52663.1562 | 0.3750 | 18.0537 | 6.4361 | 0.2694 | 1.0000 |
| delta_qv_shuffled | 12.2488 | 5.3486 | 3023.6250 | 0.7500 | 50.7420 | 19.2135 | 0.3090 | 1.0000 |
| delta_qv_wrong_layer | 11.9693 | 6.6016 | 6645.4688 | 0.6250 | 50.8504 | 19.5928 | 0.3111 | 1.0000 |
| delta_qv_wrong_query | 12.1694 | 5.3744 | 2939.0000 | 0.7500 | 50.3369 | 19.5041 | 0.3111 | 1.0000 |
| delta_qv_identity_gate | 12.0176 | 6.9899 | 11314.5000 | 0.5938 | 18.8340 | 7.3448 | 0.3111 | 0.3796 |
| delta_qv_force_gate | 12.5888 | 5.1967 | 3515.6562 | 0.7500 | 122.4741 | 47.8342 | 1.0000 | 1.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `6.973820948973298`
- `eval_delta_zero_nll_gap`: `6.401109958067536`
- `eval_delta_random_nll_gap`: `6.599159786477685`
- `eval_delta_shuffled_nll_gap`: `-0.0014509409666061401`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_shuffled`
- `delta_qv_identity_gate`: mean_delta=`1.6398`, ci95=`[1.1651, 2.1295]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_random`: mean_delta=`6.5992`, ci95=`[6.1240, 7.1227]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_shuffled`: mean_delta=`-0.0015`, ci95=`[-0.0641, 0.0632]`, win_rate=`0.3750`, permutation_p=`0.9585`
- `delta_qv_wrong_layer`: mean_delta=`1.2515`, ci95=`[0.9929, 1.5881]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_wrong_query`: mean_delta=`0.0243`, ci95=`[-0.0739, 0.1140]`, win_rate=`0.3750`, permutation_p=`0.7166`
- `delta_qv_zero`: mean_delta=`6.4011`, ci95=`[5.9286, 6.7702]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `hidden_retrieval`: mean_delta=`8.5543`, ci95=`[7.4779, 9.5898]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `no_memory`: mean_delta=`6.2610`, ci95=`[5.7433, 6.6514]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `raw_memory`: mean_delta=`8.5543`, ci95=`[7.4779, 9.5898]`, win_rate=`1.0000`, permutation_p=`0.0065`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| delta_qv | 0.0455 |
| delta_qv_identity_gate | 0.1540 |
| delta_qv_wrong_query | 0.0540 |
| no_memory | 0.2126 |

## Address Diagnostics

- `correct_address_rank`: `5.2500`
- `paired_negative_rank`: `6.2500`
- `address_margin`: `0.0007`
- `correct_vs_paired_score_margin`: `0.0003`

- `delta_qv_margin_advantage_vs_wrong_query`: `-0.0085`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
