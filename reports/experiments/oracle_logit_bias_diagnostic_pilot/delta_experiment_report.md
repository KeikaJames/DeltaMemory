# Mneme Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `32`
- `lr`: `0.003`
- `task_suite`: `address_token_binding_single_token`
- `train_samples`: `16`
- `eval_samples`: `8`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `1`
- `address_margin_weight`: `0.0`
- `address_margin`: `0.1`
- `address_score_scale`: `16.0`
- `oracle_contrastive_weight`: `0.0`
- `identity_gate_beta`: `64.0`
- `identity_gate_tau`: `0.01`
- `oracle_span_writer`: `True`
- `logit_bias_loss_weight`: `1.0`
- `logit_bias_scale`: `1.0`
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_query_to_address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 18.6584 | 18.6584 | 72719.6250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 19.5555 | 19.4387 | 71041.3750 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 19.5555 | 19.4387 | 71041.3750 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| retrieved_attention | 19.5654 | 19.1102 | 67317.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 18.9739 | 4.4997 | 23.8750 | 0.2500 | 79.2306 | 99.3438 | 0.3180 | 1.0000 |
| delta_qv_zero | 18.6584 | 18.5703 | 74574.3750 | 0.0000 | 0.8827 | 0.8935 | 0.2598 | 1.0000 |
| delta_qv_random | 18.5225 | 17.5371 | 62538.1250 | 0.0000 | 23.2708 | 9.1635 | 0.2719 | 1.0000 |
| delta_qv_shuffled | 18.9739 | 4.4997 | 23.8750 | 0.2500 | 79.2306 | 99.3438 | 0.3180 | 1.0000 |
| delta_qv_wrong_layer | 19.0443 | 5.5640 | 52.5000 | 0.1250 | 72.4330 | 93.0374 | 0.3180 | 1.0000 |
| delta_qv_wrong_query | 18.9191 | 4.4848 | 23.1250 | 0.2500 | 79.2306 | 99.3438 | 0.3180 | 1.0000 |
| delta_qv_identity_gate | 18.7550 | 5.5503 | 59.2500 | 0.2500 | 27.3964 | 34.4056 | 0.3180 | 0.3457 |
| delta_qv_force_gate | 17.8278 | 5.9430 | 107.1250 | 0.1250 | 335.9102 | 169.5679 | 1.0000 | 1.0000 |
| logit_bias | 18.6584 | 16.0778 | 83774.8750 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `14.474135845899582`
- `eval_delta_zero_nll_gap`: `14.070586889982224`
- `eval_delta_random_nll_gap`: `13.03735288977623`
- `eval_delta_shuffled_nll_gap`: `0.0`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_wrong_query`
- `no_memory`: mean_delta=`14.1587`, ci95=`[12.9415, 15.5704]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `raw_memory`: mean_delta=`14.9390`, ci95=`[13.6094, 16.4676]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `hidden_retrieval`: mean_delta=`14.9390`, ci95=`[13.6094, 16.4676]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `retrieved_attention`: mean_delta=`14.6104`, ci95=`[13.1654, 16.2980]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `logit_bias`: mean_delta=`11.5781`, ci95=`[6.4893, 16.1041]`, win_rate=`0.7500`, permutation_p=`0.0305`
- `delta_qv_zero`: mean_delta=`14.0706`, ci95=`[12.9938, 15.4546]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_random`: mean_delta=`13.0374`, ci95=`[11.3974, 14.6847]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_shuffled`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `delta_qv_wrong_layer`: mean_delta=`1.0643`, ci95=`[0.4616, 1.7377]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_wrong_query`: mean_delta=`-0.0149`, ci95=`[-0.0620, 0.0288]`, win_rate=`0.5000`, permutation_p=`0.5932`
- `delta_qv_identity_gate`: mean_delta=`1.0506`, ci95=`[0.4174, 1.7710]`, win_rate=`0.8750`, permutation_p=`0.0105`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | -0.3164 |
| delta_qv | -0.0469 |
| delta_qv_identity_gate | -0.0625 |
| delta_qv_wrong_query | -0.0469 |
| delta_qv_oracle_correct | -0.0469 |
| delta_qv_oracle_paired | -0.0469 |
| delta_qv_oracle_correct_address_paired_payload | -0.0469 |
| delta_qv_oracle_paired_address_correct_payload | -0.0469 |
| logit_bias_oracle_correct | -0.2793 |
| logit_bias_oracle_paired | -0.2871 |
| logit_bias_correct_address_paired_payload | -0.2871 |
| logit_bias_paired_address_correct_payload | -0.2793 |

## Answer Token Discrimination

| mode | binding_margin | top1_correct | top1_paired |
| --- | ---: | ---: | ---: |
| no_memory | -0.3164 | 0.0000 | 0.0000 |
| delta_qv | -0.0469 | 0.1250 | 0.1250 |
| delta_qv_identity_gate | -0.0625 | 0.0000 | 0.0000 |
| delta_qv_wrong_query | -0.0469 | 0.1250 | 0.1250 |
| delta_qv_oracle_correct | -0.0469 | 0.1250 | 0.1250 |
| delta_qv_oracle_paired | -0.0469 | 0.1250 | 0.1250 |
| delta_qv_oracle_correct_address_paired_payload | -0.0469 | 0.1250 | 0.1250 |
| delta_qv_oracle_paired_address_correct_payload | -0.0469 | 0.1250 | 0.1250 |
| logit_bias_oracle_correct | -0.2793 | 0.1250 | 0.1250 |
| logit_bias_oracle_paired | -0.2871 | 0.1250 | 0.1250 |
| logit_bias_correct_address_paired_payload | -0.2871 | 0.1250 | 0.1250 |
| logit_bias_paired_address_correct_payload | -0.2793 | 0.1250 | 0.1250 |

## Address Diagnostics

- `correct_address_rank`: `1.5000`
- `paired_negative_rank`: `1.5000`
- `address_margin`: `0.0010`
- `correct_vs_paired_score_margin`: `0.0000`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0000`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
