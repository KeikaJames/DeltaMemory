# Delta Memory Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `64`
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
- `payload_answer_loss_weight`: `1.0`
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_query_to_address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 18.6584 | 18.6584 | 72719.6250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 19.5555 | 19.7515 | 79189.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 19.5555 | 19.7515 | 79189.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| retrieved_attention | 19.5654 | 20.4592 | 109501.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 18.9945 | 5.3487 | 53.1250 | 0.2500 | 629.7967 | 272.8114 | 0.5986 | 1.0000 |
| delta_qv_zero | 18.6584 | 17.8877 | 62603.8750 | 0.0000 | 1.8822 | 2.9206 | 0.3262 | 1.0000 |
| delta_qv_random | 18.7787 | 16.6681 | 51492.8750 | 0.0000 | 26.7123 | 11.4452 | 0.3314 | 1.0000 |
| delta_qv_shuffled | 18.9945 | 5.3487 | 53.1250 | 0.2500 | 629.7967 | 272.8114 | 0.5986 | 1.0000 |
| delta_qv_wrong_layer | 18.8919 | 4.7789 | 47.1250 | 0.1250 | 638.9376 | 281.1411 | 0.5986 | 1.0000 |
| delta_qv_wrong_query | 19.0344 | 5.3775 | 54.2500 | 0.2500 | 629.7967 | 272.8114 | 0.5986 | 1.0000 |
| delta_qv_identity_gate | 18.8938 | 5.2134 | 41.1250 | 0.2500 | 218.0545 | 94.4519 | 0.5986 | 0.3457 |
| delta_qv_force_gate | 17.1592 | 5.3299 | 50.3750 | 0.2500 | 700.8657 | 274.3368 | 1.0000 | 1.0000 |
| logit_bias | 18.6584 | 15.7359 | 59529.2500 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `13.645862996578217`
- `eval_delta_zero_nll_gap`: `12.5390664935112`
- `eval_delta_random_nll_gap`: `11.31944864988327`
- `eval_delta_shuffled_nll_gap`: `0.0`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_wrong_layer`
- `no_memory`: mean_delta=`13.3097`, ci95=`[11.3685, 15.3800]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `raw_memory`: mean_delta=`14.4028`, ci95=`[12.4234, 16.4850]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `hidden_retrieval`: mean_delta=`14.4028`, ci95=`[12.4234, 16.4850]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `retrieved_attention`: mean_delta=`15.1106`, ci95=`[13.0993, 17.2280]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `logit_bias`: mean_delta=`10.3872`, ci95=`[5.7804, 14.3690]`, win_rate=`0.7500`, permutation_p=`0.0305`
- `delta_qv_zero`: mean_delta=`12.5391`, ci95=`[10.8246, 14.4243]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_random`: mean_delta=`11.3194`, ci95=`[9.2967, 13.2855]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_shuffled`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `delta_qv_wrong_layer`: mean_delta=`-0.5697`, ci95=`[-1.2885, 0.2094]`, win_rate=`0.2500`, permutation_p=`0.2194`
- `delta_qv_wrong_query`: mean_delta=`0.0288`, ci95=`[-0.0025, 0.0718]`, win_rate=`0.7500`, permutation_p=`0.1719`
- `delta_qv_identity_gate`: mean_delta=`-0.1353`, ci95=`[-0.9899, 0.7840]`, win_rate=`0.3750`, permutation_p=`0.7801`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | -0.3164 |
| delta_qv | -0.0156 |
| delta_qv_identity_gate | 0.0938 |
| delta_qv_wrong_query | -0.0781 |
| delta_qv_oracle_correct | -0.0156 |
| delta_qv_oracle_paired | -0.0781 |
| delta_qv_oracle_correct_address_paired_payload | -0.0781 |
| delta_qv_oracle_paired_address_correct_payload | -0.0156 |
| logit_bias_oracle_correct | -0.2812 |
| logit_bias_oracle_paired | -0.2969 |
| logit_bias_correct_address_paired_payload | -0.2969 |
| logit_bias_paired_address_correct_payload | -0.2812 |
| payload_probe_oracle_correct | 0.0625 |
| payload_probe_oracle_paired | -0.0625 |
| payload_probe_correct_address_paired_payload | -0.0625 |
| payload_probe_paired_address_correct_payload | 0.0625 |

## Answer Token Discrimination

| mode | binding_margin | top1_correct | top1_paired |
| --- | ---: | ---: | ---: |
| no_memory | -0.3164 | 0.0000 | 0.0000 |
| delta_qv | -0.0156 | 0.0000 | 0.1250 |
| delta_qv_identity_gate | 0.0938 | 0.0000 | 0.0000 |
| delta_qv_wrong_query | -0.0781 | 0.0000 | 0.1250 |
| delta_qv_oracle_correct | -0.0156 | 0.0000 | 0.1250 |
| delta_qv_oracle_paired | -0.0781 | 0.0000 | 0.1250 |
| delta_qv_oracle_correct_address_paired_payload | -0.0781 | 0.0000 | 0.1250 |
| delta_qv_oracle_paired_address_correct_payload | -0.0156 | 0.0000 | 0.1250 |
| logit_bias_oracle_correct | -0.2812 | 0.0000 | 0.0000 |
| logit_bias_oracle_paired | -0.2969 | 0.0000 | 0.0000 |
| logit_bias_correct_address_paired_payload | -0.2969 | 0.0000 | 0.0000 |
| logit_bias_paired_address_correct_payload | -0.2812 | 0.0000 | 0.0000 |
| payload_probe_oracle_correct | 0.0625 | 0.0000 | 0.0000 |
| payload_probe_oracle_paired | -0.0625 | 0.0000 | 0.0000 |
| payload_probe_correct_address_paired_payload | -0.0625 | 0.0000 | 0.0000 |
| payload_probe_paired_address_correct_payload | 0.0625 | 0.0000 | 0.0000 |

## Address Diagnostics

- `correct_address_rank`: `1.5000`
- `paired_negative_rank`: `1.5000`
- `address_margin`: `0.0010`
- `correct_vs_paired_score_margin`: `0.0000`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0625`

## Interpretation

This experiment trains only the Delta Memory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
