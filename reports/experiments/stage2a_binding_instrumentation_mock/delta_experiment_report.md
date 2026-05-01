# Delta Memory Q/V Multi-Example Experiment

## Config

- `model`: `mock-gemma`
- `device`: `cpu`
- `dtype`: `float32`
- `steps`: `1`
- `lr`: `0.001`
- `task_suite`: `address_token_binding_single_token`
- `train_samples`: `2`
- `eval_samples`: `2`
- `block_size`: `16`
- `memory_dim`: `32`
- `top_k`: `1`
- `address_margin_weight`: `0.0`
- `address_margin`: `0.1`
- `address_score_scale`: `16.0`
- `oracle_contrastive_weight`: `0.0`
- `identity_gate_beta`: `64.0`
- `identity_gate_tau`: `0.01`
- `oracle_span_writer`: `True`
- `logit_bias_loss_weight`: `0.0`
- `logit_bias_scale`: `1.0`
- `payload_answer_loss_weight`: `1.0`
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_query_to_address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 8.2132 | 8.2132 | 1334.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 8.2110 | 8.2109 | 1335.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 8.2110 | 8.2109 | 1335.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| retrieved_attention | 8.2043 | 8.2040 | 1333.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 8.2108 | 8.2112 | 1331.5000 | 0.0000 | 0.2059 | 0.2046 | 0.2680 | 1.0000 |
| delta_qv_zero | 8.2132 | 8.2132 | 1334.5000 | 0.0000 | 0.0018 | 0.0018 | 0.2687 | 1.0000 |
| delta_qv_random | 8.2114 | 8.2077 | 1319.5000 | 0.0000 | 0.1964 | 0.1874 | 0.2686 | 1.0000 |
| delta_qv_shuffled | 8.2108 | 8.2112 | 1331.5000 | 0.0000 | 0.2059 | 0.2046 | 0.2680 | 1.0000 |
| delta_qv_wrong_layer | 8.2092 | 8.2094 | 1329.0000 | 0.0000 | 0.2059 | 0.2046 | 0.2680 | 1.0000 |
| delta_qv_wrong_query | 8.2143 | 8.2150 | 1337.0000 | 0.0000 | 0.2059 | 0.2046 | 0.2680 | 1.0000 |
| delta_qv_identity_gate | 8.2124 | 8.2125 | 1333.0000 | 0.0000 | 0.0711 | 0.0706 | 0.2680 | 0.3452 |
| delta_qv_force_gate | 8.2042 | 8.2057 | 1317.5000 | 0.0000 | 0.7744 | 0.7688 | 1.0000 | 1.0000 |
| logit_bias | 8.2132 | 8.2132 | 1334.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| payload_probe | 8.5640 | 8.5706 | 2437.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 7.9214 | 7.9214 | 748.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `-0.0004215240478515625`
- `eval_delta_zero_nll_gap`: `0.002010345458984375`
- `eval_delta_random_nll_gap`: `-0.0035300254821777344`
- `eval_delta_shuffled_nll_gap`: `0.0`
- `eval_delta_beats_zero`: `False`
- `eval_delta_beats_random`: `False`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `retrieved_attention`
- `no_memory`: mean_delta=`0.0020`, ci95=`[0.0012, 0.0028]`, win_rate=`1.0000`, permutation_p=`0.4843`
- `raw_memory`: mean_delta=`-0.0004`, ci95=`[-0.0071, 0.0064]`, win_rate=`0.5000`, permutation_p=`1.0000`
- `hidden_retrieval`: mean_delta=`-0.0004`, ci95=`[-0.0071, 0.0064]`, win_rate=`0.5000`, permutation_p=`1.0000`
- `retrieved_attention`: mean_delta=`-0.0072`, ci95=`[-0.0364, 0.0219]`, win_rate=`0.5000`, permutation_p=`1.0000`
- `logit_bias`: mean_delta=`0.0020`, ci95=`[0.0012, 0.0028]`, win_rate=`1.0000`, permutation_p=`0.4843`
- `delta_qv_zero`: mean_delta=`0.0020`, ci95=`[0.0012, 0.0028]`, win_rate=`1.0000`, permutation_p=`0.4843`
- `delta_qv_random`: mean_delta=`-0.0035`, ci95=`[-0.0069, -0.0002]`, win_rate=`0.0000`, permutation_p=`0.4843`
- `delta_qv_shuffled`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `delta_qv_wrong_layer`: mean_delta=`-0.0019`, ci95=`[-0.0029, -0.0009]`, win_rate=`0.0000`, permutation_p=`0.4843`
- `delta_qv_wrong_query`: mean_delta=`0.0038`, ci95=`[0.0001, 0.0074]`, win_rate=`1.0000`, permutation_p=`0.4843`
- `delta_qv_identity_gate`: mean_delta=`0.0013`, ci95=`[0.0008, 0.0018]`, win_rate=`1.0000`, permutation_p=`0.4843`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | 0.0449 |
| delta_qv | 0.0489 |
| delta_qv_identity_gate | 0.0463 |
| delta_qv_wrong_query | 0.0413 |
| delta_qv_oracle_correct | 0.0489 |
| delta_qv_oracle_paired | 0.0413 |
| delta_qv_oracle_correct_address_paired_payload | 0.0413 |
| delta_qv_oracle_paired_address_correct_payload | 0.0489 |
| logit_bias_oracle_correct | 0.0449 |
| logit_bias_oracle_paired | 0.0449 |
| logit_bias_correct_address_paired_payload | 0.0449 |
| logit_bias_paired_address_correct_payload | 0.0449 |
| payload_probe_oracle_correct | -0.3026 |
| payload_probe_oracle_paired | 0.3026 |
| payload_probe_correct_address_paired_payload | 0.3026 |
| payload_probe_paired_address_correct_payload | -0.3026 |

## Answer Token Discrimination

| mode | binding_margin | top1_correct | top1_paired |
| --- | ---: | ---: | ---: |
| no_memory | 0.0449 | 0.0000 | 0.0000 |
| delta_qv | 0.0489 | 0.0000 | 0.0000 |
| delta_qv_identity_gate | 0.0463 | 0.0000 | 0.0000 |
| delta_qv_wrong_query | 0.0413 | 0.0000 | 0.0000 |
| delta_qv_oracle_correct | 0.0489 | 0.0000 | 0.0000 |
| delta_qv_oracle_paired | 0.0413 | 0.0000 | 0.0000 |
| delta_qv_oracle_correct_address_paired_payload | 0.0413 | 0.0000 | 0.0000 |
| delta_qv_oracle_paired_address_correct_payload | 0.0489 | 0.0000 | 0.0000 |
| logit_bias_oracle_correct | 0.0449 | 0.0000 | 0.0000 |
| logit_bias_oracle_paired | 0.0449 | 0.0000 | 0.0000 |
| logit_bias_correct_address_paired_payload | 0.0449 | 0.0000 | 0.0000 |
| logit_bias_paired_address_correct_payload | 0.0449 | 0.0000 | 0.0000 |
| payload_probe_oracle_correct | -0.3026 | 0.0000 | 0.0000 |
| payload_probe_oracle_paired | 0.3026 | 0.0000 | 0.0000 |
| payload_probe_correct_address_paired_payload | 0.3026 | 0.0000 | 0.0000 |
| payload_probe_paired_address_correct_payload | -0.3026 | 0.0000 | 0.0000 |

## Address Diagnostics

- `correct_address_rank`: `4.0000`
- `paired_negative_rank`: `1.0000`
- `address_margin`: `0.0624`
- `correct_vs_paired_score_margin`: `-0.1436`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0076`

## Stage 2 Binding Summary

| mode | answer_nll | top1_correct | top10 |
| --- | ---: | ---: | ---: |
| no_memory | 8.2132 | 0.0000 | 0.0000 |
| raw_memory | 8.2109 | 0.0000 | 0.0000 |
| hidden_retrieval | 8.2109 | 0.0000 | 0.0000 |
| retrieved_attention | 8.2040 | 0.0000 | 0.0000 |
| delta_qv | 8.2112 | 0.0000 | 0.0000 |
| delta_qv_zero | 8.2132 | 0.0000 | 0.0000 |
| delta_qv_random | 8.2077 | 0.0000 | 0.0000 |
| delta_qv_shuffled | 8.2112 | 0.0000 | 0.0000 |
| delta_qv_wrong_layer | 8.2094 | 0.0000 | 0.0000 |
| delta_qv_wrong_query | 8.2150 | 0.0000 | 0.0000 |
| delta_qv_identity_gate | 8.2125 | 0.0000 | 0.0000 |
| delta_qv_force_gate | 8.2057 | 0.0000 | 0.0000 |
| logit_bias | 8.2132 | 0.0000 | 0.0000 |
| payload_probe | 8.5706 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 7.9214 | 0.0000 | 0.0000 |

- `oracle_channel_pass`: `False`
- `payload_probe_layer_strategy`: `first_layer`

## Interpretation

This experiment trains only the Delta Memory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
