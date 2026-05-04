# Mneme Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `200`
- `lr`: `0.0005`
- `task_suite`: `address_token_binding_single_token`
- `train_samples`: `24`
- `eval_samples`: `24`
- `block_size`: `64`
- `memory_dim`: `256`
- `top_k`: `1`
- `address_margin_weight`: `0.0`
- `address_margin`: `0.1`
- `address_score_scale`: `16.0`
- `oracle_contrastive_weight`: `0.0`
- `identity_gate_beta`: `64.0`
- `identity_gate_tau`: `0.01`
- `oracle_span_writer`: `True`
- `logit_bias_loss_weight`: `0.5`
- `logit_bias_scale`: `50.0`
- `payload_answer_loss_weight`: `1.0`
- `payload_probe_layer_strategy`: `first_layer`
- `payload_embedding_loss_weight`: `0.5`
- `stage2_swap_loss_weight`: `0.0`
- `stage2_swap_margin`: `2.0`
- `stage2_swap_mode`: `lm_head_lora`
- `lm_head_lora_loss_weight`: `0.5`
- `lm_head_lora_rank`: `4`
- `lm_head_lora_scale`: `50.0`
- `eval_injection_modes`: `no_memory,delta_qv,payload_probe,logit_bias,lm_head_lora,oracle_logit_answer_embedding`
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_query_to_address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 19.4968 | 19.4968 | 88275.6250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 19.2363 | 4.8354 | 36.5417 | 0.3750 | 40.2559 | 26.8377 | 0.3193 | 1.0000 |
| logit_bias | 19.4968 | 9.4760 | 48375.6250 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| payload_probe | 12.3711 | 3.8942 | 13.2083 | 0.7083 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lm_head_lora | 19.4721 | 4.3665 | 21.5833 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 0.0056 | 0.0056 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `diagnosis_skipped`: `True`
- `missing_modes`: `['delta_qv_random', 'delta_qv_shuffled', 'delta_qv_zero']`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `logit_bias`
- `no_memory`: mean_delta=`14.6614`, ci95=`[13.6972, 15.5766]`, win_rate=`1.0000`, permutation_p=`0.0005`
- `logit_bias`: mean_delta=`4.6406`, ci95=`[1.0103, 8.7838]`, win_rate=`0.4583`, permutation_p=`0.0180`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | -0.1836 |
| delta_qv | 0.0208 |
| delta_qv_identity_gate | -0.1127 |
| delta_qv_wrong_query | -0.0469 |
| delta_qv_oracle_correct | 0.0208 |
| delta_qv_oracle_paired | -0.0469 |
| delta_qv_oracle_correct_address_paired_payload | -0.0469 |
| delta_qv_oracle_paired_address_correct_payload | 0.0208 |
| logit_bias_oracle_correct | 0.9818 |
| logit_bias_oracle_paired | -1.3307 |
| logit_bias_correct_address_paired_payload | -1.3307 |
| logit_bias_paired_address_correct_payload | 0.9818 |
| payload_probe_oracle_correct | 0.4089 |
| payload_probe_oracle_paired | -0.4089 |
| payload_probe_correct_address_paired_payload | -0.4089 |
| payload_probe_paired_address_correct_payload | 0.4089 |
| lm_head_lora_oracle_correct | 0.1419 |
| lm_head_lora_oracle_paired | -0.7109 |
| lm_head_lora_correct_address_paired_payload | -0.7109 |
| lm_head_lora_paired_address_correct_payload | 0.1419 |

## Answer Token Discrimination

| mode | binding_margin | top1_correct | top1_paired |
| --- | ---: | ---: | ---: |
| no_memory | -0.1836 | 0.0000 | 0.0000 |
| delta_qv | 0.0208 | 0.0000 | 0.0000 |
| delta_qv_identity_gate | -0.1127 | 0.0000 | 0.0000 |
| delta_qv_wrong_query | -0.0469 | 0.0000 | 0.0000 |
| delta_qv_oracle_correct | 0.0208 | 0.0000 | 0.0000 |
| delta_qv_oracle_paired | -0.0469 | 0.0000 | 0.0000 |
| delta_qv_oracle_correct_address_paired_payload | -0.0469 | 0.0000 | 0.0000 |
| delta_qv_oracle_paired_address_correct_payload | 0.0208 | 0.0000 | 0.0000 |
| logit_bias_oracle_correct | 0.9818 | 0.2083 | 0.0000 |
| logit_bias_oracle_paired | -1.3307 | 0.0417 | 0.2500 |
| logit_bias_correct_address_paired_payload | -1.3307 | 0.0417 | 0.2500 |
| logit_bias_paired_address_correct_payload | 0.9818 | 0.2083 | 0.0000 |
| payload_probe_oracle_correct | 0.4089 | 0.4167 | 0.0000 |
| payload_probe_oracle_paired | -0.4089 | 0.0000 | 0.4167 |
| payload_probe_correct_address_paired_payload | -0.4089 | 0.0000 | 0.4167 |
| payload_probe_paired_address_correct_payload | 0.4089 | 0.4167 | 0.0000 |
| lm_head_lora_oracle_correct | 0.1419 | 0.1250 | 0.0000 |
| lm_head_lora_oracle_paired | -0.7109 | 0.0000 | 0.1667 |
| lm_head_lora_correct_address_paired_payload | -0.7109 | 0.0000 | 0.1667 |
| lm_head_lora_paired_address_correct_payload | 0.1419 | 0.1250 | 0.0000 |

## Address Diagnostics

- `correct_address_rank`: `1.4583`
- `paired_negative_rank`: `1.5417`
- `address_margin`: `0.0048`
- `correct_vs_paired_score_margin`: `0.0004`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0677`

## Stage 2 Binding Summary

| mode | answer_nll | top1_correct | top10 |
| --- | ---: | ---: | ---: |
| no_memory | 19.4968 | 0.0000 | 0.0000 |
| delta_qv | 4.8354 | 0.0000 | 0.3750 |
| payload_probe | 3.8942 | 0.4167 | 0.7083 |
| logit_bias | 9.4760 | 0.2500 | 0.4167 |
| lm_head_lora | 4.3665 | 0.1667 | 0.4167 |
| oracle_logit_answer_embedding | 0.0056 | 1.0000 | 1.0000 |

- `oracle_channel_pass`: `True`
- `payload_probe_layer_strategy`: `first_layer`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
