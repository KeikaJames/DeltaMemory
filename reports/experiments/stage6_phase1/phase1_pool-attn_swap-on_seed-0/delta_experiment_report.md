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
- `stage2_swap_loss_weight`: `0.5`
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
| delta_qv | 19.0887 | 4.5328 | 28.6250 | 0.1667 | 44.4171 | 21.5562 | 0.3259 | 1.0000 |
| logit_bias | 19.4968 | 8.6031 | 32535.0833 | 0.6250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| payload_probe | 12.3705 | 3.2043 | 10.3333 | 0.7083 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lm_head_lora | 19.4734 | 4.1262 | 39.0000 | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 0.0056 | 0.0056 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `diagnosis_skipped`: `True`
- `missing_modes`: `['delta_qv_random', 'delta_qv_shuffled', 'delta_qv_zero']`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `logit_bias`
- `no_memory`: mean_delta=`14.9640`, ci95=`[14.0184, 15.9035]`, win_rate=`1.0000`, permutation_p=`0.0005`
- `logit_bias`: mean_delta=`4.0702`, ci95=`[0.4467, 8.1976]`, win_rate=`0.4583`, permutation_p=`0.0475`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | -0.1836 |
| delta_qv | 0.0521 |
| delta_qv_identity_gate | -0.1361 |
| delta_qv_wrong_query | -0.0208 |
| delta_qv_oracle_correct | 0.0521 |
| delta_qv_oracle_paired | -0.0208 |
| delta_qv_oracle_correct_address_paired_payload | -0.0208 |
| delta_qv_oracle_paired_address_correct_payload | 0.0521 |
| logit_bias_oracle_correct | 4.3372 |
| logit_bias_oracle_paired | -4.7181 |
| logit_bias_correct_address_paired_payload | -4.7181 |
| logit_bias_paired_address_correct_payload | 4.3372 |
| payload_probe_oracle_correct | 1.7279 |
| payload_probe_oracle_paired | -1.7279 |
| payload_probe_correct_address_paired_payload | -1.7279 |
| payload_probe_paired_address_correct_payload | 1.7279 |
| lm_head_lora_oracle_correct | 2.8607 |
| lm_head_lora_oracle_paired | -3.5286 |
| lm_head_lora_correct_address_paired_payload | -3.5286 |
| lm_head_lora_paired_address_correct_payload | 2.8607 |

## Answer Token Discrimination

| mode | binding_margin | top1_correct | top1_paired |
| --- | ---: | ---: | ---: |
| no_memory | -0.1836 | 0.0000 | 0.0000 |
| delta_qv | 0.0521 | 0.0000 | 0.0417 |
| delta_qv_identity_gate | -0.1361 | 0.0000 | 0.0000 |
| delta_qv_wrong_query | -0.0208 | 0.0000 | 0.0417 |
| delta_qv_oracle_correct | 0.0521 | 0.0000 | 0.0417 |
| delta_qv_oracle_paired | -0.0208 | 0.0000 | 0.0417 |
| delta_qv_oracle_correct_address_paired_payload | -0.0208 | 0.0000 | 0.0417 |
| delta_qv_oracle_paired_address_correct_payload | 0.0521 | 0.0000 | 0.0417 |
| logit_bias_oracle_correct | 4.3372 | 0.2917 | 0.0417 |
| logit_bias_oracle_paired | -4.7181 | 0.0417 | 0.2917 |
| logit_bias_correct_address_paired_payload | -4.7181 | 0.0417 | 0.2917 |
| logit_bias_paired_address_correct_payload | 4.3372 | 0.2917 | 0.0417 |
| payload_probe_oracle_correct | 1.7279 | 0.5000 | 0.0417 |
| payload_probe_oracle_paired | -1.7279 | 0.0417 | 0.5000 |
| payload_probe_correct_address_paired_payload | -1.7279 | 0.0417 | 0.5000 |
| payload_probe_paired_address_correct_payload | 1.7279 | 0.5000 | 0.0417 |
| lm_head_lora_oracle_correct | 2.8607 | 0.1667 | 0.0417 |
| lm_head_lora_oracle_paired | -3.5286 | 0.0417 | 0.2083 |
| lm_head_lora_correct_address_paired_payload | -3.5286 | 0.0417 | 0.2083 |
| lm_head_lora_paired_address_correct_payload | 2.8607 | 0.1667 | 0.0417 |

## Address Diagnostics

- `correct_address_rank`: `1.4583`
- `paired_negative_rank`: `1.5417`
- `address_margin`: `0.0047`
- `correct_vs_paired_score_margin`: `0.0004`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0729`

## Stage 2 Binding Summary

| mode | answer_nll | top1_correct | top10 |
| --- | ---: | ---: | ---: |
| no_memory | 19.4968 | 0.0000 | 0.0000 |
| delta_qv | 4.5328 | 0.0000 | 0.1667 |
| payload_probe | 3.2043 | 0.5000 | 0.7083 |
| logit_bias | 8.6031 | 0.2917 | 0.6250 |
| lm_head_lora | 4.1262 | 0.1667 | 0.6667 |
| oracle_logit_answer_embedding | 0.0056 | 1.0000 | 1.0000 |

- `oracle_channel_pass`: `True`
- `payload_probe_layer_strategy`: `first_layer`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
