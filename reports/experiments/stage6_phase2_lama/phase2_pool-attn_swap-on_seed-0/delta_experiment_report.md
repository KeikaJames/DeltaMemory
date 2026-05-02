# Delta Memory Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `1500`
- `lr`: `0.0005`
- `task_suite`: `factual_capital_binding`
- `train_samples`: `56`
- `eval_samples`: `56`
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
| no_memory | 17.1618 | 17.1618 | 12354.1161 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 16.7610 | 0.0010 | 1.0000 | 1.0000 | 25.8822 | 15.0541 | 0.3378 | 1.0000 |
| logit_bias | 17.1618 | 0.2166 | 1.0179 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| payload_probe | 12.6631 | 0.0268 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lm_head_lora | 17.0805 | 0.0030 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 0.7927 | 0.7927 | 42.6429 | 0.9821 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `diagnosis_skipped`: `True`
- `missing_modes`: `['delta_qv_random', 'delta_qv_shuffled', 'delta_qv_zero']`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `logit_bias`
- `no_memory`: mean_delta=`17.1608`, ci95=`[16.6955, 17.5777]`, win_rate=`1.0000`, permutation_p=`0.0005`
- `logit_bias`: mean_delta=`0.2156`, ci95=`[-0.0010, 0.5482]`, win_rate=`0.0893`, permutation_p=`0.5157`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | 1.1800 |
| delta_qv | 24.9656 |
| delta_qv_identity_gate | 6.5243 |
| delta_qv_wrong_query | 25.0347 |
| delta_qv_oracle_correct | 24.9656 |
| delta_qv_oracle_paired | 12.8786 |
| delta_qv_oracle_correct_address_paired_payload | 25.0016 |
| delta_qv_oracle_paired_address_correct_payload | 12.8427 |
| logit_bias_oracle_correct | 53.6340 |
| logit_bias_oracle_paired | -24.8857 |
| logit_bias_correct_address_paired_payload | 3.5389 |
| logit_bias_paired_address_correct_payload | 25.2093 |
| payload_probe_oracle_correct | 14.6222 |
| payload_probe_oracle_paired | -6.7654 |
| payload_probe_correct_address_paired_payload | 0.8527 |
| payload_probe_paired_address_correct_payload | 7.0041 |
| lm_head_lora_oracle_correct | 22.8297 |
| lm_head_lora_oracle_paired | -7.6976 |
| lm_head_lora_correct_address_paired_payload | 2.5526 |
| lm_head_lora_paired_address_correct_payload | 12.5794 |

## Answer Token Discrimination

| mode | binding_margin | top1_correct | top1_paired |
| --- | ---: | ---: | ---: |
| no_memory | 1.1496 | 0.0000 | 0.0000 |
| delta_qv | 25.2062 | 1.0000 | 0.0000 |
| delta_qv_identity_gate | 6.4980 | 0.1429 | 0.0000 |
| delta_qv_wrong_query | 25.2694 | 1.0000 | 0.0000 |
| delta_qv_oracle_correct | 25.2062 | 1.0000 | 0.0000 |
| delta_qv_oracle_paired | 13.0263 | 0.5000 | 0.0000 |
| delta_qv_oracle_correct_address_paired_payload | 25.2440 | 1.0000 | 0.0000 |
| delta_qv_oracle_paired_address_correct_payload | 12.9885 | 0.5000 | 0.0000 |
| logit_bias_oracle_correct | 53.8689 | 0.9821 | 0.0000 |
| logit_bias_oracle_paired | -24.7018 | 0.0000 | 0.4821 |
| logit_bias_correct_address_paired_payload | 3.6941 | 0.4821 | 0.4821 |
| logit_bias_paired_address_correct_payload | 25.4730 | 0.5000 | 0.0000 |
| payload_probe_oracle_correct | 14.6222 | 1.0000 | 0.0000 |
| payload_probe_oracle_paired | -6.7654 | 0.0179 | 0.5000 |
| payload_probe_correct_address_paired_payload | 0.8527 | 0.5000 | 0.5000 |
| payload_probe_paired_address_correct_payload | 7.0041 | 0.5179 | 0.0000 |
| lm_head_lora_oracle_correct | 23.1950 | 1.0000 | 0.0000 |
| lm_head_lora_oracle_paired | -7.9632 | 0.0179 | 0.5000 |
| lm_head_lora_correct_address_paired_payload | 2.7051 | 0.5000 | 0.5000 |
| lm_head_lora_paired_address_correct_payload | 12.5268 | 0.5179 | 0.0000 |

## Address Diagnostics

- `correct_address_rank`: `1.5714`
- `paired_negative_rank`: `0.8036`
- `address_margin`: `0.0044`
- `correct_vs_paired_score_margin`: `0.0251`

- `delta_qv_margin_advantage_vs_wrong_query`: `-0.0691`

## Stage 2 Binding Summary

| mode | answer_nll | top1_correct | top10 |
| --- | ---: | ---: | ---: |
| no_memory | 17.1618 | 0.0000 | 0.0000 |
| delta_qv | 0.0010 | 1.0000 | 1.0000 |
| payload_probe | 0.0268 | 1.0000 | 1.0000 |
| logit_bias | 0.2166 | 0.9643 | 1.0000 |
| lm_head_lora | 0.0030 | 1.0000 | 1.0000 |
| oracle_logit_answer_embedding | 0.7927 | 0.9643 | 0.9821 |

- `oracle_channel_pass`: `True`
- `payload_probe_layer_strategy`: `first_layer`

## Interpretation

This experiment trains only the Delta Memory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
