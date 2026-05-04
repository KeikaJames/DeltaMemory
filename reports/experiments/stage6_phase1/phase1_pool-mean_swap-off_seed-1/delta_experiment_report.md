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
| no_memory | 19.2244 | 19.2244 | 83152.3750 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 19.5664 | 4.2027 | 24.7083 | 0.4583 | 33.6303 | 22.0115 | 0.3143 | 1.0000 |
| logit_bias | 19.2244 | 8.0849 | 10144.4583 | 0.4583 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| payload_probe | 12.7712 | 3.7967 | 13.3750 | 0.7917 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lm_head_lora | 19.1780 | 4.0278 | 17.1250 | 0.6250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 0.0052 | 0.0052 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `diagnosis_skipped`: `True`
- `missing_modes`: `['delta_qv_random', 'delta_qv_shuffled', 'delta_qv_zero']`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `logit_bias`
- `no_memory`: mean_delta=`15.0216`, ci95=`[13.9332, 16.0515]`, win_rate=`1.0000`, permutation_p=`0.0005`
- `logit_bias`: mean_delta=`3.8821`, ci95=`[0.7920, 7.7000]`, win_rate=`0.5000`, permutation_p=`0.0470`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | 0.1061 |
| delta_qv | 0.0729 |
| delta_qv_identity_gate | 0.0791 |
| delta_qv_wrong_query | 0.0365 |
| delta_qv_oracle_correct | 0.0729 |
| delta_qv_oracle_paired | 0.0365 |
| delta_qv_oracle_correct_address_paired_payload | 0.0365 |
| delta_qv_oracle_paired_address_correct_payload | 0.0729 |
| logit_bias_oracle_correct | 1.0365 |
| logit_bias_oracle_paired | -0.8333 |
| logit_bias_correct_address_paired_payload | -0.8333 |
| logit_bias_paired_address_correct_payload | 1.0365 |
| payload_probe_oracle_correct | 0.3385 |
| payload_probe_oracle_paired | -0.3385 |
| payload_probe_correct_address_paired_payload | -0.3385 |
| payload_probe_paired_address_correct_payload | 0.3385 |
| lm_head_lora_oracle_correct | 0.3034 |
| lm_head_lora_oracle_paired | -0.2904 |
| lm_head_lora_correct_address_paired_payload | -0.2904 |
| lm_head_lora_paired_address_correct_payload | 0.3034 |

## Answer Token Discrimination

| mode | binding_margin | top1_correct | top1_paired |
| --- | ---: | ---: | ---: |
| no_memory | 0.1061 | 0.0000 | 0.0000 |
| delta_qv | 0.0729 | 0.0417 | 0.0833 |
| delta_qv_identity_gate | 0.0791 | 0.0000 | 0.0000 |
| delta_qv_wrong_query | 0.0365 | 0.0417 | 0.0833 |
| delta_qv_oracle_correct | 0.0729 | 0.0417 | 0.0833 |
| delta_qv_oracle_paired | 0.0365 | 0.0417 | 0.0833 |
| delta_qv_oracle_correct_address_paired_payload | 0.0365 | 0.0417 | 0.0833 |
| delta_qv_oracle_paired_address_correct_payload | 0.0729 | 0.0417 | 0.0833 |
| logit_bias_oracle_correct | 1.0365 | 0.2083 | 0.0000 |
| logit_bias_oracle_paired | -0.8333 | 0.0417 | 0.1667 |
| logit_bias_correct_address_paired_payload | -0.8333 | 0.0417 | 0.1667 |
| logit_bias_paired_address_correct_payload | 1.0365 | 0.2083 | 0.0000 |
| payload_probe_oracle_correct | 0.3385 | 0.4167 | 0.0417 |
| payload_probe_oracle_paired | -0.3385 | 0.0417 | 0.4167 |
| payload_probe_correct_address_paired_payload | -0.3385 | 0.0417 | 0.4167 |
| payload_probe_paired_address_correct_payload | 0.3385 | 0.4167 | 0.0417 |
| lm_head_lora_oracle_correct | 0.3034 | 0.0833 | 0.0417 |
| lm_head_lora_oracle_paired | -0.2904 | 0.0417 | 0.0833 |
| lm_head_lora_correct_address_paired_payload | -0.2904 | 0.0417 | 0.0833 |
| lm_head_lora_paired_address_correct_payload | 0.3034 | 0.0833 | 0.0417 |

## Address Diagnostics

- `correct_address_rank`: `1.5417`
- `paired_negative_rank`: `1.7500`
- `address_margin`: `0.0068`
- `correct_vs_paired_score_margin`: `0.0009`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0365`

## Stage 2 Binding Summary

| mode | answer_nll | top1_correct | top10 |
| --- | ---: | ---: | ---: |
| no_memory | 19.2244 | 0.0000 | 0.0000 |
| delta_qv | 4.2027 | 0.0417 | 0.4583 |
| payload_probe | 3.7967 | 0.4583 | 0.7917 |
| logit_bias | 8.0849 | 0.2083 | 0.4583 |
| lm_head_lora | 4.0278 | 0.1250 | 0.6250 |
| oracle_logit_answer_embedding | 0.0052 | 1.0000 | 1.0000 |

- `oracle_channel_pass`: `True`
- `payload_probe_layer_strategy`: `first_layer`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
