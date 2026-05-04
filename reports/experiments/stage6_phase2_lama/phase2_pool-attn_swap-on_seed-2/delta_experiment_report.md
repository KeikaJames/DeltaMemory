# Mneme Q/V Multi-Example Experiment

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
| delta_qv | 16.7534 | 0.0009 | 1.0000 | 1.0000 | 38.5509 | 18.1866 | 0.3554 | 1.0000 |
| logit_bias | 17.1618 | 0.2139 | 1.0179 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| payload_probe | 12.5070 | 0.0253 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lm_head_lora | 17.1725 | 0.0033 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 0.7927 | 0.7927 | 42.6429 | 0.9821 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `diagnosis_skipped`: `True`
- `missing_modes`: `['delta_qv_random', 'delta_qv_shuffled', 'delta_qv_zero']`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `logit_bias`
- `no_memory`: mean_delta=`17.1609`, ci95=`[16.7100, 17.5641]`, win_rate=`1.0000`, permutation_p=`0.0005`
- `logit_bias`: mean_delta=`0.2130`, ci95=`[-0.0009, 0.6407]`, win_rate=`0.1071`, permutation_p=`0.4998`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | 1.1027 |
| delta_qv | 23.0225 |
| delta_qv_identity_gate | 10.6131 |
| delta_qv_wrong_query | 22.9179 |
| delta_qv_oracle_correct | 23.0225 |
| delta_qv_oracle_paired | 12.5126 |
| delta_qv_oracle_correct_address_paired_payload | 22.9778 |
| delta_qv_oracle_paired_address_correct_payload | 12.5573 |
| logit_bias_oracle_correct | 56.1469 |
| logit_bias_oracle_paired | -25.6527 |
| logit_bias_correct_address_paired_payload | 2.9911 |
| logit_bias_paired_address_correct_payload | 27.5031 |
| payload_probe_oracle_correct | 16.2486 |
| payload_probe_oracle_paired | -8.3469 |
| payload_probe_correct_address_paired_payload | 0.2769 |
| payload_probe_paired_address_correct_payload | 7.6248 |
| lm_head_lora_oracle_correct | 26.7876 |
| lm_head_lora_oracle_paired | -11.8372 |
| lm_head_lora_correct_address_paired_payload | 1.1206 |
| lm_head_lora_paired_address_correct_payload | 13.8298 |

## Answer Token Discrimination

| mode | binding_margin | top1_correct | top1_paired |
| --- | ---: | ---: | ---: |
| no_memory | 1.0901 | 0.0000 | 0.0000 |
| delta_qv | 23.3337 | 1.0000 | 0.0000 |
| delta_qv_identity_gate | 10.6392 | 0.9286 | 0.0000 |
| delta_qv_wrong_query | 23.2203 | 1.0000 | 0.0000 |
| delta_qv_oracle_correct | 23.3337 | 1.0000 | 0.0000 |
| delta_qv_oracle_paired | 12.5218 | 0.5000 | 0.0000 |
| delta_qv_oracle_correct_address_paired_payload | 23.2891 | 1.0000 | 0.0000 |
| delta_qv_oracle_paired_address_correct_payload | 12.5665 | 0.5000 | 0.0000 |
| logit_bias_oracle_correct | 56.1487 | 0.9821 | 0.0000 |
| logit_bias_oracle_paired | -25.7530 | 0.0000 | 0.5000 |
| logit_bias_correct_address_paired_payload | 3.0206 | 0.5000 | 0.5000 |
| logit_bias_paired_address_correct_payload | 27.3751 | 0.4821 | 0.0000 |
| payload_probe_oracle_correct | 16.2486 | 1.0000 | 0.0000 |
| payload_probe_oracle_paired | -8.3469 | 0.0179 | 0.5000 |
| payload_probe_correct_address_paired_payload | 0.2769 | 0.5000 | 0.5000 |
| payload_probe_paired_address_correct_payload | 7.6248 | 0.5179 | 0.0000 |
| lm_head_lora_oracle_correct | 27.1293 | 1.0000 | 0.0000 |
| lm_head_lora_oracle_paired | -11.9100 | 0.0000 | 0.5179 |
| lm_head_lora_correct_address_paired_payload | 1.0485 | 0.5000 | 0.5000 |
| lm_head_lora_paired_address_correct_payload | 14.1708 | 0.5000 | 0.0179 |

## Address Diagnostics

- `correct_address_rank`: `1.5536`
- `paired_negative_rank`: `0.7321`
- `address_margin`: `0.0076`
- `correct_vs_paired_score_margin`: `0.0469`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.1046`

## Stage 2 Binding Summary

| mode | answer_nll | top1_correct | top10 |
| --- | ---: | ---: | ---: |
| no_memory | 17.1618 | 0.0000 | 0.0000 |
| delta_qv | 0.0009 | 1.0000 | 1.0000 |
| payload_probe | 0.0253 | 1.0000 | 1.0000 |
| logit_bias | 0.2139 | 0.9643 | 1.0000 |
| lm_head_lora | 0.0033 | 1.0000 | 1.0000 |
| oracle_logit_answer_embedding | 0.7927 | 0.9643 | 0.9821 |

- `oracle_channel_pass`: `True`
- `payload_probe_layer_strategy`: `first_layer`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
