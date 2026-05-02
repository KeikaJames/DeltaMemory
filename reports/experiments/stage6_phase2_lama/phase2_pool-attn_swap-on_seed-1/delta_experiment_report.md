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
| delta_qv | 17.7331 | 0.0014 | 1.0000 | 1.0000 | 39.2081 | 12.5762 | 0.3144 | 1.0000 |
| logit_bias | 17.1618 | 0.2033 | 1.0179 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| payload_probe | 12.7880 | 0.0300 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lm_head_lora | 16.9733 | 0.0048 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 0.7927 | 0.7927 | 42.6429 | 0.9821 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `diagnosis_skipped`: `True`
- `missing_modes`: `['delta_qv_random', 'delta_qv_shuffled', 'delta_qv_zero']`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `logit_bias`
- `no_memory`: mean_delta=`17.1604`, ci95=`[16.7393, 17.5687]`, win_rate=`1.0000`, permutation_p=`0.0005`
- `logit_bias`: mean_delta=`0.2019`, ci95=`[-0.0015, 0.6145]`, win_rate=`0.0893`, permutation_p=`0.4963`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | 1.1260 |
| delta_qv | 19.6637 |
| delta_qv_identity_gate | 4.4770 |
| delta_qv_wrong_query | 19.6929 |
| delta_qv_oracle_correct | 19.6637 |
| delta_qv_oracle_paired | 9.9037 |
| delta_qv_oracle_correct_address_paired_payload | 19.6939 |
| delta_qv_oracle_paired_address_correct_payload | 9.8736 |
| logit_bias_oracle_correct | 50.6529 |
| logit_bias_oracle_paired | -21.3487 |
| logit_bias_correct_address_paired_payload | 5.6483 |
| logit_bias_paired_address_correct_payload | 23.6560 |
| payload_probe_oracle_correct | 15.0615 |
| payload_probe_oracle_paired | -7.1265 |
| payload_probe_correct_address_paired_payload | 1.3978 |
| payload_probe_paired_address_correct_payload | 6.5372 |
| lm_head_lora_oracle_correct | 23.3221 |
| lm_head_lora_oracle_paired | -8.5368 |
| lm_head_lora_correct_address_paired_payload | 3.6706 |
| lm_head_lora_paired_address_correct_payload | 11.1146 |

## Answer Token Discrimination

| mode | binding_margin | top1_correct | top1_paired |
| --- | ---: | ---: | ---: |
| no_memory | 1.0814 | 0.0000 | 0.0000 |
| delta_qv | 19.8285 | 1.0000 | 0.0000 |
| delta_qv_identity_gate | 4.3846 | 0.0000 | 0.0000 |
| delta_qv_wrong_query | 19.8418 | 1.0000 | 0.0000 |
| delta_qv_oracle_correct | 19.8285 | 1.0000 | 0.0000 |
| delta_qv_oracle_paired | 9.8810 | 0.5000 | 0.0000 |
| delta_qv_oracle_correct_address_paired_payload | 19.8586 | 1.0000 | 0.0000 |
| delta_qv_oracle_paired_address_correct_payload | 9.8508 | 0.5000 | 0.0000 |
| logit_bias_oracle_correct | 50.5592 | 0.9643 | 0.0000 |
| logit_bias_oracle_paired | -21.4382 | 0.0000 | 0.5000 |
| logit_bias_correct_address_paired_payload | 5.6985 | 0.5000 | 0.5000 |
| logit_bias_paired_address_correct_payload | 23.4225 | 0.4643 | 0.0000 |
| payload_probe_oracle_correct | 15.0615 | 1.0000 | 0.0000 |
| payload_probe_oracle_paired | -7.1265 | 0.0179 | 0.5000 |
| payload_probe_correct_address_paired_payload | 1.3978 | 0.5000 | 0.5000 |
| payload_probe_paired_address_correct_payload | 6.5372 | 0.5179 | 0.0000 |
| lm_head_lora_oracle_correct | 23.7709 | 1.0000 | 0.0000 |
| lm_head_lora_oracle_paired | -8.8039 | 0.0179 | 0.5000 |
| lm_head_lora_correct_address_paired_payload | 3.5223 | 0.5000 | 0.5000 |
| lm_head_lora_paired_address_correct_payload | 11.4448 | 0.5179 | 0.0000 |

## Address Diagnostics

- `correct_address_rank`: `1.4821`
- `paired_negative_rank`: `0.7143`
- `address_margin`: `0.0059`
- `correct_vs_paired_score_margin`: `0.0587`

- `delta_qv_margin_advantage_vs_wrong_query`: `-0.0292`

## Stage 2 Binding Summary

| mode | answer_nll | top1_correct | top10 |
| --- | ---: | ---: | ---: |
| no_memory | 17.1618 | 0.0000 | 0.0000 |
| delta_qv | 0.0014 | 1.0000 | 1.0000 |
| payload_probe | 0.0300 | 1.0000 | 1.0000 |
| logit_bias | 0.2033 | 0.9643 | 1.0000 |
| lm_head_lora | 0.0048 | 1.0000 | 1.0000 |
| oracle_logit_answer_embedding | 0.7927 | 0.9643 | 0.9821 |

- `oracle_channel_pass`: `True`
- `payload_probe_layer_strategy`: `first_layer`

## Interpretation

This experiment trains only the Delta Memory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
