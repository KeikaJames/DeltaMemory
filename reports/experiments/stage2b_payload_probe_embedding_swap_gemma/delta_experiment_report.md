# Delta Memory Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `256`
- `lr`: `0.0005`
- `task_suite`: `address_token_binding_single_token`
- `train_samples`: `16`
- `eval_samples`: `16`
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
- `logit_bias_loss_weight`: `0.0`
- `logit_bias_scale`: `50.0`
- `payload_answer_loss_weight`: `1.0`
- `payload_probe_layer_strategy`: `first_layer`
- `payload_embedding_loss_weight`: `0.1`
- `stage2_swap_loss_weight`: `0.1`
- `stage2_swap_margin`: `2.0`
- `stage2_swap_mode`: `payload_probe`
- `eval_injection_modes`: `payload_probe,logit_bias,oracle_logit_answer_embedding`
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_query_to_address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 19.1040 | 19.1040 | 81249.3750 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| logit_bias | 19.1040 | 19.1040 | 81249.3750 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| payload_probe | 12.3062 | 4.6543 | 14.5625 | 0.5625 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 0.0079 | 0.0079 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `diagnosis_skipped`: `True`
- `missing_modes`: `['delta_qv', 'delta_qv_random', 'delta_qv_shuffled', 'delta_qv_zero']`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `no_memory`
- `no_memory`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `logit_bias`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | -0.2227 |
| delta_qv | -0.2227 |
| delta_qv_identity_gate | -0.2227 |
| delta_qv_wrong_query | -0.2227 |
| delta_qv_oracle_correct | -0.2227 |
| delta_qv_oracle_paired | -0.2227 |
| delta_qv_oracle_correct_address_paired_payload | -0.2227 |
| delta_qv_oracle_paired_address_correct_payload | -0.2227 |
| logit_bias_oracle_correct | -0.2227 |
| logit_bias_oracle_paired | -0.2227 |
| logit_bias_correct_address_paired_payload | -0.2227 |
| logit_bias_paired_address_correct_payload | -0.2227 |
| payload_probe_oracle_correct | 4.3555 |
| payload_probe_oracle_paired | -4.3555 |
| payload_probe_correct_address_paired_payload | -4.3555 |
| payload_probe_paired_address_correct_payload | 4.3555 |

## Answer Token Discrimination

| mode | binding_margin | top1_correct | top1_paired |
| --- | ---: | ---: | ---: |
| no_memory | -0.2227 | 0.0000 | 0.0000 |
| delta_qv | -0.2227 | 0.0000 | 0.0000 |
| delta_qv_identity_gate | -0.2227 | 0.0000 | 0.0000 |
| delta_qv_wrong_query | -0.2227 | 0.0000 | 0.0000 |
| delta_qv_oracle_correct | -0.2227 | 0.0000 | 0.0000 |
| delta_qv_oracle_paired | -0.2227 | 0.0000 | 0.0000 |
| delta_qv_oracle_correct_address_paired_payload | -0.2227 | 0.0000 | 0.0000 |
| delta_qv_oracle_paired_address_correct_payload | -0.2227 | 0.0000 | 0.0000 |
| logit_bias_oracle_correct | -0.2227 | 0.0000 | 0.0000 |
| logit_bias_oracle_paired | -0.2227 | 0.0000 | 0.0000 |
| logit_bias_correct_address_paired_payload | -0.2227 | 0.0000 | 0.0000 |
| logit_bias_paired_address_correct_payload | -0.2227 | 0.0000 | 0.0000 |
| payload_probe_oracle_correct | 4.3555 | 0.3750 | 0.0625 |
| payload_probe_oracle_paired | -4.3555 | 0.0625 | 0.3750 |
| payload_probe_correct_address_paired_payload | -4.3555 | 0.0625 | 0.3750 |
| payload_probe_paired_address_correct_payload | 4.3555 | 0.3750 | 0.0625 |

## Address Diagnostics

- `correct_address_rank`: `1.4375`
- `paired_negative_rank`: `1.5625`
- `address_margin`: `0.0055`
- `correct_vs_paired_score_margin`: `0.0004`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0000`

## Stage 2 Binding Summary

| mode | answer_nll | top1_correct | top10 |
| --- | ---: | ---: | ---: |
| no_memory | 19.1040 | 0.0000 | 0.0000 |
| payload_probe | 4.6543 | 0.3750 | 0.5625 |
| logit_bias | 19.1040 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 0.0079 | 1.0000 | 1.0000 |

- `oracle_channel_pass`: `True`
- `payload_probe_layer_strategy`: `first_layer`

## Interpretation

This experiment trains only the Delta Memory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
