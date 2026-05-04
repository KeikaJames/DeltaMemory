# Mneme Q/V Multi-Example Experiment

## Config

- `model`: `mock-gemma`
- `device`: `cpu`
- `dtype`: `float32`
- `steps`: `2`
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
- `payload_probe_layer_strategy`: `first_layer`
- `payload_embedding_loss_weight`: `0.1`
- `stage2_swap_loss_weight`: `0.1`
- `stage2_swap_margin`: `2.0`
- `stage2_swap_mode`: `payload_probe`
- `eval_injection_modes`: `payload_probe,logit_bias,oracle_logit_answer_embedding`
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_query_to_address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 8.4919 | 8.4919 | 2062.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| logit_bias | 8.4919 | 8.4919 | 2062.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| payload_probe | 8.3623 | 8.3347 | 1882.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 8.1310 | 8.1310 | 1146.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

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
| no_memory | 0.0398 |
| delta_qv | 0.0413 |
| delta_qv_identity_gate | 0.0403 |
| delta_qv_wrong_query | 0.0387 |
| delta_qv_oracle_correct | 0.0413 |
| delta_qv_oracle_paired | 0.0387 |
| delta_qv_oracle_correct_address_paired_payload | 0.0387 |
| delta_qv_oracle_paired_address_correct_payload | 0.0413 |
| logit_bias_oracle_correct | 0.0398 |
| logit_bias_oracle_paired | 0.0398 |
| logit_bias_correct_address_paired_payload | 0.0398 |
| logit_bias_paired_address_correct_payload | 0.0398 |
| payload_probe_oracle_correct | -0.0202 |
| payload_probe_oracle_paired | 0.0202 |
| payload_probe_correct_address_paired_payload | 0.0202 |
| payload_probe_paired_address_correct_payload | -0.0202 |

## Answer Token Discrimination

| mode | binding_margin | top1_correct | top1_paired |
| --- | ---: | ---: | ---: |
| no_memory | 0.0398 | 0.0000 | 0.0000 |
| delta_qv | 0.0413 | 0.0000 | 0.0000 |
| delta_qv_identity_gate | 0.0403 | 0.0000 | 0.0000 |
| delta_qv_wrong_query | 0.0387 | 0.0000 | 0.0000 |
| delta_qv_oracle_correct | 0.0413 | 0.0000 | 0.0000 |
| delta_qv_oracle_paired | 0.0387 | 0.0000 | 0.0000 |
| delta_qv_oracle_correct_address_paired_payload | 0.0387 | 0.0000 | 0.0000 |
| delta_qv_oracle_paired_address_correct_payload | 0.0413 | 0.0000 | 0.0000 |
| logit_bias_oracle_correct | 0.0398 | 0.0000 | 0.0000 |
| logit_bias_oracle_paired | 0.0398 | 0.0000 | 0.0000 |
| logit_bias_correct_address_paired_payload | 0.0398 | 0.0000 | 0.0000 |
| logit_bias_paired_address_correct_payload | 0.0398 | 0.0000 | 0.0000 |
| payload_probe_oracle_correct | -0.0202 | 0.0000 | 0.0000 |
| payload_probe_oracle_paired | 0.0202 | 0.0000 | 0.0000 |
| payload_probe_correct_address_paired_payload | 0.0202 | 0.0000 | 0.0000 |
| payload_probe_paired_address_correct_payload | -0.0202 | 0.0000 | 0.0000 |

## Address Diagnostics

- `correct_address_rank`: `5.0000`
- `paired_negative_rank`: `1.0000`
- `address_margin`: `0.0399`
- `correct_vs_paired_score_margin`: `-0.2164`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0026`

## Stage 2 Binding Summary

| mode | answer_nll | top1_correct | top10 |
| --- | ---: | ---: | ---: |
| no_memory | 8.4919 | 0.0000 | 0.0000 |
| payload_probe | 8.3347 | 0.0000 | 0.0000 |
| logit_bias | 8.4919 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 8.1310 | 0.0000 | 0.0000 |

- `oracle_channel_pass`: `False`
- `payload_probe_layer_strategy`: `first_layer`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
