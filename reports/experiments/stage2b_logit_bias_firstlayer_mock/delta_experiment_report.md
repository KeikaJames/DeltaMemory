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
- `logit_bias_loss_weight`: `1.0`
- `logit_bias_scale`: `1.0`
- `payload_answer_loss_weight`: `0.0`
- `payload_probe_layer_strategy`: `first_layer`
- `payload_embedding_loss_weight`: `0.0`
- `stage2_swap_loss_weight`: `0.1`
- `stage2_swap_margin`: `2.0`
- `stage2_swap_mode`: `logit_bias`
- `eval_injection_modes`: `logit_bias,payload_probe`
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_query_to_address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 8.5316 | 8.5316 | 2174.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| logit_bias | 8.5316 | 8.5317 | 2174.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| payload_probe | 8.5625 | 8.5618 | 2512.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `diagnosis_skipped`: `True`
- `missing_modes`: `['delta_qv', 'delta_qv_random', 'delta_qv_shuffled', 'delta_qv_zero']`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `no_memory`
- `no_memory`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `logit_bias`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`

## Stage 2 Binding Summary

| mode | answer_nll | top1_correct | top10 |
| --- | ---: | ---: | ---: |
| no_memory | 8.5316 | 0.0000 | 0.0000 |
| logit_bias | 8.5317 | 0.0000 | 0.0000 |
| payload_probe | 8.5618 | 0.0000 | 0.0000 |

- `oracle_channel_pass`: `False`
- `payload_probe_layer_strategy`: `first_layer`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
