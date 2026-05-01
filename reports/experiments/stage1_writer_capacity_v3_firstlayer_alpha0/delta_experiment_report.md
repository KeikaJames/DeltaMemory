# Delta Memory Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `64`
- `lr`: `0.003`
- `task_suite`: `address_token_binding_single_token`
- `train_samples`: `16`
- `eval_samples`: `16`
- `block_size`: `64`
- `memory_dim`: `256`
- `top_k`: `2`
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
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_query_to_address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 19.1040 | 19.1040 | 81249.3750 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 19.9946 | 19.9763 | 81316.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 19.9946 | 19.9763 | 81316.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| retrieved_attention | 19.9947 | 19.9609 | 81053.9375 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 19.1040 | 19.1040 | 81249.3750 | 0.0000 | 0.0000 | 0.0000 | 0.2695 | 1.0000 |
| delta_qv_zero | 19.1040 | 19.1040 | 81249.3750 | 0.0000 | 0.0000 | 0.0000 | 0.2695 | 1.0000 |
| delta_qv_random | 19.1040 | 19.1040 | 81249.3750 | 0.0000 | 0.0000 | 0.0000 | 0.2695 | 1.0000 |
| delta_qv_shuffled | 19.1040 | 19.1040 | 81249.3750 | 0.0000 | 0.0000 | 0.0000 | 0.2695 | 1.0000 |
| delta_qv_wrong_layer | 19.1040 | 19.1040 | 81249.3750 | 0.0000 | 0.0000 | 0.0000 | 0.2695 | 1.0000 |
| delta_qv_wrong_query | 19.1040 | 19.1040 | 81249.3750 | 0.0000 | 0.0000 | 0.0000 | 0.2695 | 1.0000 |
| delta_qv_identity_gate | 19.1040 | 19.1040 | 81249.3750 | 0.0000 | 0.0000 | 0.0000 | 0.2695 | 0.3457 |
| delta_qv_force_gate | 19.1040 | 19.1040 | 81249.3750 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| logit_bias | 19.1040 | 19.1040 | 81249.3750 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 0.0079 | 0.0079 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `0.0`
- `eval_delta_zero_nll_gap`: `0.0`
- `eval_delta_random_nll_gap`: `0.0`
- `eval_delta_shuffled_nll_gap`: `0.0`
- `eval_delta_beats_zero`: `False`
- `eval_delta_beats_random`: `False`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `no_memory`
- `no_memory`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `raw_memory`: mean_delta=`0.8723`, ci95=`[0.7522, 0.9911]`, win_rate=`1.0000`, permutation_p=`0.0005`
- `hidden_retrieval`: mean_delta=`0.8723`, ci95=`[0.7522, 0.9911]`, win_rate=`1.0000`, permutation_p=`0.0005`
- `retrieved_attention`: mean_delta=`0.8569`, ci95=`[0.7211, 0.9956]`, win_rate=`1.0000`, permutation_p=`0.0005`
- `logit_bias`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `delta_qv_zero`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `delta_qv_random`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `delta_qv_shuffled`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `delta_qv_wrong_layer`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `delta_qv_wrong_query`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `delta_qv_identity_gate`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`

## Interpretation

This experiment trains only the Delta Memory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
