# Mneme Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `16`
- `lr`: `0.003`
- `task_suite`: `address_token_binding`
- `train_samples`: `4`
- `eval_samples`: `4`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `1`
- `address_margin_weight`: `0.0`
- `address_margin`: `0.1`
- `address_score_scale`: `16.0`
- `oracle_contrastive_weight`: `0.0`
- `identity_gate_beta`: `64.0`
- `identity_gate_tau`: `0.01`
- `oracle_span_writer`: `True`
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_query_to_address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 12.1946 | 12.1946 | 30092.6250 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 14.3684 | 14.3684 | 30253.6875 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 14.3684 | 14.3684 | 30253.6875 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 12.1672 | 3.5718 | 463.9375 | 0.8125 | 295.4765 | 36.8350 | 0.2710 | 1.0000 |
| delta_qv_zero | 12.1946 | 12.1371 | 28799.8125 | 0.2500 | 1.1844 | 0.6191 | 0.2695 | 1.0000 |
| delta_qv_random | 12.2000 | 12.1606 | 27913.3125 | 0.1250 | 23.0601 | 8.2851 | 0.2728 | 1.0000 |
| delta_qv_shuffled | 12.1672 | 3.5718 | 463.9375 | 0.8125 | 295.4765 | 36.8350 | 0.2710 | 1.0000 |
| delta_qv_wrong_layer | 12.4191 | 5.1430 | 3119.8750 | 0.6250 | 316.6914 | 32.0923 | 0.2710 | 1.0000 |
| delta_qv_wrong_query | 12.2162 | 3.5810 | 474.5000 | 0.8125 | 295.4765 | 36.8350 | 0.2710 | 1.0000 |
| delta_qv_identity_gate | 12.3054 | 4.1590 | 774.9375 | 0.7500 | 102.3015 | 12.7521 | 0.2710 | 0.3457 |
| delta_qv_force_gate | 10.9792 | 3.5813 | 283.8750 | 0.8125 | 369.6329 | 136.9475 | 1.0000 | 1.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `8.595407511107624`
- `eval_delta_zero_nll_gap`: `8.565354491584003`
- `eval_delta_random_nll_gap`: `8.588842685334384`
- `eval_delta_shuffled_nll_gap`: `0.0`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_shuffled`
- `no_memory`: mean_delta=`8.6228`, ci95=`[7.2444, 9.7803]`, win_rate=`1.0000`, permutation_p=`0.1129`
- `raw_memory`: mean_delta=`10.7967`, ci95=`[9.1780, 11.8954]`, win_rate=`1.0000`, permutation_p=`0.1129`
- `hidden_retrieval`: mean_delta=`10.7967`, ci95=`[9.1780, 11.8954]`, win_rate=`1.0000`, permutation_p=`0.1129`
- `delta_qv_zero`: mean_delta=`8.5654`, ci95=`[7.0929, 9.7088]`, win_rate=`1.0000`, permutation_p=`0.1129`
- `delta_qv_random`: mean_delta=`8.5888`, ci95=`[7.1896, 9.5222]`, win_rate=`1.0000`, permutation_p=`0.1129`
- `delta_qv_shuffled`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `delta_qv_wrong_layer`: mean_delta=`1.5712`, ci95=`[0.8779, 2.2645]`, win_rate=`1.0000`, permutation_p=`0.1129`
- `delta_qv_wrong_query`: mean_delta=`0.0092`, ci95=`[-0.0376, 0.0560]`, win_rate=`0.5000`, permutation_p=`0.7481`
- `delta_qv_identity_gate`: mean_delta=`0.5873`, ci95=`[-0.1826, 1.2056]`, win_rate=`0.7500`, permutation_p=`0.2329`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | -0.0828 |
| delta_qv | -0.1039 |
| delta_qv_identity_gate | 0.1911 |
| delta_qv_wrong_query | -0.0955 |
| delta_qv_oracle_correct | -0.1039 |
| delta_qv_oracle_paired | -0.0955 |
| delta_qv_oracle_correct_address_paired_payload | -0.0955 |
| delta_qv_oracle_paired_address_correct_payload | -0.1039 |

## Address Diagnostics

- `correct_address_rank`: `1.5000`
- `paired_negative_rank`: `1.5000`
- `address_margin`: `0.0007`
- `correct_vs_paired_score_margin`: `0.0000`

- `delta_qv_margin_advantage_vs_wrong_query`: `-0.0084`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
