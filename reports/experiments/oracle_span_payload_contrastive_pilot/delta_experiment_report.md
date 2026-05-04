# Mneme Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `64`
- `lr`: `0.005`
- `task_suite`: `address_token_binding`
- `train_samples`: `4`
- `eval_samples`: `4`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `1`
- `address_margin_weight`: `0.0`
- `address_margin`: `0.1`
- `address_score_scale`: `16.0`
- `oracle_contrastive_weight`: `5.0`
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
| delta_qv | 12.1672 | 4.3541 | 3012.7500 | 0.6875 | 133.1368 | 325.9275 | 0.4320 | 1.0000 |
| delta_qv_zero | 12.1946 | 11.9752 | 30192.8750 | 0.1875 | 1.9390 | 5.0491 | 0.3477 | 1.0000 |
| delta_qv_random | 12.2000 | 10.8354 | 28668.4375 | 0.3125 | 26.6927 | 15.1410 | 0.3443 | 1.0000 |
| delta_qv_shuffled | 12.1672 | 4.3541 | 3012.7500 | 0.6875 | 133.1368 | 325.9275 | 0.4320 | 1.0000 |
| delta_qv_wrong_layer | 12.4191 | 7.6348 | 21350.2500 | 0.2500 | 107.7189 | 339.1318 | 0.4320 | 1.0000 |
| delta_qv_wrong_query | 12.2162 | 4.3522 | 3038.0000 | 0.6875 | 133.1368 | 325.9275 | 0.4320 | 1.0000 |
| delta_qv_identity_gate | 12.3054 | 6.0785 | 12640.8750 | 0.6875 | 46.1895 | 112.8563 | 0.4320 | 0.3457 |
| delta_qv_force_gate | 10.9792 | 5.8853 | 11275.6875 | 0.7500 | 937.0084 | 402.3262 | 1.0000 | 1.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `7.813078220933676`
- `eval_delta_zero_nll_gap`: `7.621118690818548`
- `eval_delta_random_nll_gap`: `6.481277044862509`
- `eval_delta_shuffled_nll_gap`: `0.0`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_wrong_query`
- `no_memory`: mean_delta=`7.8405`, ci95=`[7.3336, 8.6834]`, win_rate=`1.0000`, permutation_p=`0.1129`
- `raw_memory`: mean_delta=`10.0143`, ci95=`[9.2672, 10.7636]`, win_rate=`1.0000`, permutation_p=`0.1129`
- `hidden_retrieval`: mean_delta=`10.0143`, ci95=`[9.2672, 10.7636]`, win_rate=`1.0000`, permutation_p=`0.1129`
- `delta_qv_zero`: mean_delta=`7.6211`, ci95=`[7.2068, 8.3899]`, win_rate=`1.0000`, permutation_p=`0.1129`
- `delta_qv_random`: mean_delta=`6.4813`, ci95=`[5.9693, 7.3054]`, win_rate=`1.0000`, permutation_p=`0.1129`
- `delta_qv_shuffled`: mean_delta=`0.0000`, ci95=`[0.0000, 0.0000]`, win_rate=`0.0000`, permutation_p=`1.0000`
- `delta_qv_wrong_layer`: mean_delta=`3.2807`, ci95=`[3.1524, 3.3707]`, win_rate=`1.0000`, permutation_p=`0.1129`
- `delta_qv_wrong_query`: mean_delta=`-0.0019`, ci95=`[-0.0110, 0.0058]`, win_rate=`0.5000`, permutation_p=`0.8746`
- `delta_qv_identity_gate`: mean_delta=`1.7244`, ci95=`[1.4423, 1.9974]`, win_rate=`1.0000`, permutation_p=`0.1129`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | -0.0828 |
| delta_qv | 0.0285 |
| delta_qv_identity_gate | -0.0035 |
| delta_qv_wrong_query | 0.0018 |
| delta_qv_oracle_correct | 0.0285 |
| delta_qv_oracle_paired | 0.0018 |
| delta_qv_oracle_correct_address_paired_payload | 0.0018 |
| delta_qv_oracle_paired_address_correct_payload | 0.0285 |

## Address Diagnostics

- `correct_address_rank`: `1.5000`
- `paired_negative_rank`: `1.5000`
- `address_margin`: `0.0007`
- `correct_vs_paired_score_margin`: `0.0000`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0267`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
