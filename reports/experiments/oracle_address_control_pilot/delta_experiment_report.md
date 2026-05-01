# Delta Memory Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `16`
- `lr`: `0.001`
- `task_suite`: `address_token_binding`
- `train_samples`: `8`
- `eval_samples`: `8`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `address_margin_weight`: `5.0`
- `address_margin`: `0.05`
- `address_score_scale`: `32.0`
- `identity_gate_beta`: `64.0`
- `identity_gate_tau`: `0.01`
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_query_to_address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 12.1370 | 12.1370 | 31464.2812 | 0.3438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 14.2806 | 14.2786 | 31592.7812 | 0.3438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 14.2806 | 14.2786 | 31592.7812 | 0.3438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 11.6696 | 5.1881 | 2000.3438 | 0.5625 | 67.1045 | 25.6578 | 0.2982 | 1.0000 |
| delta_qv_zero | 12.1370 | 12.2104 | 32302.5312 | 0.3125 | 0.2366 | 0.0951 | 0.2695 | 1.0000 |
| delta_qv_random | 12.0234 | 12.2649 | 31695.1250 | 0.3125 | 18.4198 | 6.5392 | 0.2698 | 1.0000 |
| delta_qv_shuffled | 11.6971 | 5.1931 | 2019.8438 | 0.5938 | 67.1023 | 25.6314 | 0.2981 | 1.0000 |
| delta_qv_wrong_layer | 11.7645 | 6.2116 | 2603.0938 | 0.4375 | 68.7804 | 25.6788 | 0.2982 | 1.0000 |
| delta_qv_wrong_query | 11.7166 | 5.1709 | 2024.3438 | 0.5938 | 67.1045 | 25.6578 | 0.2982 | 1.0000 |
| delta_qv_identity_gate | 12.0162 | 7.4669 | 9442.5312 | 0.5000 | 25.3346 | 9.6616 | 0.2982 | 0.3759 |
| delta_qv_force_gate | 10.4265 | 5.2246 | 6614.5312 | 0.7500 | 195.1760 | 69.2077 | 1.0000 | 1.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `6.481580700725317`
- `eval_delta_zero_nll_gap`: `7.0223369263112545`
- `eval_delta_random_nll_gap`: `7.076830696314573`
- `eval_delta_shuffled_nll_gap`: `0.005001685582101345`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_wrong_query`
- `no_memory`: mean_delta=`6.9489`, ci95=`[6.2311, 7.6760]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `raw_memory`: mean_delta=`9.0905`, ci95=`[8.1576, 9.9777]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `hidden_retrieval`: mean_delta=`9.0905`, ci95=`[8.1576, 9.9777]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_zero`: mean_delta=`7.0223`, ci95=`[6.2326, 7.8199]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_random`: mean_delta=`7.0768`, ci95=`[6.4396, 7.7849]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_shuffled`: mean_delta=`0.0050`, ci95=`[-0.0323, 0.0514]`, win_rate=`0.3750`, permutation_p=`0.8611`
- `delta_qv_wrong_layer`: mean_delta=`1.0235`, ci95=`[0.8127, 1.2156]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_wrong_query`: mean_delta=`-0.0172`, ci95=`[-0.0422, 0.0012]`, win_rate=`0.1250`, permutation_p=`0.2259`
- `delta_qv_identity_gate`: mean_delta=`2.2789`, ci95=`[1.9722, 2.5125]`, win_rate=`1.0000`, permutation_p=`0.0065`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | -0.1068 |
| delta_qv | 0.0804 |
| delta_qv_identity_gate | -0.0915 |
| delta_qv_wrong_query | 0.0810 |
| delta_qv_oracle_correct | 0.0923 |
| delta_qv_oracle_paired | 0.0756 |

## Address Diagnostics

- `correct_address_rank`: `4.5000`
- `paired_negative_rank`: `4.5000`
- `address_margin`: `0.0020`
- `correct_vs_paired_score_margin`: `0.0000`

- `delta_qv_margin_advantage_vs_wrong_query`: `-0.0005`

## Interpretation

This experiment trains only the Delta Memory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
