# Mneme Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `64`
- `lr`: `0.01`
- `task_suite`: `address_token_binding`
- `train_samples`: `8`
- `eval_samples`: `8`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `address_margin_weight`: `20.0`
- `address_margin`: `0.05`
- `address_score_scale`: `64.0`
- `oracle_contrastive_weight`: `5.0`
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
| raw_memory | 14.2806 | 14.2795 | 31614.1875 | 0.3438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 14.2806 | 14.2795 | 31614.1875 | 0.3438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 11.6696 | 2.9874 | 62.7500 | 0.7812 | 2558.0676 | 43.5793 | 0.1865 | 1.0000 |
| delta_qv_zero | 12.1370 | 10.4983 | 13983.5312 | 0.3125 | 103.5785 | 2.3837 | 0.2031 | 1.0000 |
| delta_qv_random | 12.0234 | 9.1654 | 11465.8125 | 0.3750 | 148.5389 | 20.1049 | 0.2632 | 1.0000 |
| delta_qv_shuffled | 11.6971 | 2.9888 | 67.0312 | 0.7812 | 2557.8329 | 43.5865 | 0.1864 | 1.0000 |
| delta_qv_wrong_layer | 11.7645 | 4.7514 | 2938.0938 | 0.7188 | 2602.1653 | 29.8467 | 0.1865 | 1.0000 |
| delta_qv_wrong_query | 11.7166 | 2.9911 | 62.4688 | 0.7812 | 2558.0676 | 43.5793 | 0.1865 | 1.0000 |
| delta_qv_identity_gate | 12.0162 | 4.4096 | 1356.2812 | 0.7500 | 918.6808 | 17.0171 | 0.1865 | 0.3619 |
| delta_qv_force_gate | 10.4265 | 3.5583 | 58.1875 | 0.7188 | 3160.0095 | 564.7304 | 1.0000 | 1.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `8.682209433056414`
- `eval_delta_zero_nll_gap`: `7.510856674052775`
- `eval_delta_random_nll_gap`: `6.177931217011064`
- `eval_delta_shuffled_nll_gap`: `0.0013651726767420769`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_shuffled`
- `no_memory`: mean_delta=`9.1495`, ci95=`[8.1997, 10.0229]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `raw_memory`: mean_delta=`11.2920`, ci95=`[10.1346, 12.3304]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `hidden_retrieval`: mean_delta=`11.2920`, ci95=`[10.1346, 12.3304]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_zero`: mean_delta=`7.5109`, ci95=`[6.7929, 8.1000]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_random`: mean_delta=`6.1779`, ci95=`[4.8275, 7.4979]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_shuffled`: mean_delta=`0.0014`, ci95=`[-0.0180, 0.0203]`, win_rate=`0.6250`, permutation_p=`0.8861`
- `delta_qv_wrong_layer`: mean_delta=`1.7640`, ci95=`[1.4174, 2.0941]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_wrong_query`: mean_delta=`0.0037`, ci95=`[-0.0174, 0.0288]`, win_rate=`0.2500`, permutation_p=`0.8731`
- `delta_qv_identity_gate`: mean_delta=`1.4221`, ci95=`[1.1759, 1.6595]`, win_rate=`1.0000`, permutation_p=`0.0065`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| no_memory | -0.1068 |
| delta_qv | 0.0202 |
| delta_qv_identity_gate | 0.0093 |
| delta_qv_wrong_query | 0.0042 |
| delta_qv_oracle_correct | 0.0127 |
| delta_qv_oracle_paired | 0.0305 |

## Address Diagnostics

- `correct_address_rank`: `4.5000`
- `paired_negative_rank`: `4.5000`
- `address_margin`: `0.0000`
- `correct_vs_paired_score_margin`: `0.0000`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0160`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
