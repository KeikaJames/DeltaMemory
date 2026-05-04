# Mneme Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `16`
- `lr`: `0.001`
- `task_suite`: `paired_conflict_binding`
- `train_samples`: `8`
- `eval_samples`: `8`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `address_margin_weight`: `5.0`
- `address_margin`: `0.05`
- `identity_gate_beta`: `64.0`
- `identity_gate_tau`: `0.01`
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 11.6111 | 11.6111 | 56127.8438 | 0.2812 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 13.9044 | 13.9061 | 56193.4688 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 13.9044 | 13.9061 | 56193.4688 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 12.3239 | 3.9278 | 619.5625 | 0.7500 | 58.1661 | 34.4080 | 0.3717 | 1.0000 |
| delta_qv_zero | 11.6111 | 11.5306 | 56075.1875 | 0.2812 | 0.2786 | 0.1554 | 0.2695 | 1.0000 |
| delta_qv_random | 11.7563 | 11.8825 | 53150.7188 | 0.3125 | 18.3616 | 6.5598 | 0.2717 | 1.0000 |
| delta_qv_shuffled | 12.2488 | 3.9230 | 632.4688 | 0.7500 | 58.2626 | 34.4107 | 0.3720 | 1.0000 |
| delta_qv_wrong_layer | 11.9693 | 4.8931 | 2225.3750 | 0.7500 | 56.9539 | 33.2966 | 0.3717 | 1.0000 |
| delta_qv_wrong_query | 12.1694 | 3.9226 | 604.4375 | 0.7500 | 58.1661 | 34.4080 | 0.3717 | 1.0000 |
| delta_qv_identity_gate | 12.0176 | 6.5213 | 7746.3125 | 0.7188 | 22.3920 | 13.5307 | 0.3717 | 0.3911 |
| delta_qv_force_gate | 12.5888 | 4.3115 | 385.6562 | 0.7500 | 128.1316 | 61.2371 | 1.0000 | 1.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `8.39606855623424`
- `eval_delta_zero_nll_gap`: `7.602756155654788`
- `eval_delta_random_nll_gap`: `7.9546672981232405`
- `eval_delta_shuffled_nll_gap`: `-0.004835073836147785`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_wrong_query`
- `delta_qv_identity_gate`: mean_delta=`2.5935`, ci95=`[2.2888, 2.9119]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_random`: mean_delta=`7.9547`, ci95=`[7.3933, 8.4474]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_shuffled`: mean_delta=`-0.0048`, ci95=`[-0.0474, 0.0321]`, win_rate=`0.5000`, permutation_p=`0.8591`
- `delta_qv_wrong_layer`: mean_delta=`0.9653`, ci95=`[0.6978, 1.2595]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_wrong_query`: mean_delta=`-0.0052`, ci95=`[-0.0221, 0.0156]`, win_rate=`0.3750`, permutation_p=`0.6402`
- `delta_qv_zero`: mean_delta=`7.6028`, ci95=`[7.1269, 7.9229]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `hidden_retrieval`: mean_delta=`9.9782`, ci95=`[8.9736, 10.9085]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `no_memory`: mean_delta=`7.6833`, ci95=`[7.2506, 7.9781]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `raw_memory`: mean_delta=`9.9782`, ci95=`[8.9736, 10.9085]`, win_rate=`1.0000`, permutation_p=`0.0065`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| delta_qv | 0.0827 |
| delta_qv_identity_gate | 0.6872 |
| delta_qv_wrong_query | 0.0834 |
| no_memory | 0.2126 |

## Address Diagnostics

- `correct_address_rank`: `4.5000`
- `paired_negative_rank`: `4.5000`
- `address_margin`: `0.0013`
- `correct_vs_paired_score_margin`: `-0.0000`

- `delta_qv_margin_advantage_vs_wrong_query`: `-0.0007`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
