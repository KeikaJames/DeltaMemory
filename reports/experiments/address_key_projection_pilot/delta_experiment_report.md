# Mneme Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `8`
- `lr`: `0.001`
- `task_suite`: `paired_conflict_binding`
- `train_samples`: `8`
- `eval_samples`: `8`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `address_margin_weight`: `0.0`
- `address_margin`: `0.1`
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
| raw_memory | 13.9044 | 13.9044 | 56189.8125 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 13.9044 | 13.9044 | 56189.8125 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 12.3239 | 4.8159 | 1884.5000 | 0.7500 | 45.8517 | 16.5814 | 0.2986 | 0.0000 |
| delta_qv_zero | 11.6111 | 11.4274 | 56144.2500 | 0.2812 | 0.1658 | 0.0821 | 0.2695 | 0.0000 |
| delta_qv_random | 11.7563 | 11.9364 | 53038.0938 | 0.3438 | 18.0250 | 6.3894 | 0.2691 | 0.0000 |
| delta_qv_shuffled | 12.2488 | 4.8220 | 1912.6875 | 0.7500 | 45.4834 | 16.0715 | 0.2949 | 0.0000 |
| delta_qv_wrong_layer | 11.9693 | 6.1500 | 7482.4688 | 0.6250 | 46.4423 | 16.5596 | 0.2986 | 0.0000 |
| delta_qv_wrong_query | 12.1694 | 4.8162 | 1816.3750 | 0.7188 | 45.8517 | 16.5814 | 0.2986 | 0.0000 |
| delta_qv_force_gate | 12.5888 | 6.0098 | 7410.4688 | 0.5625 | 113.5601 | 44.6364 | 1.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `7.5080635184422135`
- `eval_delta_zero_nll_gap`: `6.611546949483454`
- `eval_delta_random_nll_gap`: `7.120507285930216`
- `eval_delta_shuffled_nll_gap`: `0.006157150957733393`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `False`
- `mechanism_supported_on_eval`: `False`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_wrong_query`
- `delta_qv_random`: mean_delta=`7.1205`, ci95=`[6.6356, 7.6728]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_shuffled`: mean_delta=`0.0062`, ci95=`[-0.0375, 0.0631]`, win_rate=`0.5000`, permutation_p=`0.9005`
- `delta_qv_wrong_layer`: mean_delta=`1.3342`, ci95=`[1.1546, 1.5191]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_wrong_query`: mean_delta=`0.0004`, ci95=`[-0.0421, 0.0495]`, win_rate=`0.3750`, permutation_p=`0.9655`
- `delta_qv_zero`: mean_delta=`6.6115`, ci95=`[6.1485, 7.0235]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `hidden_retrieval`: mean_delta=`9.0885`, ci95=`[8.0832, 10.0560]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `no_memory`: mean_delta=`6.7953`, ci95=`[6.2960, 7.2282]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `raw_memory`: mean_delta=`9.0885`, ci95=`[8.0832, 10.0560]`, win_rate=`1.0000`, permutation_p=`0.0065`

## Conflict Margins

| mode | foreign_minus_correct_nll |
| --- | ---: |
| delta_qv | 0.1916 |
| delta_qv_wrong_query | 0.1849 |
| no_memory | 0.2126 |

## Address Diagnostics

- `correct_address_rank`: `5.2500`
- `paired_negative_rank`: `6.2500`
- `address_margin`: `0.0007`
- `correct_vs_paired_score_margin`: `0.0003`

- `delta_qv_margin_advantage_vs_wrong_query`: `0.0067`

## Interpretation

This experiment trains only the Mneme writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
