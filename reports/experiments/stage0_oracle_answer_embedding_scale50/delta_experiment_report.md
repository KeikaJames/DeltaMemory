# Delta Memory Q/V Multi-Example Experiment

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `dtype`: `bfloat16`
- `steps`: `4`
- `lr`: `0.001`
- `task_suite`: `address_token_binding_single_token`
- `train_samples`: `8`
- `eval_samples`: `8`
- `block_size`: `64`
- `memory_dim`: `256`
- `top_k`: `2`
- `address_margin_weight`: `0.0`
- `address_margin`: `0.1`
- `address_score_scale`: `16.0`
- `oracle_contrastive_weight`: `0.0`
- `identity_gate_beta`: `64.0`
- `identity_gate_tau`: `0.01`
- `oracle_span_writer`: `False`
- `logit_bias_loss_weight`: `0.0`
- `logit_bias_scale`: `50.0`
- `payload_answer_loss_weight`: `0.0`
- `control_margin_min`: `0.05`
- `layer_ids`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`
- `trainable_base_params`: `0`
- `prompt_insertion_used`: `False`
- `retrieval_key`: `address_query_to_address_key`

## Eval Aggregate

| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 18.6584 | 18.6584 | 72719.6250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 19.5645 | 19.5645 | 72775.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hidden_retrieval | 19.5645 | 19.5645 | 72775.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| retrieved_attention | 19.5570 | 19.5570 | 72880.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 18.5668 | 12.2010 | 15348.1250 | 0.0000 | 16.8567 | 7.1355 | 0.2793 | 1.0000 |
| delta_qv_zero | 18.6584 | 18.5929 | 72140.7500 | 0.0000 | 0.1056 | 0.0437 | 0.2695 | 1.0000 |
| delta_qv_random | 18.4384 | 18.4620 | 74224.7500 | 0.0000 | 12.6907 | 4.5190 | 0.2694 | 1.0000 |
| delta_qv_shuffled | 18.2775 | 16.7997 | 59818.1250 | 0.0000 | 15.0796 | 5.8116 | 0.2742 | 1.0000 |
| delta_qv_wrong_layer | 18.5030 | 15.1126 | 37883.8750 | 0.0000 | 15.9646 | 6.7301 | 0.2793 | 1.0000 |
| delta_qv_wrong_query | 18.6198 | 12.3432 | 17130.1250 | 0.0000 | 16.8567 | 7.1355 | 0.2793 | 1.0000 |
| delta_qv_identity_gate | 18.7805 | 14.3591 | 31065.5000 | 0.0000 | 8.0953 | 3.4621 | 0.2793 | 0.4784 |
| delta_qv_force_gate | 19.4227 | 10.5873 | 6427.5000 | 0.0000 | 62.2353 | 24.2457 | 1.0000 | 1.0000 |
| logit_bias | 18.6584 | 18.6584 | 72719.6250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle_logit_answer_embedding | 0.0015 | 0.0015 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Diagnosis

- `control_margin_min`: `0.05`
- `eval_delta_nll_drop`: `6.365760803222656`
- `eval_delta_zero_nll_gap`: `6.391834020614624`
- `eval_delta_random_nll_gap`: `6.260904550552368`
- `eval_delta_shuffled_nll_gap`: `4.598650813102722`
- `eval_delta_beats_zero`: `True`
- `eval_delta_beats_random`: `True`
- `eval_delta_beats_shuffled`: `True`
- `mechanism_supported_on_eval`: `True`

## Paired Statistics

- `strongest_non_prompt_baseline`: `delta_qv_wrong_query`
- `no_memory`: mean_delta=`6.4573`, ci95=`[5.9762, 7.0319]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `raw_memory`: mean_delta=`7.3634`, ci95=`[6.8452, 7.9516]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `hidden_retrieval`: mean_delta=`7.3634`, ci95=`[6.8452, 7.9516]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `retrieved_attention`: mean_delta=`7.3559`, ci95=`[6.8472, 7.9465]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `logit_bias`: mean_delta=`6.4573`, ci95=`[5.9762, 7.0319]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_zero`: mean_delta=`6.3918`, ci95=`[5.9584, 6.9054]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_random`: mean_delta=`6.2609`, ci95=`[5.8093, 6.9080]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_shuffled`: mean_delta=`4.5987`, ci95=`[3.9134, 5.2872]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_wrong_layer`: mean_delta=`2.9116`, ci95=`[2.2454, 3.5192]`, win_rate=`1.0000`, permutation_p=`0.0065`
- `delta_qv_wrong_query`: mean_delta=`0.1422`, ci95=`[-0.2662, 0.6488]`, win_rate=`0.5000`, permutation_p=`0.6097`
- `delta_qv_identity_gate`: mean_delta=`2.1580`, ci95=`[1.7294, 2.5200]`, win_rate=`1.0000`, permutation_p=`0.0065`

## Interpretation

This experiment trains only the Delta Memory writer and Q/V adapter while the base model remains frozen.
A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.
