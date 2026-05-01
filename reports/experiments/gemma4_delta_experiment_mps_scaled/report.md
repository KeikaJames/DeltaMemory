# RCV-HC Gemma4 Delta Q/V MPS Scaled Report

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `backend`: `Apple Metal / PyTorch MPS`
- `dtype`: `bfloat16`
- `steps`: `8`
- `train_samples`: `8`
- `eval_samples`: `8`
- `seeds`: `[0, 1, 2]`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `base_frozen`: `True`
- `prompt_insertion_used`: `False`
- `controls`: `['delta_qv_zero', 'delta_qv_random', 'delta_qv_shuffled']`

## Per-Seed Held-Out Results

| seed | delta_nll | zero_nll | random_nll | shuffled_nll | no_memory_nll | nll_drop | supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8.2896 | 11.9196 | 11.8848 | 10.2355 | 11.9019 | 3.6310 | True |
| 1 | 7.4858 | 12.2266 | 12.2238 | 10.7552 | 12.2138 | 4.7704 | True |
| 2 | 9.2081 | 12.8525 | 12.8249 | 10.0721 | 12.8407 | 3.6309 | True |

## Aggregate

- `num_seeds`: `3`
- `support_rate`: `1.0000`
- `mean_delta_nll`: `8.3279`
- `mean_zero_nll`: `12.3329`
- `mean_random_nll`: `12.3112`
- `mean_shuffled_nll`: `10.3543`
- `mean_no_memory_nll`: `12.3188`
- `mean_eval_delta_nll_drop`: `4.0107`
- `std_eval_delta_nll_drop`: `0.5371`
- `mean_delta_rank`: `13084.6250`
- `mean_zero_rank`: `40171.0312`
- `mean_random_rank`: `40217.6042`
- `mean_shuffled_rank`: `25773.6458`
- `mean_no_memory_rank`: `40113.6562`
- `mean_q_delta_norm`: `17.9191`
- `mean_v_delta_norm`: `7.3799`
- `mean_gate_v`: `0.7271`
- `all_beat_zero`: `True`
- `all_beat_random`: `True`
- `all_beat_shuffled`: `True`

## Diagnosis

- `mechanism_signal_status`: `supported_on_scaled_mps_probe`
- `engineering_status`: `mps_path_working`
- `claim_scope`: `small held-out mechanism experiment; not benchmark-scale proof`

## Interpretation

This scaled run uses the M4 Max through PyTorch MPS/Metal, not CPU. The frozen Gemma4 base remains unchanged and the trained RCV-HC Delta Q/V adapter beats zero, random, and shuffled controls on held-out later-reference examples for all three seeds.

This supports the attention-memory mechanism at small experimental scale. It is not yet a large benchmark result, and it does not use prompt insertion or retrieved source text as the answer path.
