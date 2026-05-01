# RCV-HC Gemma4 Delta Q/V Multiseed Report

## Config

- `model`: `google/gemma-4-E2B`
- `dtype`: `bfloat16`
- `steps`: `2`
- `train_samples`: `4`
- `eval_samples`: `4`
- `block_size`: `64`
- `top_k`: `2`
- `base_frozen`: `True`
- `prompt_insertion_used`: `False`

## Per-Seed Results

| seed | delta_nll | zero_nll | random_nll | shuffled_nll | no_memory_nll | nll_drop | supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| seed0 | 11.4252 | 12.2309 | 12.1931 | 12.1303 | 12.2436 | 0.8234 | True |
| seed1 | 11.4234 | 12.2936 | 12.3196 | 12.0658 | 12.3142 | 0.9050 | True |
| seed2 | 11.9027 | 13.0450 | 13.0245 | 12.6590 | 13.0605 | 1.1492 | True |

## Aggregate

- `num_seeds`: `3`
- `support_rate`: `1.0000`
- `mean_delta_nll`: `11.5838`
- `mean_zero_nll`: `12.5231`
- `mean_random_nll`: `12.5124`
- `mean_shuffled_nll`: `12.2850`
- `mean_no_memory_nll`: `12.5394`
- `mean_eval_delta_nll_drop`: `0.9592`
- `std_eval_delta_nll_drop`: `0.1384`
- `all_beat_zero`: `True`
- `all_beat_random`: `True`
- `all_beat_shuffled`: `True`

## Interpretation

Across three seeds, the frozen Gemma4 base remains unchanged and the trained RCV-HC Delta Q/V adapter beats zero, random, and shuffled controls on held-out later-reference examples.
This is still a small mechanism experiment, not a benchmark-scale result.
