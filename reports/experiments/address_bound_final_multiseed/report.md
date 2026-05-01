# Address-Bound Delta Memory Final Multiseed Report

This run closes the current Address-Bound Delta Memory experiment plan. It uses
the strongest current configuration from the address-token/contrastive pilots
and repeats it across three real Gemma/MPS seeds.

## Configuration

| field | value |
| --- | --- |
| model | `google/gemma-4-E2B` |
| device | `mps` |
| dtype | `bfloat16` |
| task suite | `address_token_binding` |
| seeds | `0, 1, 2` |
| train / eval samples | `8 / 8` per seed |
| steps | `16` |
| layers | `all` |
| memory_dim | `512` |
| top_k | `2` |
| address loss | weight `5.0`, margin `0.05` |
| contrastive loss | weight `1.0`, margin `0.5` |
| identity gate | beta `64.0`, tau `0.01` |
| control gate | require at least `0.05` NLL gap |

## Seed results

| seed | delta_nll | no_memory_nll | shuffled_nll | wrong_query_nll | address_rank | address_margin | shuffled_gap | wrong_query_gap | supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 3.0843 | 12.1370 | 3.0847 | 3.0924 | 4.50 | 0.0017 | 0.0004 | 0.0081 | false |
| 1 | 5.0610 | 12.1616 | 5.0574 | 5.0610 | 4.50 | 0.0005 | -0.0036 | 0.0000 | false |
| 2 | 4.1443 | 12.0347 | 4.1567 | 4.1443 | 5.75 | 0.0036 | 0.0124 | 0.0000 | false |

## Aggregate

| metric | value |
| --- | ---: |
| support_rate | 0.0000 |
| mean_delta_nll | 4.0965 |
| mean_no_memory_nll | 12.1111 |
| mean_zero_nll | 12.2507 |
| mean_random_nll | 12.1342 |
| mean_shuffled_nll | 4.0996 |
| mean_wrong_query_nll | 4.0992 |
| mean_identity_gate_nll | 6.1768 |
| mean_hidden_retrieval_nll | 14.2313 |
| mean_zero_gap | 8.1541 |
| mean_random_gap | 8.0377 |
| mean_shuffled_gap | 0.0031 |
| mean_wrong_query_gap | 0.0027 |
| mean_address_rank | 4.9167 |
| mean_paired_negative_rank | 4.7500 |
| mean_address_margin | 0.0020 |
| mean_correct_vs_paired_score_margin | -0.0003 |
| mean_identity_gate | 0.3868 |
| mean_q_delta_norm | 119.3451 |
| mean_v_delta_norm | 30.0290 |

## Interpretation

The memory channel is robust: across all three seeds, Delta Q/V injection
substantially lowers answer NLL relative to no-memory, zero, random, and hidden
retrieval baselines.

The binding claim fails. Shuffled and wrong-query controls remain effectively
tied with correct retrieval under the stricter `0.05` NLL control-margin gate.
The address rank is also poor: the correct address averages rank `4.9167`, while
the paired negative averages rank `4.7500`. This means the model is not using a
reliably selected query-specific address as the causal source of the gain.

The correct conclusion is therefore:

```text
Delta Memory currently demonstrates a strong attention-internal memory channel.
It does not yet demonstrate query-specific address binding as the causal source.
```

Further larger-seed scaling is not warranted until the address selection
mechanism itself becomes identifiable.
