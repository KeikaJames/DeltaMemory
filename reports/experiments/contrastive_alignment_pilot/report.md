# Mneme Contrastive Alignment Pilot

## Question

Can contrastive training and shared-memory retrieval force Mneme to use
query-specific memory identity rather than only a generic Delta activation?

This pilot follows the negative paired-conflict result and adds:

- `contrastive_margin_weight=1.0`;
- `contrastive_margin=0.5`;
- correct-vs-paired-foreign answer margin loss during training;
- a second pilot with `shared_memory_retrieval=true`, so retrieval happens from
  a pool containing all paired-conflict memories instead of an isolated sample
  memory.

Both pilots use corrected question-only retrieval queries and no prompt
insertion.

## Results

| setup | delta_qv_nll | no_memory_nll | wrong_query_nll | shuffled_nll | delta_margin | wrong_query_margin | margin_advantage | mechanism_supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| isolated_memory + contrastive | 5.4597 | 12.3341 | 5.4334 | 13.4421 | -0.2131 | -0.2036 | -0.0095 | true |
| shared_memory + contrastive | 5.2961 | 12.3341 | 5.3277 | 5.2929 | -0.1087 | -0.1382 | 0.0295 | false |

Prior non-contrastive isolated paired-conflict baseline:

| setup | delta_qv_nll | no_memory_nll | wrong_query_nll | margin_advantage |
| --- | ---: | ---: | ---: | ---: |
| isolated_memory | 4.6836 | 12.3341 | 4.6609 | 0.0094 |

## Interpretation

Contrastive training alone did not improve held-out margin alignment. It lowered
neither wrong-query effectiveness nor produced a positive correct-vs-foreign
margin advantage.

Shared-memory retrieval moved the margin advantage in the right direction
(`0.0295` vs `0.0094`), but it is still too small and the run failed the existing
mechanism gate because `delta_qv` did not beat the shuffled control. This is not
a publishable positive result; it is a useful ablation showing the current
training objective and retrieval representation are still insufficient for
query-specific factual binding.

The strongest current claim remains:

```text
Delta Q/V injection inside attention gives a large memory-channel improvement
over ordinary frozen attention, but query-specific retrieval/binding is not yet
isolated by the current training objective.
```

## Next adjustment

The next meaningful experiment should compare against a hidden/KV retrieval
baseline and then redesign retrieval supervision. More seeds on this exact
contrastive setup would mostly quantify a negative result rather than solve the
mechanism issue.

