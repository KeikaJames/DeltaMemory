# Stage 7A diagnostic — google/gemma-4-E2B

- task suite: `address_token_binding_single_token`
- closed vocab size: `32` (random baseline = `0.031`)
- train / eval examples kept: `32 / 32`
- layers probed (hidden_states idx): `[4, 16, 28]`
- pools: `['mean']`
- spans: `['value', 'address']`
- steps / lr: `200` / `0.0005`

## Best cell

- span=value, layer=4, pool=mean
- eval top1 = **0.062** (top3 = 0.188, margin = -0.21, correct_logp = -3.44)
- train top1 = 0.875

## Strict gate (eval top1 >= 0.85): **FAIL**

## All cells (sorted by eval top1)

| span | layer | pool | train_top1 | eval_top1 | eval_top3 | margin | correct_logp |
|---|---:|---|---:|---:|---:|---:|---:|
| value | 4 | mean | 0.875 | 0.062 | 0.188 | -0.21 | -3.44 |
| address | 16 | mean | 1.000 | 0.062 | 0.125 | -1.72 | -3.84 |
| address | 4 | mean | 1.000 | 0.031 | 0.031 | -1.46 | -3.64 |
| address | 28 | mean | 1.000 | 0.031 | 0.062 | -1.69 | -3.73 |
| value | 16 | mean | 1.000 | 0.000 | 0.094 | -2.46 | -4.30 |
| value | 28 | mean | 1.000 | 0.000 | 0.000 | -1.51 | -3.80 |

## Verdict

No (span, layer, pool) cell reaches eval top1 >= 0.85 on held-out. The frozen
LLM does not surface the answer token at oracle spans under this prompt format.
Per the Stage 7 stop rule, do **not** start Phase 7B; instead either redesign
the address-card prompt to expose answer identity at a readable position, or
package the negative result as part of Story A.
