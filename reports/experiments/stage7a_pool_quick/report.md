# Stage 7A diagnostic — google/gemma-4-E2B

- task suite: `address_token_binding_single_token`
- closed vocab size: `32` (random baseline = `0.031`)
- train / eval examples kept: `32 / 32`
- layers probed (hidden_states idx): `[1, 8, 16, 28]`
- pools: `['mean', 'last']`
- spans: `['value', 'address']`
- steps / lr: `200` / `0.0005`

## Best cell

- span=value, layer=1, pool=mean
- eval top1 = **0.094** (top3 = 0.188, margin = -0.05, correct_logp = -3.46)
- train top1 = 0.406

## Strict gate (eval top1 >= 0.85): **FAIL**

## All cells (sorted by eval top1)

| span | layer | pool | train_top1 | eval_top1 | eval_top3 | margin | correct_logp |
|---|---:|---|---:|---:|---:|---:|---:|
| value | 1 | mean | 0.406 | 0.094 | 0.188 | -0.05 | -3.46 |
| value | 1 | last | 0.469 | 0.094 | 0.188 | -0.06 | -3.46 |
| value | 8 | mean | 1.000 | 0.062 | 0.125 | -1.19 | -3.70 |
| value | 8 | last | 1.000 | 0.062 | 0.125 | -0.93 | -3.51 |
| address | 8 | last | 1.000 | 0.062 | 0.062 | -3.90 | -4.96 |
| value | 16 | last | 1.000 | 0.031 | 0.125 | -2.27 | -4.20 |
| address | 8 | mean | 1.000 | 0.031 | 0.094 | -2.10 | -3.89 |
| address | 16 | mean | 1.000 | 0.031 | 0.094 | -1.77 | -3.87 |
| address | 16 | last | 1.000 | 0.031 | 0.125 | -2.99 | -4.59 |
| address | 28 | mean | 1.000 | 0.031 | 0.062 | -1.70 | -3.74 |
| value | 16 | mean | 1.000 | 0.000 | 0.031 | -2.42 | -4.27 |
| value | 28 | mean | 1.000 | 0.000 | 0.031 | -1.54 | -3.79 |
| value | 28 | last | 1.000 | 0.000 | 0.031 | -1.72 | -3.87 |
| address | 1 | mean | 0.969 | 0.000 | 0.000 | -1.27 | -3.65 |
| address | 1 | last | 0.969 | 0.000 | 0.156 | -3.14 | -4.57 |
| address | 28 | last | 1.000 | 0.000 | 0.188 | -3.12 | -4.38 |

## Verdict

No (span, layer, pool) cell reaches eval top1 >= 0.85 on held-out. The frozen
LLM does not surface the answer token at oracle spans under this prompt format.
Per the Stage 7 stop rule, do **not** start Phase 7B; instead either redesign
the address-card prompt to expose answer identity at a readable position, or
package the negative result as part of Story A.
