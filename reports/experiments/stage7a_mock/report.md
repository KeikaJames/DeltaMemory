# Stage 7A diagnostic — mock-gemma

- task suite: `address_token_binding_single_token`
- closed vocab size: `32` (random baseline = `0.031`)
- train / eval examples kept: `16 / 16`
- layers probed (hidden_states idx): `[1, 2]`
- pools: `['mean', 'attn']`
- spans: `['value', 'address']`
- steps / lr: `100` / `0.0005`

## Best cell

- span=value, layer=1, pool=mean
- eval top1 = **0.375** (top3 = 0.375, margin = -1.68, correct_logp = -3.99)
- train top1 = 0.938

## Strict gate (eval top1 >= 0.85): **FAIL**

## All cells (sorted by eval top1)

| span | layer | pool | train_top1 | eval_top1 | eval_top3 | margin | correct_logp |
|---|---:|---|---:|---:|---:|---:|---:|
| value | 1 | mean | 0.938 | 0.375 | 0.375 | -1.68 | -3.99 |
| value | 1 | attn | 1.000 | 0.375 | 0.375 | -1.68 | -3.91 |
| value | 2 | mean | 1.000 | 0.375 | 0.375 | -1.71 | -4.06 |
| value | 2 | attn | 1.000 | 0.312 | 0.375 | -1.61 | -4.01 |
| address | 1 | attn | 1.000 | 0.062 | 0.062 | -1.62 | -3.83 |
| address | 1 | mean | 1.000 | 0.000 | 0.125 | -1.33 | -3.54 |
| address | 2 | mean | 1.000 | 0.000 | 0.125 | -1.85 | -3.88 |
| address | 2 | attn | 1.000 | 0.000 | 0.125 | -1.64 | -3.77 |

## Verdict

No (span, layer, pool) cell reaches eval top1 >= 0.85 on held-out. The frozen
LLM does not surface the answer token at oracle spans under this prompt format.
Per the Stage 7 stop rule, do **not** start Phase 7B; instead either redesign
the address-card prompt to expose answer identity at a readable position, or
package the negative result as part of Story A.
