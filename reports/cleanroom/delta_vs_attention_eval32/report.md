# Delta Memory vs Ordinary Attention Eval32

> Superseded note: this report used the original synthetic answer generator,
> whose answer sequence was deterministic across train/eval seeds. Treat it as
> an early mechanism signal. The corrected random-answer rerun is
> `reports/cleanroom/corrected_random_answers_eval32/report.md`.

## Question

Does layerwise Delta Memory injection inside attention outperform the frozen
Gemma model's ordinary attention path?

This experiment compares:

- `delta_qv`: retrieved Delta Memory projected into Q/V residuals and injected
  inside every exposed attention layer.
- `no_memory`: ordinary frozen Gemma attention with no external memory and no
  injection.

It is not MCP, not tool context, and not RAG. Retrieved source text is not
inserted into the prompt.

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `backend`: `Apple Metal / PyTorch MPS`
- `dtype`: `bfloat16`
- `steps`: `12`
- `train_samples`: `16`
- `eval_samples`: `32`
- `seeds`: `[3]`
- `task_suites`: `single_fact_late_reference`, `multi_hop_binding`, `temporal_overwrite`, `paraphrase_nolima_style`, `adversarial_negative`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `layers`: `all`
- `injected_layers`: `15`
- `base_frozen`: `true`
- `prompt_insertion_used`: `false`

## Global Result

| metric | value |
| --- | ---: |
| runs | 5 |
| total_eval_examples | 160 |
| all_supported | 1 |
| mean_delta_nll | 5.7511 |
| mean_ordinary_attention_nll | 12.6248 |
| mean_raw_memory_nll | 15.1354 |
| mean_zero_nll | 12.6006 |
| mean_random_nll | 12.5373 |
| mean_shuffled_nll | 9.9458 |
| mean_wrong_layer_nll | 7.0647 |
| mean_nll_drop_vs_ordinary_attention | 6.8737 |
| per_example_win_rate_vs_ordinary_attention | 1.0000 |

## Per Suite

| task | seed | delta_nll | ordinary_attention_nll | raw | zero | random | shuffled | wrong_layer | drop_vs_attention | win_rate | ci95 | permutation_p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| adversarial_negative | 3 | 5.7006 | 12.3991 | 14.9525 | 12.3728 | 12.3710 | 9.1290 | 6.0552 | 6.6985 | 1.0000 | [6.2005, 7.1765] | 0.0005 |
| multi_hop_binding | 3 | 5.2458 | 11.8099 | 13.9724 | 11.7892 | 11.5740 | 9.2085 | 5.8128 | 6.5641 | 1.0000 | [6.2044, 6.9395] | 0.0005 |
| paraphrase_nolima_style | 3 | 4.3414 | 13.1659 | 16.7138 | 13.2139 | 13.1120 | 9.3804 | 6.4138 | 8.8246 | 1.0000 | [8.4414, 9.2062] | 0.0005 |
| single_fact_late_reference | 3 | 6.7323 | 12.3902 | 14.8822 | 12.3281 | 12.3639 | 10.8776 | 8.1283 | 5.6579 | 1.0000 | [5.3586, 5.9519] | 0.0005 |
| temporal_overwrite | 3 | 6.7353 | 13.3589 | 15.1559 | 13.2993 | 13.2658 | 11.1332 | 8.9133 | 6.6236 | 1.0000 | [6.2140, 7.0083] | 0.0005 |

## Interpretation

In this eval32 run, Delta Memory's attention-internal Q/V residual injection is
substantially better than ordinary frozen Gemma attention on every evaluated
example. The comparison is direct: `delta_qv` vs `no_memory`, with the same
questions and answer-token NLL metric.

The base model remains frozen, the injected path uses all 15 exposed attention
layers, and no retrieved source text is inserted into the prompt. This supports
the claim that the effect is from Delta injection inside attention, not from RAG
or an external MCP/tool-context path.
