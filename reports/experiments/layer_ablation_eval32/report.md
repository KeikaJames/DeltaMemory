# Delta Memory Layer Ablation Eval32

> Superseded note: this report used the original synthetic answer generator,
> whose answer sequence was deterministic across train/eval seeds. Treat it as
> an early layer-policy signal, not a final ablation. Re-run layer ablations on
> the corrected random-answer suites before making a strong layerwise claim.

## Question

Does injecting Delta Memory into every attention layer improve over a last-layer
only attention injection ablation?

This compares the same Gemma/MPS eval32 setup under:

- `layers=all`: Delta Q/V residuals injected into all 15 exposed attention layers.
- `layers=max_exposed`: Delta Q/V residuals injected only into the last exposed attention layer.
- `no_memory`: ordinary frozen Gemma attention.

No retrieved source text is inserted into the prompt.

## Global Result

| metric | value |
| --- | ---: |
| tasks | 5 |
| total_eval_examples | 160 |
| tasks_all_layer_better | 4 |
| mean_all_layers_nll | 5.7511 |
| mean_last_layer_nll | 6.5633 |
| mean_ordinary_attention_nll | 12.6248 |
| mean_all_drop_vs_attention | 6.8737 |
| mean_last_drop_vs_attention | 6.0616 |
| mean_last_minus_all_nll | 0.8122 |
| per_example_all_better_rate | 0.6750 |

## Per Task

| task | all_layers_nll | last_layer_nll | ordinary_attention_nll | all_better | all_drop | last_drop | injected_layers_all | injected_layers_last |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| single_fact_late_reference | 6.7323 | 6.2863 | 12.3902 | False | 5.6579 | 6.1039 | 15 | 1 |
| multi_hop_binding | 5.2458 | 5.7360 | 11.8099 | True | 6.5641 | 6.0739 | 15 | 1 |
| temporal_overwrite | 6.7353 | 6.9612 | 13.3589 | True | 6.6236 | 6.3977 | 15 | 1 |
| paraphrase_nolima_style | 4.3414 | 7.1812 | 13.1659 | True | 8.8246 | 5.9847 | 15 | 1 |
| adversarial_negative | 5.7006 | 6.6515 | 12.3991 | True | 6.6985 | 5.7476 | 15 | 1 |

## Interpretation

Both Delta Memory variants are much better than ordinary frozen Gemma attention.
The all-layer path is stronger on average and wins 4 of 5 task suites, with a
mean NLL advantage of `0.8122` over last-layer-only injection.

The one exception is `single_fact_late_reference`, where last-layer-only is
better. That suggests the next step should not blindly assume all-layer is
always optimal. The right next experiment is a layer-policy sweep:
`last`, `first+middle+last`, `every_2nd`, and `all`, plus wrong-query controls.
