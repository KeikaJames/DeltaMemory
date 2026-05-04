# Mneme Expanded Gemma/MPS Report

> Superseded note: this report used the original synthetic answer generator,
> whose answer sequence was deterministic across train/eval seeds. Treat it as
> an early mechanism signal. The corrected random-answer rerun is
> `reports/experiments/corrected_random_answers_eval32/report.md`.

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `mps`
- `backend`: `Apple Metal / PyTorch MPS`
- `dtype`: `bfloat16`
- `steps`: `8`
- `train_samples`: `8`
- `eval_samples`: `8`
- `seeds`: `[0, 1, 2]`
- `task_suites`: `single_fact_late_reference`, `multi_hop_binding`, `temporal_overwrite`, `paraphrase_nolima_style`, `adversarial_negative`
- `block_size`: `64`
- `memory_dim`: `512`
- `top_k`: `2`
- `layers`: `all`
- `injected_layers`: `15`
- `base_frozen`: `true`
- `prompt_insertion_used`: `false`

## Global Aggregate

| metric | value |
| --- | ---: |
| runs | 15 |
| support_rate | 1.0000 |
| mean_delta_nll | 6.2974 |
| mean_no_memory_nll | 12.6521 |
| mean_raw_memory_nll | 15.1717 |
| mean_zero_nll | 12.6285 |
| mean_random_nll | 12.5611 |
| mean_shuffled_nll | 9.5711 |
| mean_wrong_layer_nll | 8.0515 |
| mean_eval_delta_nll_drop | 6.3507 |
| std_eval_delta_nll_drop | 0.8661 |
| mean_q_delta_norm | 54.8222 |
| mean_v_delta_norm | 19.2876 |
| mean_gate_v | 0.2917 |

All runs kept `trainable_base_params = 0`, used `layers = all`, injected into 15 attention layers, and kept retrieved source text out of the prompt.

## By Task

| task | runs | support | delta | no_mem | raw | zero | random | shuffled | wrong_layer | drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single_fact_late_reference | 3 | 1.0000 | 6.3231 | 12.3188 | 14.9266 | 12.2186 | 12.3022 | 8.2789 | 7.9031 | 6.0369 |
| multi_hop_binding | 3 | 1.0000 | 6.1553 | 11.9977 | 14.1662 | 11.9984 | 11.8637 | 9.4412 | 8.8838 | 5.7625 |
| temporal_overwrite | 3 | 1.0000 | 6.6669 | 13.2571 | 15.0809 | 13.2202 | 13.2068 | 9.6797 | 7.7492 | 6.6112 |
| paraphrase_nolima_style | 3 | 1.0000 | 5.9444 | 13.2017 | 16.6333 | 13.2058 | 13.0263 | 10.5245 | 6.8911 | 7.0878 |
| adversarial_negative | 3 | 1.0000 | 6.3976 | 12.4852 | 15.0513 | 12.4993 | 12.4064 | 9.9312 | 8.8303 | 6.2551 |

## Per-Run Results

| task | seed | supported | delta | no_mem | raw | zero | random | shuffled | wrong_layer | drop | q_norm | v_norm | gate_v |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial_negative | 0 | True | 6.2686 | 12.3156 | 15.1120 | 12.3156 | 12.2289 | 9.8199 | 8.6355 | 6.2164 | 56.0002 | 19.6691 | 0.2920 |
| adversarial_negative | 1 | True | 6.2714 | 12.5639 | 15.1700 | 12.6151 | 12.4051 | 9.9639 | 8.8729 | 6.4776 | 58.9641 | 20.8801 | 0.2915 |
| adversarial_negative | 2 | True | 6.6528 | 12.5760 | 14.8718 | 12.5673 | 12.5853 | 10.0097 | 8.9825 | 6.0712 | 58.5375 | 21.1829 | 0.2915 |
| multi_hop_binding | 0 | True | 5.5926 | 12.3026 | 14.2573 | 12.3026 | 12.2111 | 9.0915 | 8.9340 | 6.6779 | 56.5461 | 19.6208 | 0.2917 |
| multi_hop_binding | 1 | True | 5.8826 | 12.0682 | 14.2135 | 12.1105 | 11.9830 | 9.4560 | 8.8959 | 5.8453 | 60.3925 | 21.0526 | 0.2914 |
| multi_hop_binding | 2 | True | 6.9907 | 11.6221 | 14.0278 | 11.5821 | 11.3970 | 9.7760 | 8.8214 | 4.7643 | 58.0816 | 21.2416 | 0.2915 |
| paraphrase_nolima_style | 0 | True | 5.5390 | 13.3620 | 16.8753 | 13.3620 | 13.2570 | 10.5929 | 6.4550 | 7.6465 | 54.2368 | 19.2361 | 0.2917 |
| paraphrase_nolima_style | 1 | True | 5.5959 | 12.9648 | 16.2348 | 12.9791 | 12.7527 | 10.2718 | 7.1403 | 6.8480 | 56.0151 | 19.5020 | 0.2918 |
| paraphrase_nolima_style | 2 | True | 6.6984 | 13.2785 | 16.7899 | 13.2763 | 13.0692 | 10.7089 | 7.0781 | 6.7690 | 55.3448 | 19.9815 | 0.2916 |
| single_fact_late_reference | 0 | True | 6.3113 | 12.2260 | 14.6495 | 11.9196 | 11.8848 | 8.5228 | 7.8387 | 5.9214 | 49.8757 | 17.4072 | 0.2921 |
| single_fact_late_reference | 1 | True | 5.7729 | 12.5839 | 15.1044 | 12.2266 | 12.2238 | 8.0069 | 7.3689 | 6.9336 | 53.3097 | 18.3419 | 0.2916 |
| single_fact_late_reference | 2 | True | 6.8849 | 12.1465 | 15.0259 | 12.5095 | 12.7981 | 8.3070 | 8.5016 | 5.2557 | 51.7586 | 18.4342 | 0.2919 |
| temporal_overwrite | 0 | True | 6.3769 | 12.6296 | 14.8329 | 12.6296 | 12.5520 | 9.8087 | 7.3331 | 6.2979 | 50.6722 | 17.9749 | 0.2917 |
| temporal_overwrite | 1 | True | 7.3171 | 12.8581 | 15.0535 | 12.8344 | 12.7664 | 9.6533 | 8.0165 | 5.3384 | 55.2035 | 19.4624 | 0.2916 |
| temporal_overwrite | 2 | True | 6.3066 | 14.2838 | 15.3564 | 14.1966 | 14.3019 | 9.5760 | 7.8980 | 8.1973 | 57.3946 | 19.3279 | 0.2915 |

## Interpretation

This is an expanded mechanism experiment, not an MCP/tool-context or RAG experiment. Delta is injected directly into the frozen Gemma attention stack through Q/K/V residual hooks for every exposed attention layer. Retrieved source text remains debug metadata and is not inserted into the prompt.

The result strengthens the mechanism signal across five harder synthetic suites: multi-hop binding, temporal overwrite, paraphrase/NoLiMa-style low-overlap recall, adversarial near-negative recall, and the previous single-fact later-reference task. The strongest remaining gap is that shuffled and wrong-layer controls are still better than no-memory in some settings, so the next experiment should include last-layer-vs-all-layer, oracle retrieval, and wrong-query controls with larger eval sets.
