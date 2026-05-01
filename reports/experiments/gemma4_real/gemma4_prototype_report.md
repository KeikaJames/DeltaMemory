# RCV-HC Gemma4 Attention-Memory Prototype

## Config

- `model`: `google/gemma-4-E2B`
- `device`: `auto`
- `dtype`: `bfloat16`
- `block_size`: `128`
- `memory_dim`: `512`
- `top_k`: `2`
- `layers`: `max_exposed`
- `alpha_scale`: `0.2`
- `gate_bias`: `-1.0`
- `prompt_insertion_used`: `False`
- `claim_boundary`: `real_model_prototype_controls_required_for_scientific_claim`

## Ingest

- `total_tokens_ingested`: `177`
- `enabled_layers`: `[14]`
- `memory_blocks`: `2`
- `storage_bytes`: `21528`
- `trainable_base_params`: `0`

## Comparisons

| mode | nll | rank | top10 | q_delta | v_delta | gate_v |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 11.7211 | 37901.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| raw_memory | 15.4504 | 37232.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| delta_qv | 11.7461 | 37810.2500 | 0.5000 | 1.5410 | 0.5392 | 0.2695 |
| delta_qv_zero | 11.7211 | 37901.0000 | 0.5000 | 0.0000 | 0.0000 | 0.2695 |
| delta_qv_random | 11.7421 | 36901.5000 | 0.5000 | 1.5658 | 0.5456 | 0.2695 |
| delta_qv_shuffled | 11.7374 | 37932.5000 | 0.5000 | 1.5444 | 0.5172 | 0.2695 |
| delta_qv_force_gate | 11.7967 | 38872.2500 | 0.5000 | 5.7189 | 2.0007 | 1.0000 |

## Retrieved Memory

Retrieved source snippets are debug metadata only and were not inserted into the prompt.

- layer `14`, block `0`, score `0.2294`, range `[0, 128]`, usage `19.9175`
- layer `14`, block `1`, score `0.1822`, range `[128, 177]`, usage `0.0000`

## Diagnosis

- `delta_qv_q_nonzero`: `True`
- `delta_qv_v_nonzero`: `True`
- `delta_beats_zero_random`: `False`
- `delta_beats_shuffled`: `False`
- `force_gate_stronger`: `False`
- `signal_status`: `wiring_signal_only`

## Interpretation

This is the first practical Gemma4-oriented RCV-HC prototype path: external attention memory, top-k retrieval, and Q/K/V injection into a frozen decoder LM.
A scientific claim requires aligned Delta to beat zero, random, and shuffled controls on a real model run. Otherwise the result should be treated as engineering progress only.
