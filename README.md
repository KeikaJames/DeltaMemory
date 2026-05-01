<p align="center">
  <h1 align="center">Delta Memory</h1>
</p>

<p align="center">
  <strong>Layerwise external memory injection inside frozen Transformer attention.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-MPS%2FCPU-EE4C2C.svg">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-orange.svg">
</p>

<p align="center">
  <a href="docs/address_bound_delta_memory_plan.md">Research Plan</a> ·
  <a href="docs/design.md">Design</a> ·
  <a href="docs/apple_silicon.md">Apple Silicon</a> ·
  <a href="reports/experiments">Reports</a>
</p>

---

Delta Memory is an experimental research prototype for injecting external memory
directly into frozen Gemma-style decoder attention layers. The package is still
named `rcvhc` for compatibility with earlier experiments.

It is **not RAG**, **not MCP**, and **not prompt insertion**. Retrieved source
text is retained only as debug metadata; it is not appended to the answer
prompt in the Delta Memory path.

## At a glance

| Question | Current answer |
| --- | --- |
| What is changed? | Q/V residuals inside every enabled attention layer. |
| What stays frozen? | The base Gemma model; only writer/projector/gates train. |
| What is proven? | A strong in-attention memory-channel effect over ordinary frozen attention. |
| What is not proven yet? | Query-specific retrieval/binding as the causal source. |
| Next direction | **Token/Span-Bound Delta Memory**: separate address spans, value spans, and payload injection. |

## Mechanism

Delta Memory writes per-block Raw/Delta attention memory from a source context,
retrieves memory blocks by query, and projects retrieved Delta payloads into
attention-internal residuals:

```text
q' = q + alpha_q * gate_q * P_q(Delta)
v' = v + alpha_v * gate_v * P_v(Delta)
```

The current research hypothesis is now stricter:

```text
question address span -> memory address key
source value span     -> signed payload Delta
address classifier    -> identity gate -> Q/V residual
```

See [`docs/address_bound_delta_memory_plan.md`](docs/address_bound_delta_memory_plan.md)
for the next experiment plan.

## Current evidence

All current real-model evidence uses `google/gemma-4-E2B` on Apple Metal/MPS,
keeps the base model frozen, and does not insert retrieved text into the prompt.

> **Claim boundary:** Delta Q/V injection is a strong memory channel, but current
> controls do not yet isolate query-specific retrieval/binding.

### Main reports

| Experiment | Delta vs ordinary attention | Alignment/control result | Report |
| --- | ---: | --- | --- |
| Question-only query eval32 | `5.5366` vs `12.7923` NLL | `wrong_query` tied with correct Delta (`5.5399`) | [report](reports/experiments/question_only_query_eval32) |
| Conflict-margin pilot | `5.1661` vs `12.8438` NLL | margin advantage `-0.0070` | [report](reports/experiments/conflict_margin_pilot) |
| Paired-conflict pilot | `4.6836` vs `12.3341` NLL | margin advantage `0.0094` | [report](reports/experiments/paired_conflict_pilot) |
| Contrastive alignment pilot | `5.2961` vs `12.3341` NLL | shared contrastive fails shuffled gate | [report](reports/experiments/contrastive_alignment_pilot) |
| Address-key projection pilot | `4.8159` vs `11.6111` NLL | address rank is poor (`5.25`); shuffled gap only `0.0062`, below the `0.05` gate | [report](reports/experiments/address_key_projection_pilot) |
| Identity-gate pilot | `5.3501` vs `11.6111` NLL; identity-gated Delta `6.9899` | gate suppresses weak addresses (`0.3796`) but correct rank remains `5.25`; shuffled gate fails | [report](reports/experiments/identity_gate_pilot) |
| Address-supervised pilot | `3.9278` vs `11.6111` NLL | address ranking loss improves channel but not binding; wrong-query and shuffled remain tied | [report](reports/experiments/address_supervised_pilot) |
| Address-token contrastive pilot | `3.0843` vs `12.1370` NLL | explicit address tokens plus contrastive loss still fail the `0.05` shuffled/wrong-query gate | [report](reports/experiments/address_token_contrastive_pilot) |
| Final address-bound multiseed | mean `4.0965` vs no-memory `12.1111` NLL over 3 seeds | support rate `0.0`; shuffled gap `0.0031`, wrong-query gap `0.0027`, address rank `4.9167` | [report](reports/experiments/address_bound_final_multiseed) |
| Query-address projector pilot | `2.5695` vs `12.1370` NLL | trainable query projection strengthens the channel but still fails shuffled/wrong-query; address rank `4.375` | [report](reports/experiments/query_address_projector_pilot) |
| Oracle payload control pilot | `5.1881` vs `12.1370` NLL | even forced correct-address payload barely beats forced paired payload; oracle margin advantage `0.0166` | [report](reports/experiments/oracle_address_control_pilot) |
| Binding stress pilot | `2.9874` vs `12.1370` NLL | high LR/weight stress test still fails; address scores collapse and oracle margin advantage is `-0.0178` | [report](reports/experiments/binding_stress_pilot) |
| Query-specific binding follow-up | best Delta NLL `2.5695` vs no-memory `12.1370` | all follow-ups fail binding; forced oracle payload also fails to separate paired answers | [report](reports/experiments/query_specific_binding_followup) |
| Oracle span payload pilot | `3.5718` vs no-memory `12.1946` NLL | oracle address/value spans preserve the channel but payload swap remains tied; margin advantage vs wrong-query `-0.0084` | [report](reports/experiments/oracle_span_payload_pilot) |
| Oracle span contrastive pilot | `4.3541` vs no-memory `12.1946` NLL | oracle contrastive raises margin advantage only to `0.0267`, far below the `0.5` payload-specificity gate | [report](reports/experiments/oracle_span_payload_contrastive_pilot) |
| Hidden retrieval baseline | `5.8246` vs `12.2118` NLL | hidden late-fusion baseline is weak (`14.5274`) | [report](reports/experiments/hidden_retrieval_baseline_pilot) |
| Long-distance NoLiMa-style | `4.9367` vs `11.8879` NLL | fails shuffled gate (`4.8210`) | [report](reports/experiments/long_distance_nolima_pilot) |

### Superseded reports

Earlier runs are preserved for provenance but should not be used as final
evidence because later experiments fixed answer-pattern leakage, teacher-forced
retrieval queries, or stronger controls.

| Experiment | Status | Report |
| --- | --- | --- |
| Scaled MPS Delta experiment | early mechanism signal | [report](reports/experiments/gemma4_delta_experiment_mps_scaled) |
| Expanded Delta Memory eval8 | superseded by answer-randomization and question-only-query fixes | [report](reports/experiments/delta_memory_expanded) |
| Delta vs ordinary attention eval32 | superseded by evaluation fixes | [report](reports/experiments/delta_vs_attention_eval32) |
| Corrected random-answer eval32 | superseded because retrieval query used answer tokens | [report](reports/experiments/corrected_random_answers_eval32) |
| Layer ablation eval32 | useful only as early layer-policy signal | [report](reports/experiments/layer_ablation_eval32) |

## Research direction

The literature tension is useful:

| Line of work | Strength | Gap for Delta Memory |
| --- | --- | --- |
| RETRO / Memorizing Transformers / LongMem / RetrievalAttention | explicit external records | retrieval identity can still be brittle or non-causal |
| Titans / neural long-term memory | adaptive memory channel | memory is powerful but opaque |
| Mamba / SSMs | efficient long-state propagation | weak for exact content-addressed binding |
| Infini-attention / compressive memory | bounded streaming memory | compression can erase counterfactual identity |
| NoLiMa / RULER | exposes long-context shortcuts | NLL alone is not enough |
| Delta-rule / fast-weight views | binding and anti-interference framing | current Delta path needs explicit address supervision |

The proposed synthesis is now **Token/Span-Bound Delta Memory**:

```text
memory item = (address span key, value span payload delta, anti-key metadata)
query      -> address span competition -> causal gate -> payload injection
```

Pass/fail gates before larger scaling:

| Gate | Requirement |
| --- | --- |
| Channel | Delta beats no-memory, zero, and random controls. |
| Address | correct memory ranks above paired negative in a shared pool. |
| Shuffled | correct-address Delta beats shuffled-address Delta. |
| Wrong-query | correct-address Delta beats wrong-query/foreign-address Delta. |
| Margin | correct memory improves `foreign_nll - correct_nll`. |
| Payload swap | correct address + foreign payload differs from correct address + correct payload. |
| Oracle span | oracle value-span payload beats paired value-span payload before learned retrieval is trusted. |
| Baseline | Delta beats hidden retrieval and a real retrieved-KV/attention baseline. |

## Quick start

```bash
python3 -m venv .venv-mac
.venv-mac/bin/python -m pip install torch transformers accelerate safetensors tokenizers pytest
.venv-mac/bin/python -m pytest -q
```

Run a fast mock demo:

```bash
.venv-mac/bin/python scripts/run_gemma4_prototype.py \
  --model mock-gemma \
  --device cpu \
  --dtype float32 \
  --block-size 32 \
  --memory-dim 128
```

Run a real Gemma/MPS experiment outside restricted sandboxes:

```bash
.venv-mac/bin/python scripts/run_delta_experiment.py \
  --model google/gemma-4-E2B \
  --device mps \
  --dtype bfloat16 \
  --steps 12 \
  --train-samples 16 \
  --eval-samples 16 \
  --task-suite paired_conflict_binding \
  --shared-memory-retrieval \
  --conflict-margins
```

See [`docs/apple_silicon.md`](docs/apple_silicon.md) for MPS/Metal notes.

## Documentation

| Document | Purpose |
| --- | --- |
| [`docs/address_bound_delta_memory_plan.md`](docs/address_bound_delta_memory_plan.md) | next experimental plan |
| [`docs/design.md`](docs/design.md) | architecture and evidence boundary |
| [`docs/gemma4_prototype.md`](docs/gemma4_prototype.md) | Gemma prototype runbook |
| [`docs/apple_silicon.md`](docs/apple_silicon.md) | Apple Silicon / MPS setup |
| [`reports/experiments`](reports/experiments) | tracked experiment artifacts |

## References

| Topic | Citation | DOI / link |
| --- | --- | --- |
| Retrieval-augmented language modeling | Borgeaud et al., "Improving Language Models by Retrieving from Trillions of Tokens" (RETRO), arXiv:2112.04426 | [10.48550/arXiv.2112.04426](https://doi.org/10.48550/arXiv.2112.04426) |
| External kNN memory | Wu et al., "Memorizing Transformers", arXiv:2203.08913 | [10.48550/arXiv.2203.08913](https://doi.org/10.48550/arXiv.2203.08913) |
| Long-context memory transformers | Wang et al., "Augmenting Language Models with Long-Term Memory" (LongMem), arXiv:2306.07174 | [10.48550/arXiv.2306.07174](https://doi.org/10.48550/arXiv.2306.07174) |
| Selective state-space models | Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv:2312.00752 | [10.48550/arXiv.2312.00752](https://doi.org/10.48550/arXiv.2312.00752) |
| Bounded compressive memory | Munkhdalai et al., "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention", arXiv:2404.07143 | [10.48550/arXiv.2404.07143](https://doi.org/10.48550/arXiv.2404.07143) |
| Real context-size evaluation | Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?", arXiv:2404.06654 | [10.48550/arXiv.2404.06654](https://doi.org/10.48550/arXiv.2404.06654) |
| Delta-rule / fast-weight memory | Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", arXiv:2406.06484 | [10.48550/arXiv.2406.06484](https://doi.org/10.48550/arXiv.2406.06484) |
| Test-time neural long-term memory | Behrouz et al., "Titans: Learning to Memorize at Test Time", arXiv:2501.00663 | [10.48550/arXiv.2501.00663](https://doi.org/10.48550/arXiv.2501.00663) |
| Long-context beyond literal matching | Modarressi et al., "NoLiMa: Long-Context Evaluation Beyond Literal Matching", arXiv:2502.05167 | [10.48550/arXiv.2502.05167](https://doi.org/10.48550/arXiv.2502.05167) |
| KV-cache retrieval baseline direction | "RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval", OpenReview | [OpenReview](https://openreview.net/forum?id=8z3cOVER4z) |

## Repository layout

```text
rcvhc/core/       config and shared typed records
rcvhc/memory/     external Delta Memory store and writer
rcvhc/gemma/      Gemma-style adapter and layerwise Q/K/V injector
rcvhc/engine/     ingest, ask, training, experiments, statistics
scripts/          runnable demos and experiment CLIs
docs/             design notes and research plans
reports/          tracked experiment reports
tests/            CI-safe mock tests; no Gemma download required
```

## License

Code and project documentation are released under the [MIT License](LICENSE).

Model weights, datasets, papers, and third-party dependencies are governed by
their own licenses and terms. Experiments that load `google/gemma-4-E2B` require
the user to comply with the applicable Gemma model license and access terms.
