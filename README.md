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
  <a href="reports/cleanroom">Reports</a>
</p>

---

Delta Memory is a cleanroom research prototype for injecting external memory
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
| Next direction | **Address-Bound Delta Memory**: separate memory identity from payload injection. |

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
query-address margin -> identity gate -> signed Delta payload -> Q/V residual
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
| Question-only query eval32 | `5.5366` vs `12.7923` NLL | `wrong_query` tied with correct Delta (`5.5399`) | [report](reports/cleanroom/question_only_query_eval32) |
| Conflict-margin pilot | `5.1661` vs `12.8438` NLL | margin advantage `-0.0070` | [report](reports/cleanroom/conflict_margin_pilot) |
| Paired-conflict pilot | `4.6836` vs `12.3341` NLL | margin advantage `0.0094` | [report](reports/cleanroom/paired_conflict_pilot) |
| Contrastive alignment pilot | `5.2961` vs `12.3341` NLL | shared contrastive fails shuffled gate | [report](reports/cleanroom/contrastive_alignment_pilot) |
| Hidden retrieval baseline | `5.8246` vs `12.2118` NLL | hidden late-fusion baseline is weak (`14.5274`) | [report](reports/cleanroom/hidden_retrieval_baseline_pilot) |
| Long-distance NoLiMa-style | `4.9367` vs `11.8879` NLL | fails shuffled gate (`4.8210`) | [report](reports/cleanroom/long_distance_nolima_pilot) |

### Superseded reports

Earlier runs are preserved for provenance but should not be used as final
evidence because later experiments fixed answer-pattern leakage, teacher-forced
retrieval queries, or stronger controls.

| Experiment | Status | Report |
| --- | --- | --- |
| Scaled MPS Delta experiment | early mechanism signal | [report](reports/cleanroom/gemma4_delta_experiment_mps_scaled) |
| Expanded Delta Memory eval8 | superseded by answer-randomization and question-only-query fixes | [report](reports/cleanroom/delta_memory_expanded) |
| Delta vs ordinary attention eval32 | superseded by evaluation fixes | [report](reports/cleanroom/delta_vs_attention_eval32) |
| Corrected random-answer eval32 | superseded because retrieval query used answer tokens | [report](reports/cleanroom/corrected_random_answers_eval32) |
| Layer ablation eval32 | useful only as early layer-policy signal | [report](reports/cleanroom/layer_ablation_eval32) |

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

The proposed synthesis is **Address-Bound Delta Memory**:

```text
memory item = (address key, payload delta, anti-key metadata)
query      -> address competition -> causal gate -> payload injection
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
| [`reports/cleanroom`](reports/cleanroom) | tracked cleanroom experiment artifacts |

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
reports/          tracked cleanroom experiment reports
tests/            CI-safe mock tests; no Gemma download required
```

## License

Code and project documentation are released under the [MIT License](LICENSE).

Model weights, datasets, papers, and third-party dependencies are governed by
their own licenses and terms. Experiments that load `google/gemma-4-E2B` require
the user to comply with the applicable Gemma model license and access terms.
