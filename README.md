# Delta Memory

Delta Memory is a cleanroom prototype for **layerwise external memory injection
inside frozen Gemma-style decoder attention layers**. The Python package is still
named `rcvhc` for compatibility with earlier experiments.

The current research path is:

```text
long context
-> frozen Gemma forward
-> per-block Raw/Delta attention memory
-> external CPU/disk store
-> top-k memory retrieval
-> per-layer Gemma Q/K/V residual injection
-> answer metrics and attention-memory trace
```

It is not a RAG prompt-insertion system. Retrieved `source_text` is kept only as
debug metadata and is not appended to the answer prompt in the Delta Memory path.

## Delta Memory

The core memory signal is:

```text
source block b = [start, end)
future set F(b) = tokens i where i >= end
w[i,b] = sum_j in block A_fwd[i,j]
U[b,i] = w[i,b] / (sum_future w + eps)

c_self[b] = compressed source-block state
c_use[i]  = compressed future-use state
v2[b]     = sum_i U[b,i] * c_use[i]
Delta[b]  = RMSNorm(v2[b] - c_self[b])
```

For attention-path intervention, retrieved Delta is projected into Q/K/V
residuals at each enabled attention layer:

```text
q' = q + alpha_q * gate_q * P_q(Delta[b])
k' = k + alpha_k * gate_k * P_k(Delta[b])
v' = v + alpha_v * gate_v * P_v(Delta[b])
```

The base Gemma model is frozen. Only the Delta Memory writer, gates, and Q/K/V
projection adapters are trainable.

## Quick Start

Install dependencies in a local virtual environment:

```bash
python3 -m venv .venv-mac
.venv-mac/bin/python -m pip install torch transformers accelerate safetensors tokenizers pytest
```

Run tests:

```bash
.venv-mac/bin/python -m pytest -q
```

Run the fast mock wiring demo:

```bash
.venv-mac/bin/python scripts/run_gemma4_prototype.py \
  --model mock-gemma \
  --device cpu \
  --dtype float32 \
  --block-size 32 \
  --memory-dim 128
```

Run the real Gemma4 E2B wiring demo:

```bash
.venv-mac/bin/python scripts/run_gemma4_prototype.py \
  --model google/gemma-4-E2B \
  --device mps \
  --dtype bfloat16 \
  --block-size 128 \
  --memory-dim 512 \
  --report-dir reports/cleanroom/gemma4_real
```

Train only the Delta Memory writer and Q/V adapter:

```bash
.venv-mac/bin/python scripts/train_delta_qv_prototype.py \
  --model google/gemma-4-E2B \
  --device mps \
  --dtype bfloat16 \
  --steps 5 \
  --block-size 128 \
  --memory-dim 512 \
  --report-dir reports/cleanroom/gemma4_training
```

On Apple Silicon, run real Gemma experiments with `--device mps` outside
restricted sandboxes. See `docs/apple_silicon.md`. For quick verification of
adapter optimization, use `--model mock-gemma`; for scientific claims, run the
real model and keep zero/random/shuffled controls.

## Current Evidence

All current real-model reports use `google/gemma-4-E2B` on Apple Metal/MPS,
freeze the base model (`trainable_base_params = 0`), and keep retrieved source
text out of the prompt. The strongest current claim is deliberately narrow:

```text
Delta Q/V injection inside attention creates a large memory-channel improvement
over ordinary frozen attention, but query-specific retrieval/binding is not yet
isolated as the causal source.
```

### Main controlled evidence

| experiment | main result | alignment/control result | remote report |
| --- | --- | --- | --- |
| Question-only query eval32 | `delta_qv` NLL `5.5366` vs no-memory `12.7923`; 160/160 held-out examples improve | `wrong_query` is tied with correct Delta (`5.5399`), so binding is not isolated | [reports/cleanroom/question_only_query_eval32](https://github.com/KeikaJames/RCV-HC/tree/main/reports/cleanroom/question_only_query_eval32) |
| Conflict-margin pilot | `delta_qv` NLL `5.1661` vs no-memory `12.8438` | mean margin advantage vs wrong-query is `-0.0070`; negative alignment result | [reports/cleanroom/conflict_margin_pilot](https://github.com/KeikaJames/RCV-HC/tree/main/reports/cleanroom/conflict_margin_pilot) |
| Paired-conflict pilot | `delta_qv` NLL `4.6836` vs no-memory `12.3341` | paired same-unit conflict margin advantage is only `0.0094`; not enough | [reports/cleanroom/paired_conflict_pilot](https://github.com/KeikaJames/RCV-HC/tree/main/reports/cleanroom/paired_conflict_pilot) |
| Contrastive alignment pilot | isolated contrastive `delta_qv` NLL `5.4597`; shared contrastive `5.2961` vs no-memory `12.3341` | shared contrastive margin improves to `0.0295` but fails shuffled gate (`shuffled` `5.2929`) | [reports/cleanroom/contrastive_alignment_pilot](https://github.com/KeikaJames/RCV-HC/tree/main/reports/cleanroom/contrastive_alignment_pilot) |
| Hidden retrieval baseline pilot | `delta_qv` NLL `5.8246` vs hidden retrieval `14.5274` and no-memory `12.2118` | lightweight hidden/raw late-fusion baseline is not competitive; not a full RetrievalAttention baseline | [reports/cleanroom/hidden_retrieval_baseline_pilot](https://github.com/KeikaJames/RCV-HC/tree/main/reports/cleanroom/hidden_retrieval_baseline_pilot) |
| Long-distance NoLiMa-style pilot | `delta_qv` NLL `4.9367` vs no-memory `11.8879` | fails shuffled-control gate (`shuffled` `4.8210`), margin advantage near zero | [reports/cleanroom/long_distance_nolima_pilot](https://github.com/KeikaJames/RCV-HC/tree/main/reports/cleanroom/long_distance_nolima_pilot) |

### Superseded or early mechanism reports

These runs are retained for provenance but should not be used as final evidence
because later experiments found and fixed stronger controls.

| experiment | status | remote report |
| --- | --- | --- |
| Scaled MPS Delta experiment | early controlled mechanism result; `delta_qv` NLL `8.3279` vs no-memory `12.3188` across 3 seeds | [reports/cleanroom/gemma4_delta_experiment_mps_scaled](https://github.com/KeikaJames/RCV-HC/tree/main/reports/cleanroom/gemma4_delta_experiment_mps_scaled) |
| Expanded Delta Memory eval8 | superseded by answer-randomization and question-only-query fixes | [reports/cleanroom/delta_memory_expanded](https://github.com/KeikaJames/RCV-HC/tree/main/reports/cleanroom/delta_memory_expanded) |
| Delta vs ordinary attention eval32 | superseded by answer-randomization and question-only-query fixes | [reports/cleanroom/delta_vs_attention_eval32](https://github.com/KeikaJames/RCV-HC/tree/main/reports/cleanroom/delta_vs_attention_eval32) |
| Corrected random-answer eval32 | superseded because retrieval query still used teacher-forced answer tokens | [reports/cleanroom/corrected_random_answers_eval32](https://github.com/KeikaJames/RCV-HC/tree/main/reports/cleanroom/corrected_random_answers_eval32) |
| Layer ablation eval32 | superseded by later evaluation fixes; useful only as early layer-policy signal | [reports/cleanroom/layer_ablation_eval32](https://github.com/KeikaJames/RCV-HC/tree/main/reports/cleanroom/layer_ablation_eval32) |

## Literature Positioning and Causal Proof Plan

Recent memory-system work points to a stricter bar than "NLL improves":

- **Memorizing Transformers / RetrievalAttention / LongMem-style systems** make
  retrieval identity central: a query should select specific external KV/hidden
  records and the selected record should control the answer.
- **Titans-style neural long-term memory** emphasizes test-time memory updates
  and explicit interaction between short-term attention and long-term memory.
- **Mamba/SSM hybrids** are a different axis: they improve long-sequence
  efficiency by changing the backbone/state update, while Delta Memory keeps the
  Transformer backbone frozen and injects external memory into attention.

For Delta Memory, the missing causal proof is query-specific binding. The next
valid proof should use these gates:

1. **Shared memory competition**: retrieval must choose from a pool containing
   correct and conflicting memories, not from an isolated per-sample store.
2. **Counterfactual memory swap**: holding the question fixed, swapping in a
   paired foreign memory should move likelihood toward the foreign answer or at
   least reduce the correct-vs-foreign margin.
3. **Margin objective and metric**: report
   `foreign_answer_nll - correct_answer_nll`; correct memory must increase this
   margin over wrong-query/foreign memory.
4. **Shuffled/wrong-query gates**: `delta_qv` must beat shuffled and wrong-query
   controls, not only zero/random/no-memory.
5. **Stronger baseline**: compare against a real retrieved-KV/attention baseline,
   not only the lightweight hidden late-fusion `hidden_retrieval` baseline.

Until those gates pass, larger-seed confirmation is intentionally deferred.

## Repository Layout

```text
rcvhc/core/       config and shared typed records
rcvhc/memory/     external Delta Memory store and writer
rcvhc/gemma/      Gemma-style model adapter and layerwise Q/K/V injector
rcvhc/engine/     ingest, ask, prototype, and adapter-training runners
scripts/          runnable demos
docs/             design and evidence boundary
reports/          minimal tracked cleanroom reports only
tests/            cleanroom tests; no Gemma download required
```
