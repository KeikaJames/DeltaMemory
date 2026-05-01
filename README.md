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

Recent memory-system work points to a stricter bar than "NLL improves". The
important papers also disagree with each other in useful ways:

| line of work | what it solves | what it fails to settle for Delta Memory |
| --- | --- | --- |
| Memorizing Transformers, RETRO, LongMem, RetrievalAttention | explicit external retrieval and KV/hidden record selection | retrieval can be approximate, brittle under near-collisions, or not causally tied to answer identity |
| Titans / MIRAS-style neural long-term memory | adaptive neural memory updated at test time, combining short-term attention and persistent memory | memory is powerful but opaque; a learned memory channel can improve loss without proving address-level binding |
| Mamba / SSM hybrids | efficient long-sequence state propagation without quadratic attention | compressed recurrent state is efficient but weak for exact content-addressed key-value recall |
| Infini-attention / compressive memory | bounded-memory streaming attention with local and long-term components | compression helps scale, but compression can erase the identity needed for counterfactual factual swaps |
| NoLiMa / RULER-style benchmarks | expose literal-match shortcuts and long-context binding failures | ordinary NLL gains are insufficient; controls must remove lexical overlap, answer leakage, and isolated-memory shortcuts |
| DeltaNet / fast-weight delta-rule views | frame attention-like lookup as associative memory update/readout | suggests Delta Memory should learn binding and anti-interference, not only inject a generic residual vector |

### Synthesis: Address-Bound Delta Memory

The contradiction is this:

```text
Attention/KV retrieval gives explicit addresses but is expensive and brittle.
Mamba/SSM/neural-memory compression is efficient but weak at exact binding.
Delta injection is powerful but can become a generic activation channel.
```

The proposed unification is **Address-Bound Delta Memory**:

```text
memory item = (address key, payload delta, anti-key metadata)
query      -> address competition -> causal gate -> payload injection
```

Instead of asking only whether `Delta` lowers answer NLL, the next architecture
should separate two roles:

1. **Address binding**: a query must select the correct memory identity from a
   shared pool containing near-collision and foreign memories.
2. **Payload effect**: only after the address gate wins should the Delta payload
   be allowed to affect Q/V.

This changes the mechanism from:

```text
retrieved Delta -> Q/V residual
```

to:

```text
query-address margin -> identity gate -> signed Delta payload -> Q/V residual
```

The practical consequence is that wrong memories should not merely be "less
helpful"; they should be detectably unable to open the correct gate, or should
move the model toward the paired foreign answer in a counterfactual swap. This
is the missing causal test in the current reports.

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

### Concrete next implementation direction

The next implementation should not just add another loss term to the current
adapter. It should add an explicit address-binding path:

1. **Memory records**
   - keep the current Delta payload (`delta_q`, `delta_v`);
   - add a learned `address_key` optimized for retrieval identity;
   - record paired negative ids for conflict suites.
2. **Retriever**
   - retrieve from a shared store;
   - compute a top-1 vs top-2 address margin;
   - expose `address_margin`, `correct_address_rank`, and `paired_negative_rank`
     in reports.
3. **Identity gate**
   - gate Delta injection by address confidence:
     `gate_identity = sigmoid(beta * (score_top1 - score_top2 - tau))`;
   - wrong-query or shuffled retrieval should close this gate.
4. **Signed counterfactual training**
   - correct memory: maximize
     `foreign_answer_nll - correct_answer_nll`;
   - paired foreign memory: minimize or reverse that margin;
   - null memory: match no-memory behavior.
5. **Pass criterion**
   - `delta_qv` must beat no-memory and hidden/KV baselines;
   - `delta_qv` must beat shuffled/wrong-query controls;
   - correct memory must show a positive margin advantage over paired foreign
     memory;
   - reports must include address-rank evidence, not only answer NLL.

This is the "对立统一" outcome of the literature read: keep the Transformer
attention path for precise Q/K/V intervention, borrow explicit identity
competition from retrieval/KV memory, borrow anti-interference pressure from
fast-weight/delta-rule memory, and reject pure compressed-state memory as
insufficient for exact binding.

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
