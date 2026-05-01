# RCV-HC

RCV-HC is a cleanroom prototype for **external attention memory on frozen
Gemma-style decoder language models**.

The current research path is:

```text
long context
-> frozen Gemma forward
-> per-block Raw/Delta attention memory
-> external CPU/disk store
-> top-k memory retrieval
-> Gemma Q/K/V attention intervention
-> answer metrics and attention-memory trace
```

It is not a RAG prompt-insertion system. Retrieved `source_text` is kept only as
debug metadata and is not appended to the answer prompt in the RCV-HC path.

## Delta Memory

The core memory signal is the original DeltaMemory idea:

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

For attention-path intervention, Delta is projected into Q/K/V residuals:

```text
q' = q + alpha_q * gate_q * P_q(Delta[b])
k' = k + alpha_k * gate_k * P_k(Delta[b])
v' = v + alpha_v * gate_v * P_v(Delta[b])
```

The base Gemma model is frozen. Only the RCV-HC writer, gates, and Q/K/V
projection adapters are trainable.

## Quick Start

Install dependencies:

```bash
python3 -m venv .venv-mac
.venv-mac/bin/python -m pip install -r requirements-macos.txt
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

Train only the RCV-HC Delta Q/V adapter:

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

Tracked cleanroom evidence:

- `reports/cleanroom/gemma4_real/gemma4_prototype_report.md`
- `reports/cleanroom/gemma4_training_probe/delta_training_report.md`
- `reports/cleanroom/gemma4_delta_experiment_main/delta_experiment_report.md`
- `reports/cleanroom/gemma4_delta_experiment_mps_scaled/report.md`

The current real Gemma4 run shows engineering success:

- base model frozen,
- `trainable_base_params = 0`,
- Q/V Delta intervention has non-zero norm,
- zero/random/shuffled/force-gate controls run,
- no retrieved source text is inserted into the prompt.

It also shows the current scientific boundary:

- the tracked run is `wiring_signal_only`,
- therefore it is not an effectiveness claim.

The training probe is stronger but still intentionally small:

- `google/gemma-4-E2B`,
- `steps = 1`,
- `top_k = 2`,
- `trainable_base_params = 0`,
- trained `delta_qv` improves NLL from `11.8004` to `11.3691`,
- trained `delta_qv` beats zero, random, and shuffled controls on this single
  demo instance.

This is a mechanism signal, not a benchmark result. The next step is to repeat
the same controlled training on more generated examples.

The current multi-example Gemma4 probe repeats that test on generated
later-reference examples:

- `train_samples = 4`,
- `eval_samples = 4`,
- `steps = 2`,
- `block_size = 64`,
- `trainable_base_params = 0`,
- held-out `delta_qv` improves NLL from `12.2486` to `11.4252`,
- held-out `delta_qv` beats zero, random, and shuffled controls.

The scaled MPS run strengthens this mechanism signal:

- `model = google/gemma-4-E2B`,
- `device = mps`,
- `dtype = bfloat16`,
- `seeds = [0, 1, 2]`,
- `train_samples = 8`,
- `eval_samples = 8`,
- `steps = 8`,
- `trainable_base_params = 0`,
- mean held-out `delta_qv` NLL is `8.3279`,
- mean no-memory NLL is `12.3188`,
- trained `delta_qv` beats zero, random, and shuffled controls on all seeds.

This is now a controlled small-scale mechanism result on a real Gemma4 base.
It is still not a broad benchmark result.

## Proof Plan

The project should prove the mechanism in this order:

1. **Wiring proof**: frozen Gemma accepts external attention memory and Q/V
   Delta injection; zero/random/shuffled controls run.
2. **Optimization proof**: train only RCV-HC adapters and show trained
   `delta_qv` improves answer-token NLL/rank while base parameters remain
   frozen.
3. **Alignment proof**: trained `delta_qv` must beat `delta_qv_zero`,
   `delta_qv_random`, and `delta_qv_shuffled`.
4. **Storage proof**: old blocks outside the active local window are retrieved
   from external storage and affect Q/K/V without prompt insertion.
5. **External baseline comparison**: only after the above passes, compare against
   a carefully implemented external baseline under the same frozen model,
   trainable-parameter budget, and data.

## Repository Layout

```text
rcvhc/core/       config and shared typed records
rcvhc/memory/     external attention-memory store and writer
rcvhc/gemma/      Gemma-style model adapter and Q/K/V injector
rcvhc/engine/     ingest, ask, prototype, and adapter-training runners
scripts/          runnable demos
docs/             design and evidence boundary
reports/          minimal tracked cleanroom reports only
tests/            cleanroom tests; no Gemma download required
```

Ignored local artifacts include `runs/`, `stores/`, `cache/`, checkpoints,
model weights, JSONL logs, virtualenvs, and pycache files.
