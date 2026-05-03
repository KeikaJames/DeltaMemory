<p align="center">
  <h1 align="center">DeltaMemory</h1>
</p>

<p align="center">
  <strong>External K/V memory injected inside frozen Transformer attention.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA-EE4C2C.svg">
  <img alt="Hardware" src="https://img.shields.io/badge/Apple%20MPS%20%7C%20GB10%20CUDA-bf16-555.svg">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-orange.svg">
</p>

<p align="center">
  <strong>Languages:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <a href="docs/design.md">Design</a> ·
  <a href="docs/apple_silicon.md">Apple Silicon</a> ·
  <a href="transcripts/v31_intervention/CROSS_ARCH_REPORT.md">v3.1 cross-arch report</a> ·
  <a href="reports/cleanroom">Reports</a>
</p>

---

DeltaMemory is a research prototype for **persistent external memory in a
frozen LLM**. The mechanism is **DeltaMemory attn-native bank**.

It is **not RAG**, **not prompt insertion**, and **not a weight edit**. During
the read pass, the prompt contains only the question. The value is supplied by
a per-layer external K/V bank that is concatenated into supported attention
layers. The base LLM weights remain frozen.

## Current headline: counter-prior memory injection

The strongest test is not to reinforce a fact the model already knows. The
strongest test is to write a **false fact** into the bank, ask the corresponding
question, and measure whether the frozen model raises the log-probability of
the counter-prior target.

That is now reproduced on two model families and two hardware backends:

| Model | Hardware | α | Counter-prior result |
|---|---|---:|---:|
| `google/gemma-4-E2B` | GB10 CUDA bf16 | 1.0 | **5 / 5 positive** |
| `google/gemma-4-E2B` | Mac MPS bf16 | 1.0 | **5 / 5 positive** |
| `Qwen/Qwen3-4B-Instruct-2507` | GB10 CUDA bf16 | 0.05 | **5 / 5 positive** |
| `Qwen/Qwen3-4B-Instruct-2507` | Mac MPS bf16 | 0.05 | **5 / 5 positive** |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | GB10 CUDA bf16 | 0.05–0.30 sweep | mixed; stronger 32B prior needs a trained projector |

<p align="center"><img src="docs/figures/v31/v31_false_fact_lift.svg" alt="Counter-prior target log-prob lift" width="860"></p>

The cleanest single example is:

```text
write into bank:  Fact: Python was created by Ada Lovelace.
read prompt:      Q: Who created the Python programming language?
                  A:
target token:     " Ada"
```

On Gemma-4-E2B, the no-memory model puts `" Ada"` at about `-12` nats. With the
bank attached and the LLM weights unchanged, `" Ada"` rises by **+2.68 nats on
GB10 CUDA** and **+2.86 nats on Mac MPS**.

## Mechanism

DeltaMemory v3.1 injects memory directly into every supported attention layer:

$$
\mathrm{Attn}_\ell\bigl(Q,\; [K\,;\, M_K^{(\ell)}],\; [V\,;\, \alpha M_V^{(\ell)}]\bigr)
$$

`M_K` and `M_V` are captured from a write forward pass. At read time they are
merged into the frozen model's attention computation. The only trainable
surface in the v3 family is the bank-side K-projector; it does not alter the
LLM's token path and `α=0` / empty bank remains bit-equal to the base model.

<p align="center"><img src="docs/figures/v31/v31_architecture.svg" alt="DeltaMemory v3.1 architecture" width="860"></p>

## Evidence table

All numbers below are `Δ = target_logprob(v3_attn_bank) − target_logprob(B0_no_memory)`.
The target is deliberately the answer contradicted by the base model prior.

| Target | Written bank fact | Gemma GB10 | Gemma Mac | Qwen3 GB10 | Qwen3 Mac |
|---|---|---:|---:|---:|---:|
| Napoleon | Paris mayor is Napoleon Bonaparte | +0.586 | +0.729 | +0.764 | +0.543 |
| Pablo | Eiffel Tower architect is Pablo Picasso | +1.376 | +1.511 | +0.244 | +0.496 |
| Vincent | Mona Lisa was painted by Vincent van Gogh | +1.189 | +1.146 | +0.852 | +0.855 |
| Isaac | General relativity was developed by Isaac Newton | +0.698 | +0.757 | +1.047 | +1.010 |
| Ada | Python was created by Ada Lovelace | +2.678 | +2.855 | +1.446 | +1.595 |

<p align="center"><img src="docs/figures/v31/v31_cross_hardware.svg" alt="GB10 CUDA and Mac MPS reproduce the same counter-prior effect" width="860"></p>

The raw model inputs, top-5 predictions, and target log-probabilities are
committed verbatim:

- `transcripts/v31_intervention/gemma-4-e2b-gb10-FALSE/`
- `transcripts/v31_intervention/gemma-4-e2b-mac-FALSE/`
- `transcripts/v31_intervention/qwen3-4b-gb10-FALSE/`
- `transcripts/v31_intervention/qwen3-4b-mac-FALSE/`
- `transcripts/v31_intervention/deepseek-r1-distill-qwen-32b-gb10-FALSE*/`

## DeepSeek-32B limitation

DeepSeek-R1-Distill-Qwen-32B is routed through the Qwen2/Llama-family adapter.
The true-fact reinforcement sweet spot is around `α=0.05`, but counter-prior
targets on this 32B model start from much stronger priors. The identity-init
bank improves some targets but does not yet override all five.

<p align="center"><img src="docs/figures/v31/v31_deepseek_alpha.svg" alt="DeepSeek-32B counter-prior alpha sweep" width="860"></p>

This is recorded as a real limitation of the identity-init bank, not hidden as
a success. The next research step is a trained K-projector for Qwen2/DeepSeek
counter-prior override.

## Recall context

The counter-prior test proves causal injection. The held-out recall benchmark
shows the broader state of v3.1 on Gemma-4 dev_v31:

<p align="center"><img src="docs/figures/v31/v31_recall_context.svg" alt="Gemma-4 dev_v31 held-out recall context" width="860"></p>

| Condition | recall@1 |
|---|---:|
| B0 no memory | 0.351 |
| v2 raw bank | 0.012 |
| **v3.1 K-projector** | **0.559** |
| B1 prompt insertion | 0.637 |
| B2 RAG oracle | 0.656 |

The result is deliberately framed narrowly: v3.1 strongly lifts the raw bank
and can causally move counter-prior logits, but it has not yet surpassed the
prompt/RAG upper bars on the full held-out recall benchmark.

## Generation history

| Generation | Mechanism | Trainable surface | LLM weights | Best current reading |
|---|---|---|---|---|
| v1 / Stages 8–12 | external writer, address bank, residual/logit-side paths | writer / projector / LoRA depending on stage | frozen | useful pilots; terminology now deprecated |
| v2 / Stage 13 | raw per-layer K/V bank concatenated into attention | none | frozen | bit-equal locality; chat recall fails without K-space bridge |
| v3 / Stage 14 | v2 + InfoNCE K-projector | bank-side K-projector | frozen | preregistered test negative vs B0; positive vs raw v2 |
| **v3.1 / Stage 15** | attn-native bank + per-arch α + cross-arch adapters | bank-side K-projector only | frozen | counter-prior injection reproduced on Gemma-4 and Qwen3 across GB10/Mac |

## Per-architecture α defaults

`scripts/run_intervention_demo.py` now defaults `--alpha` from
`ArchAdapter.default_alpha`:

| Adapter | Default α | Reason |
|---|---:|---|
| Gemma4Adapter | 1.0 | Gemma-4 applies `v_norm`, so bank V activations are small enough for α=1 |
| Qwen3Adapter | 0.05 | no `v_norm`; α=1 collapses logits |
| LlamaAdapter / Qwen2-family | 0.05 | covers Llama-style and DeepSeek-R1-Distill-Qwen-32B path |
| Glm4Adapter | 0.05 | conservative default for GLM-family attention |

## Reproduce the README figures

The README charts are generated from committed JSON artifacts:

```bash
python3 scripts/make_v31_readme_figures.py
```

Run the intervention demo:

```bash
# Gemma-4, default α=1.0
python scripts/run_intervention_demo.py \
  --model google/gemma-4-E2B \
  --device cuda \
  --dtype bfloat16 \
  --false-facts

# Qwen3, default α=0.05
python scripts/run_intervention_demo.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --dtype bfloat16 \
  --false-facts
```

On Apple Silicon, use the stable MPS stack documented in
[`docs/apple_silicon.md`](docs/apple_silicon.md).

## Repository map

| Path | Purpose |
|---|---|
| `deltamemory/` | library code, attention-bank patcher, architecture adapters |
| `scripts/run_intervention_demo.py` | cross-architecture true/false-fact intervention demo |
| `scripts/make_v31_readme_figures.py` | dependency-free SVG generator for README charts |
| `transcripts/v31_intervention/` | raw inputs, outputs, top-5 predictions, target log-probs |
| `reports/cleanroom/` | preregistered and cleanroom experiment reports |
| `docs/figures/v31/` | current README figure set |
| `tests/` | unit and real-model conservation checks |

## License

MIT. See [`LICENSE`](LICENSE).
