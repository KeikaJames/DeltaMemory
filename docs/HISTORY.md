# DeltaMemory — Phase / stage history (long form)

This file is the long-form narrative log that previously lived in the README.
The README itself now only carries a condensed Phase table; consult this file
for the full per-stage rationale, evidence, and red-line bookkeeping.

For the current canonical architecture (AttnNativeBank + Dynamic LOPI v3.4 +
U-LOPI Phase S + persistence), see [`README.md`](../README.md).
For per-stage code/config diffs, see also [`../CHANGELOG.md`](../CHANGELOG.md).

---

## Phase R+ / Stage 16 era — what the README used to lead with

The pre-restructure README led with the **counter-prior memory injection**
result on Gemma-4-E2B and Qwen3-4B. Those numbers are still valid as a v3.1
snapshot and the raw transcripts remain committed verbatim:

* `transcripts/v31_intervention/gemma-4-e2b-gb10-FALSE/`
* `transcripts/v31_intervention/gemma-4-e2b-mac-FALSE/`
* `transcripts/v31_intervention/qwen3-4b-gb10-FALSE/`
* `transcripts/v31_intervention/qwen3-4b-mac-FALSE/`
* `transcripts/v31_intervention/deepseek-r1-distill-qwen-32b-gb10-FALSE*/`

| Model | Hardware | α | Counter-prior result |
|---|---|---:|---:|
| `google/gemma-4-E2B` | GB10 CUDA bf16 | 1.0 | 5 / 5 positive |
| `google/gemma-4-E2B` | Mac MPS bf16 | 1.0 | 5 / 5 positive |
| `Qwen/Qwen3-4B-Instruct-2507` | GB10 CUDA bf16 | 0.05 | 5 / 5 positive |
| `Qwen/Qwen3-4B-Instruct-2507` | Mac MPS bf16 | 0.05 | 5 / 5 positive |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | GB10 CUDA bf16 | 0.05–0.30 sweep | mixed; stronger 32B prior needs a trained projector |

Per-target Δ = `target_logprob(v3_attn_bank) − target_logprob(B0_no_memory)`:

| Target | Written bank fact | Gemma GB10 | Gemma Mac | Qwen3 GB10 | Qwen3 Mac |
|---|---|---:|---:|---:|---:|
| Napoleon | Paris mayor is Napoleon Bonaparte | +0.586 | +0.729 | +0.764 | +0.543 |
| Pablo | Eiffel Tower architect is Pablo Picasso | +1.376 | +1.511 | +0.244 | +0.496 |
| Vincent | Mona Lisa was painted by Vincent van Gogh | +1.189 | +1.146 | +0.852 | +0.855 |
| Isaac | General relativity was developed by Isaac Newton | +0.698 | +0.757 | +1.047 | +1.010 |
| Ada | Python was created by Ada Lovelace | +2.678 | +2.855 | +1.446 | +1.595 |

Held-out recall context (Gemma-4 dev_v31):

| Condition | recall@1 |
|---|---:|
| B0 no memory | 0.351 |
| v2 raw bank | 0.012 |
| **v3.1 K-projector** | **0.559** |
| B1 prompt insertion | 0.637 |
| B2 RAG oracle | 0.656 |

The v3.1 K-projector strongly lifts the raw bank and can causally move
counter-prior logits, but it has not surpassed the prompt/RAG upper bars on
the full held-out recall benchmark.

### v3.1 per-architecture α defaults

`scripts/run_intervention_demo.py` defaults `--alpha` from
`ArchAdapter.default_alpha`:

| Adapter | Default α | Reason |
|---|---:|---|
| Gemma4Adapter | 1.0 | Gemma-4 applies `v_norm`, so bank V activations are small enough for α=1 |
| Qwen3Adapter | 0.05 | no `v_norm`; α=1 collapses logits |
| LlamaAdapter / Qwen2-family | 0.05 | covers Llama-style and DeepSeek-R1-Distill-Qwen-32B path |
| Glm4Adapter | 0.05 | conservative default for GLM-family attention |

U-LOPI (Phase S) replaces the per-architecture `norm_base` constant with a
one-shot residual profile so the same LOPI runs across these families without
manual α retuning; see [`README.md`](../README.md#u-lopi-phase-s) and
[`../deltamemory/memory/lopi_profiler.py`](../deltamemory/memory/lopi_profiler.py).

### DeepSeek-32B limitation (v3.1 era)

DeepSeek-R1-Distill-Qwen-32B is routed through the Qwen2/Llama-family adapter.
True-fact reinforcement sweet spot is around `α=0.05`, but counter-prior
targets on this 32B model start from much stronger priors. The identity-init
bank improves some targets but does not yet override all five — recorded as
a real limitation, not hidden as a success.

---

## Generation log

| Generation | Mechanism | Trainable surface | LLM weights | Best current reading |
|---|---|---|---|---|
| v1 / Stages 8–12 | external writer, address bank, residual/logit-side paths | writer / projector / LoRA depending on stage | frozen | useful pilots; terminology now deprecated |
| v2 / Stage 13 | raw per-layer K/V bank concatenated into attention | none | frozen | bit-equal locality; chat recall fails without K-space bridge |
| v3 / Stage 14 | v2 + InfoNCE K-projector | bank-side K-projector | frozen | preregistered test negative vs B0; positive vs raw v2 |
| **v3.1 / Stage 15** | attn-native bank + per-arch α + cross-arch adapters | bank-side K-projector only | frozen | counter-prior injection reproduced on Gemma-4 and Qwen3 across GB10/Mac |
| **v3.2 / Stage 16** | v3.1 + mHC spectral shield (Sinkhorn-Knopp on merged attention weights) | bank-side, parameter-free | frozen | bounds σ_max(W) ≤ 1 uniformly in α; cross-flagship sweep in `reports/cleanroom/mhc_flagship_sweep/` |
| **v3.3 / Phase R-3** | v3.2 + Dynamic LOPI (orthogonal novelty + Gaussian layer routing + derivative gate) | none | frozen | 630-cell ablation in `reports/cleanroom/lopi_v33/` |
| **v3.4 / Phase R-3.5** | LOPI defaults flipped: orthogonal=False, gaussian=True, derivative=True | none | frozen | high-α drift collapse + lift preserved at α=1 (see `lopi_v33/AGGREGATE.md`) |
| **v3.4 / Phase R-4** | cross-arch α-safety sweep on Gemma / Qwen3 / GLM-4 | none | frozen | `reports/cleanroom/lopi_v33/R4_xarch/` |
| **v3.4 / Phase R-5.1** | Q3 adversarial chat × LOPI on Gemma-4-E2B | none | frozen | `reports/cleanroom/lopi_v33/R5_q3/` |
| **v3.4 / Phase R-6** | persistent bank: safetensors + filelock + content-addressed dirs | none | frozen | `deltamemory/memory/bank_persistence.py`, `tests/test_bank_persistence.py` |
| **v3.5 / Phase S** | U-LOPI auto-calibration profiler (per-arch Z-score baselines replace hard-coded `norm_base=10.0`) | none | frozen | `deltamemory/memory/lopi_profiler.py`, `tests/test_lopi_profiler.py`, `tests/test_lopi_universal.py` |

---

## Where the v3.1-era figures live

The pre-restructure README embedded SVGs under `docs/figures/v31/`:
architecture, counter-prior lift, Mac-vs-GB10 reproduction, DeepSeek-32B α
sweep, and held-out recall context. They are dependency-free SVGs generated
by `scripts/make_v31_readme_figures.py` from committed JSON artifacts and
remain in-tree for reference.

For a per-stage code/config breakdown beyond what this file captures, see
[`../CHANGELOG.md`](../CHANGELOG.md).
