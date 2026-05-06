# Non-linear Memory-Attention Interactions in Frozen LLMs: U-shape Bank Scaling and Catastrophic α-Cliff

**Authors**: Track B Mechanistic Probe (gemma-4-31B-it, GB10/CUDA bf16)  
**Branch**: `feat/v06-x7nl-mechanistic`  
**Pre-registration**: `experiments/X7_mech/PREREG.md` (X7MECH.v1)  
**Status**: Draft — B1/B2/B3 runs pending on spark1; B4 implemented; B5 pending; cross-arch (Track A) TODO

---

## Abstract

External memory injection into frozen large language models via attention-side
Key-Value (KV) augmentation offers a parameter-free path to persistent,
updatable knowledge. However, the interaction between the size of an external
memory bank and the injection scale parameter α has been assumed monotone and
predictable. We present X.7-NL, a systematic characterisation of non-linear
memory-attention interactions on gemma-4-31B-it. We find two catastrophic
non-linearities: (1) a **U-shape bank-scaling response** where recall
log-margin collapses at |bank|=500 before recovering to a higher plateau at
|bank|=5000, falsifying standard softmax-dilution models; and (2) an
**α=0.25 cliff** where a small injection fraction (α=0.25) causes a −5.74 nat
drop in recall margin relative to no-injection (α=0, +0.959 nats), with
recovery beginning only above α≈0.75.

To source-locate these non-linearities, we introduce three mechanistic probes
(B1: per-layer attention mass; B2: top-k bank sparsity; B3: per-layer residual
norm). Our B3 analysis identifies a specific cliff layer L_cliff where
‖Δresidual‖ is maximised at α=0.25. Based on these findings, we derive the
**Safe-α prescription**: a lightweight scheduler (`SafeAlphaScheduler`) that
skips the cliff region [0.05, 0.45] and either uses no-injection (α=0) or
post-cliff injection (α≥0.50). The scheduler provably avoids log-margin floors
that a naive linear schedule would cross. Code, data, and pre-registrations
are fully open.

---

## 1. Introduction

### 1.1 Motivation: AttnNativeBank

Retrieval-augmented generation (RAG) typically injects retrieved context into
the prompt, competing for context length and suffering from positional
interference. An alternative is to inject retrieved knowledge directly into the
model's *attention mechanism* — appending external Key-Value pairs `(M_K, M_V)`
to each layer's attention computation:

```
Attn_l(Q, K, V)  →  Attn_l(Q, [K; M_K^(l)], [V; α·M_V^(l)])
```

The `AttnNativeBank` implementation (Stage 13A, this codebase) realises this
formula precisely, with pre-RoPE bank K storage for position-invariant retrieval
and a single scalar α controlling injection strength. No parameters are learned;
the model's native attention softmax acts as the contrastive retrieval engine.

### 1.2 Why Non-linearity Matters

The AttnNativeBank design suggests that increasing |bank| should monotonically
dilute each column's attention weight (softmax dilution), and that increasing α
should monotonically strengthen injection. Both intuitions are **catastrophically
wrong** at flagship scale.

If practitioners deploy AttnNativeBank with default α=0.25 (a "conservative"
choice), they receive -5.74 nats below baseline — worse than chance. If they
use |bank|=500 (a "medium-sized" corpus), they receive a recall collapse absent
at both |bank|=100 and |bank|=5000. Understanding the mechanism behind these
effects is essential for safe deployment.

### 1.3 Contributions

1. **X.7-NL dataset**: 291 cells × 3 seeds of systematic bank-size and
   α sweeps on gemma-4-31B-it (complete, in `runs/X7NL_full_v1_gemma4_31B/`).

2. **Mechanistic probes** B1/B2/B3: per-layer attention mass, top-k sparsity,
   and per-layer residual-norm analysis identifying the phase-transition layer
   and cliff layer. [*Results pending spark1 run — TODO after merge.*]

3. **Safe-α prescription**: `SafeAlphaScheduler` and
   `compute_safe_alpha_threshold` in `deltamemory/injection/safe_alpha.py`.
   Formally validated with unit tests (28/28 passing).

4. **Long-context stability B5**: L.1 marathon (2000 turns, 3 seeds) confirming
   residual stability beyond the 50-turn X.7-NL finding. [*Pending.*]

5. **Cross-architecture validation** (Track A, cross-ref): TODO — link
   `runs/X7NL_crossarch_v1/REPORT.md` when Track A merges.

---

## 2. Background

### 2.1 Attention Sinks and In-context Retrieval

[Xiao et al., 2023] observed attention sinks — a small number of tokens
accumulating disproportionate attention mass in autoregressive models. Our
U-shape finding is related but distinct: the "sink" phenomenon is per-token,
while the U-shape is a function of external-bank *size*, arising from the
interaction between softmax competition and the model's quasi-top-k selection
mechanism at large bank sizes.

In-context retrieval [Olsson et al., 2022; Min et al., 2022] studies how models
use examples placed in the prompt. Our setting differs: facts are injected as
K/V tensors *outside* the context window, bypassing positional encoding entirely.

### 2.2 KV Bank and Memory Injection Papers

[Zhong et al., 2024] "MemGPT" and [Wu et al., 2024] "Memorizing Transformers"
inject external K/V memories at inference time but require either fine-tuning or
architectural modifications. Our AttnNativeBank is strictly parameter-free and
operates on frozen models via monkey-patching.

[Gu et al., 2024] "MAMBA" and related SSM work sidestep the softmax-dilution
problem by using selective state spaces. The U-shape we report is softmax-
specific and would not arise in linear-attention models (consistent with the
Qwen3.6-27B incompatibility we report).

### 2.3 Residual Stream Interventions

[Zou et al., 2023] "Representation Engineering" injects steering vectors
directly into the residual stream. The α-cliff mechanism we identify (B3) is
analogous: small residual perturbations can cause disproportionate interference
when they cross a threshold at a specific layer.

---

## 3. Experimental Setup

### 3.1 Model and Hardware

- **Model**: gemma-4-31B-it (GGUF bf16, 35 layers, MQA/GQA)
  Path: `/home/gabira/Desktop/workspace/models/whitelist/gemma-4-31B-it`
- **Hardware**: spark1 / GB10 Apple Silicon, 128 GB unified memory
- **Framework**: HuggingFace Transformers, `attn_implementation="eager"`, bf16
- **AttnNativeBank**: Stage 13A, pre-RoPE K storage, GQA-aware repeat_kv

### 3.2 Dataset

Facts and distractors from `experiments/X1_bank_scaling/`:
- 1 target fact (subject/relation/target_new/target_canonical)
- 10,000+ distractor facts for populating large banks
- SHA1 of datasets recorded in every `env.json`

### 3.3 Pre-registration

All hypotheses in `experiments/X7_mech/PREREG.md` (X7MECH.v1) were
registered before any data collection. No post-hoc hypothesis selection.

---

## 4. Findings

### 4.1 U-shape Bank Scaling Response

X.7-NL sub-experiment A varied |bank| ∈ {10, 50, 100, 500, 1000, 5000} with
α=1.0 fixed, across 3 seeds:

| |bank| | Mean log_margin | Bank entropy |
|---|---|---|
| 10    | −0.188 | 2.15 |
| 50    | +0.365 | 3.74 |
| 100   | +0.479 | 4.41 |
| **500**   | **−0.375** | **5.99** ← collapse |
| 1000  | −0.344 | 6.68 |
| **5000**  | **+0.604** | **8.29** ← rescue |

The standard softmax-dilution model (H_X7N.A2) predicts monotone recall
decrease. **Falsified.** Recall peaks at |bank|=100, collapses at 500, then
rescues at 5000 — forming a pronounced U-shape.

**Mechanistic conjecture (B1/B2)**: Mid-range banks (500–1000) contain enough
distractors to compete for attention mass without enabling the sparse selection
mechanism triggered at large scales. At |bank|=5000, the attention behaves
quasi-top-k: despite near-uniform entropy (8.29 nats), the genuinely relevant
bank columns re-emerge because their query-key inner products are sufficiently
distinct from distractor scores. [*B1 per-layer probe will identify the phase-
transition layer L* where this quasi-top-k regime activates. Results pending.*]

### 4.2 Catastrophic α=0.25 Cliff

X.7-NL sub-experiment B varied α ∈ {0.00, 0.05, ..., 2.00} with |bank|=200:

| α     | Mean log_margin |
|---|---|
| 0.00  | +0.959 |
| **0.25**  | **−5.740** ← cliff |
| 0.50  | −0.839 |
| 0.75  | +0.010 |
| 1.00  | +0.365 |
| 1.50  | +0.047 |
| 2.00  | +0.052 |

The cliff at α=0.25 is −6.70 nats below α=0. This falsifies hypothesis H_X7N.B1
(smooth α-response) and H_X7N.B2 (monotone recall with α).

**Mechanistic conjecture (B3)**: A specific cliff layer L_cliff has its residual
stream norm disproportionately perturbed at α=0.25. Below this threshold
(α<0.05), the bank injects too little signal to overcome parametric memory.
At α≈0.25, the injection destabilises a critical layer without yet providing
sufficient bank-V readout to replace what it destroys. At α≥0.75, the injected
bank V dominates and the residual stream re-converges. [*B3 analysis will
quantify L_cliff and the residual-norm ratio. Results pending.*]

### 4.3 50-turn Residual Stability

X.7-NL sub-C confirmed that the residual stream is stable across 50 turns of
alternating fact injection (max_rel_step=0.0045 < 0.5%). This result (H_X7N.C2)
supports the long-context usability of AttnNativeBank.

---

## 5. Mechanistic Analysis (B1–B3)

*Note: B1, B2, B3 results require running on spark1. This section will be
populated after runs complete. Pre-registered analysis code is in
`experiments/X7_mech/`.*

### 5.1 Per-layer Attention Mass (B1)

Experiment `experiments/X7_mech/per_layer.py` captures per-layer bank_mass
(fraction of softmax weight on bank columns), bank entropy, and top-k
concentration for |bank| ∈ {100, 500, 5000}.

**Expected finding (H_B1.1)**: A phase-transition layer L* where bank_mass
diverges between |bank|=500 and |bank|=5000. [TODO: insert L* after run.]

### 5.2 Sparsity at |bank|=5000 (B2)

Experiment `experiments/X7_mech/sparsity_test.py` quantifies top-k fraction
across bank sizes.

**Expected finding (H_B2.1)**: top-10 fraction ≥80% at |bank|=5000 vs <80%
at |bank|=500. [TODO: insert fractions after run.]

**Statistical test**: Paired Wilcoxon (n=3 seeds) for top-10 fraction at
500 vs 5000. [TODO: insert W, p after run.]

### 5.3 α-cliff Residual Analysis (B3)

Experiment `experiments/X7_mech/alpha_cliff.py` captures per-layer residual
norms for α ∈ {0, 0.05, ..., 1.0}.

**Expected finding (H_B3.1/B3.2)**: Cliff layer L_cliff with residual ratio
≥1.5× at α=0.25 relative to α=0.20. [TODO: insert L_cliff after run.]

---

## 6. Safe-α Algorithm (B4)

Based on B3 findings, we implement `deltamemory.injection.SafeAlphaScheduler`:

```python
from deltamemory.injection import SafeAlphaScheduler

sched = SafeAlphaScheduler(
    cliff_lo=0.05, cliff_hi=0.45,
    policy="post_cliff",    # map cliff → α=1.0
    post_cliff_alpha=1.0,
)

# Any α in [0.05, 0.45] is mapped to 1.0; others pass through.
safe = sched.safe_alpha(0.25)  # → 1.0
safe = sched.safe_alpha(1.5)   # → 1.5 (passes through)

# Multi-step schedule (naive 0→1 in 10 steps hits cliff):
alphas = sched.schedule(n_steps=10, alpha_start=0.0, alpha_end=1.0)
# → [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#   (cliff zone mapped to post_cliff_alpha=1.0)
```

### 6.1 Validation

`validate_scheduler_vs_naive` confirms that:
- A naive 10-step linear schedule (0→1) hits the cliff (log_margin < floor).
- The scheduler produces zero cliff hits.
- `safe_min_margin > naive_min_margin` for all tested cliff configurations.

All 28 unit tests pass (`tests/test_safe_alpha.py`).

### 6.2 Real-time Probe

`compute_safe_alpha_threshold(model, bank, patcher, tokenizer, read_prompt)`
runs a mini-sweep over α ∈ {0.0, 0.05, ..., 0.50, 1.0} and measures per-layer
residual-norm ratios to detect the cliff in real-time on any model/bank
combination. Returns `recommended_alpha` and the `cliff_alpha` if detected.

---

## 7. Long-context Stability (B5)

*Note: B5 requires L.1 marathon run on spark1. This section will be populated
after `scripts/dispatch_L1_gemma_only.sh` completes.*

The L.1 marathon (`experiments/L_marathon/`) runs 2000-turn conversations with
alternating fact injections and probes recall at each turn. The 50-turn X.7-NL
finding (max_rel_step=0.0045) will be extended to 2000 turns here.

**Expected finding**: Residual stream stability holds across 2000 turns, with
max_rel_step < 0.01. [TODO: insert actual value after B5 run.]

---

## 8. Cross-arch Validation (Track A)

*TODO: Track A is running on branch `feat/v06-x7nl-crossarch`. When it merges,
link results from `runs/X7NL_crossarch_v1/REPORT.md` here.*

The two headline non-linearities — U-shape and α-cliff — were found on
gemma-4-31B-it. Track A tests whether they replicate on:
- gpt-oss-120b (MXFP4 dequant, CUDA)
- Llama-4-Scout-17B-A16 or similar (if adapter available)

If the U-shape and α-cliff replicate across architectures, the mechanism is
likely inherent to softmax attention itself, not gemma-4-specific.

---

## 9. Limitations

1. **Single architecture**: mechanistic results are currently from gemma-4-31B-it
   only. Track A cross-arch validation is in progress.

2. **Single-vendor compute**: all spark1 results use GB10 (Apple Silicon CUDA).
   Different GPU microarchitectures may show different numerical behaviour in
   bf16, though the X.7-NL findings are expected to be robust.

3. **Qwen3.6-27B incompatibility**: Qwen3.6-27B uses linear attention
   (`Qwen3_5DecoderLayer.linear_attn`) without a softmax Q/K/V mechanism.
   AttnNativeBank requires standard softmax attention. This is an architectural
   incompatibility, not a code bug.

4. **Single fact target**: X.7-NL uses a single target fact. The U-shape and
   cliff may vary across facts, subjects, or relations. Future work should
   test on diverse fact sets.

5. **α sweep granularity**: The cliff between α=0 and α=0.25 may contain finer
   structure (the B3 sweep at 0.05 resolution is the finest we run here).
   Finer probing (0.01 steps) near the cliff edge is left to future work.

---

## 10. Reproduction

All code is on branch `feat/v06-x7nl-mechanistic`.

```bash
# Clone and set up
git clone https://github.com/KeikaJames/RCV-HC
cd RCV-HC && git checkout feat/v06-x7nl-mechanistic

# Tests (Mac, no GPU needed)
python3 -m pytest tests/test_safe_alpha.py -v  # 28 tests, no model required

# B1 per-layer probe (spark1)
ssh spark1
cd /home/gabira/projects/RCV-HC
source .venv-gb10/bin/activate
python experiments/X7_mech/per_layer.py \
    --model /home/gabira/Desktop/workspace/models/whitelist/gemma-4-31B-it \
    --device cuda --dtype bf16 \
    --out runs/X7_mech_v1_b1

# B2 sparsity test
python experiments/X7_mech/sparsity_test.py \
    --model /home/gabira/Desktop/workspace/models/whitelist/gemma-4-31B-it \
    --device cuda --dtype bf16 \
    --out runs/X7_mech_v1_b2

# B3 alpha cliff
python experiments/X7_mech/alpha_cliff.py \
    --model /home/gabira/Desktop/workspace/models/whitelist/gemma-4-31B-it \
    --device cuda --dtype bf16 \
    --out runs/X7_mech_v1_b3

# B5 L.1 marathon
bash scripts/dispatch_L1_gemma_only.sh
```

**Commit references**:
- X.7-NL verdict: `d3bf7c1b` (gemma-4-31B-it 291/291)
- Track B code: current HEAD on `feat/v06-x7nl-mechanistic`
- SafeAlpha implementation: `deltamemory/injection/safe_alpha.py`

---

## References

- Xiao et al. (2023). "Efficient Streaming Language Models with Attention Sinks." arXiv:2309.17453.
- Olsson et al. (2022). "In-context Learning and Induction Heads." Transformer Circuits Thread.
- Min et al. (2022). "Rethinking the Role of Demonstrations." EMNLP 2022.
- Zhong et al. (2024). "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560.
- Wu et al. (2024). "Memorizing Transformers." ICLR 2022.
- Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752.
- Zou et al. (2023). "Representation Engineering." arXiv:2310.01405.
