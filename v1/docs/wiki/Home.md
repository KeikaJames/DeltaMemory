# Mneme Wiki — Home

> **Mneme** is an external, attention-bank memory module for **frozen** transformer LLMs. The base model's weights are never modified. The bank is plugged into each attention layer's softmax and can be written/read at inference time.

---

## TL;DR

| Property | Value |
|---|---|
| LLM weights | **Frozen** (no LoRA, no MEMIT, no fine-tune of base) |
| Trained module | External K-projector (~MB), per-layer Linear init = identity |
| Conservation law | α=0 or empty bank → output is **bit-identical** to the unpatched LLM (max-abs-diff = 0.000 verified on Gemma-4-E2B and Qwen3-4B) |
| Architectures | Gemma-4 / Qwen3 / Llama family (Llama / Qwen2 / Mistral / DeepSeek-R1-Distill) / GLM-4 (via `ArchAdapter`) |
| Storage | Per-layer K/V tensors in a small bank (~`(n_layers × n_heads × head_dim)` per fact) |

## What does it do?

Given an LLM that **does not know** "the mayor of Paris is Anne Hidalgo", Mneme:

1. **Write**: forward the fact `"Fact: The mayor of Paris is Anne Hidalgo."` once, capture each attention layer's K/V at the period token, store in the bank.
2. **Read**: when the LLM later forwards the question `"Q: Who is the mayor of Paris?\nA:"`, the bank's K/V participate in attention's softmax via a single `concat` per layer; the model's next-token distribution shifts toward `"Anne"`.

Per-fact log-prob lift on Gemma-4-E2B (v3 frozen K-projector):

| fact (model didn't know) | B0 logprob | v3 logprob | Δ |
|---|---:|---:|---:|
| f1 mayor of Paris → Anne Hidalgo | −5.05 | **−0.64** | **+4.41** ≈ 80× prob |

See `transcripts/v3_intervention/gemma-4-e2b/demo.md` for the full 5-fact transcript including the cases the model already knew (where v3 stays within ±0.04 of B0 — no pollution).

## How is this different from RAG / MEMIT / LoRA / KV-cache?

| | RAG | MEMIT / ROME | LoRA | Mneme |
|---|---|---|---|---|
| Modifies LLM weights | no | **yes** | **yes** | no |
| Inference-time write | yes (re-prompt) | no (offline edit) | no | **yes (single forward)** |
| Per-layer injection | no (prompt only) | yes (one MLP) | varies | **yes (every attn layer)** |
| Conservation @ off | n/a | n/a | n/a | **bit-equal** |
| Multi-arch support | trivial | per-arch math | per-arch | per-arch via `ArchAdapter` |

## Pages

- [[Architecture]] — patcher / adapter / bank / writer
- [[ArchAdapter]] — how to add a new model family
- [[Reproduction]] — full pipeline from `git clone` to numbers
- [[Benchmark-Results]] — Phase G test, Phase M cross-arch, Phase N intervention
- [[Preregistration]] — frozen v3 spec, Phase G amendments, val-2 protocol
- [[Phase-G-and-G-plus-1]] — honest negative + methodology revision
- [[Roadmap-v3.2]] — what we're not claiming yet, and what's next

## Status

| version | status | dataset | architecture | LLM frozen | trained module |
|---|---|---|---|---|---|
| v1 | shipped | LAMA-curated 183 facts | Gemma-4-E2B | yes | none (single-point residual injection) |
| v2 | shipped | Stage 11 paraphrase | Gemma-4-E2B | yes | none (raw attn-bank) |
| v3 | shipped (frozen) | Stage 14 dev N=33 | Gemma-4-E2B | yes | InfoNCE K-projector |
| v3.1 | in progress | ≥30 relations × ≥60 paraphrases, cross-arch hard negatives | Gemma-4 + Qwen3 + (DeepSeek/Llama pending) | yes | K-projector + writer |

The honest record: Phase G test eval of v3 = 0.278 (Wilcoxon p=0.007 vs B0=0.359; **H1 rejected**, B1 prompt-insertion at 0.658 remains the bar). v3.1 addresses softmax dilution at N≥30 (`bank_topk`), surface-form lexical fingerprinting, and cross-arch generalization.

---

## Negative finding (Exp23–Exp27, 2026 Q2) — Site-Stratified ANB does not scale

A four-experiment attack on **site-stratified, fact-routed memory** —
relation-site K capture + subject/object-site V capture, then native
sparse-attention readout over a bank of N facts — was carried out on
Qwen3-4B (MPS bf16, CounterFact) in
`experiments/atb_validation_v1/exp13_anb_readdressability/`. All four
attacks produced the **identical N=100 PASS → N=200 FAIL curve**.

| Attack | Architecture | N=100 gates | N=200 gates |
|---|---|---|---|
| Exp24 K-routing | additive readout, single K site | DIRECTIONAL +0.193 nat | weak |
| Exp26 single-V  | K=relation_last, V=object_last (1 tok) | A+C+D PASS_STRONG | all FAIL |
| Exp26b multi-V  | K=relation_last, V=subj→obj (~8 tok) | A+C+D PASS | all FAIL |
| Exp27 sparse-attn | joint softmax `Attn(Q,[K;M_K],[V;M_V])`, α∈{0.05..3.0} | C+D PASS only at α=0.05 | all FAIL |

`retrieval_accuracy` never escapes 2–3× chance at N=100 and decays to
~1× chance at N≥200, **independent of** K site, V site, V span length,
α (4 orders of magnitude), and joint-vs-additive softmax. At α≥1.0 the
joint softmax actively downweights the bank (`bank_mass` 0.34 → 0.13).

**Conclusion**: native attention traces are re-addressable at small
scale, but the pre-RoPE K-space of Qwen3-4B does not contain enough
query-key discriminability to route a 200-fact bank by cosine on raw
`q·M_K^T`. The N=100 PASS_STRONG signals reported in earlier verdicts
are real **steering** (Gate A) but not **routed memory** (Gate B). The
v3 single-fact lift table above (f1 +4.41 nat on Gemma-4-E2B) is
**unaffected** — that is per-fact intervention, not bank-scale routing
— but should be read as a *steering* result, not as evidence of
sparse-routed memory at bank-size N≥100.

Conservation guarantees of the prototype (α=0 bit-equality, frozen base
weights) are unaffected. Detailed verdicts:
`EXP25_VERDICT.md`, `EXP26_VERDICT.md`, `EXP26b_VERDICT.md`,
`EXP27_VERDICT.md`, `EXP27_SPARSE_VERDICT.md` under
`experiments/atb_validation_v1/exp13_anb_readdressability/`.

### Cross-architecture replication (2026-05-13)

The Qwen-only N=100→N=200 falsification was replicated on
**Gemma-4-E2B** and **Mistral-7B-Instruct-v0.3** under the same
infrastructure (per-architecture α grid rescaled by native V norm).
Pattern holds across all three transformer families:

| Architecture | Gate A peak N=100 | retr_acc peak | N=200 collapse |
|---|---:|---:|---|
| Qwen3-4B-Instruct | +0.447 | 1.89× chance | ≤+0.05 |
| Gemma-4-E2B | +0.033 | 3.11× chance | ≤+0.012 |
| Mistral-7B-Instruct-v0.3 | ≤0 | **10× chance** | (no co-tracking lift) |

The Qwen N=100 PASS_STRONG result is an **outlier in magnitude**, not
in mechanism: gate sign and α-dependence carry over identically.
Mistral has the strongest **trace-level routing signal** of any tested
arch (retr_acc 10× chance) but still no content-mediated lift. Full
report: `EXP_CROSS_ARCH_VERDICT.md`.

### Architectural ceiling — Exp31 + Exp32 double-negative (2026-05-13)

Two follow-up experiments each picked one orthogonal hypothesis to
revive cosine-routed bank memory. Both rejected with the same signature.

| Hypothesis | Lever pulled | Embedding val top-1 | LM-output Gate B | Gate E shuffled-pair control |
|---|---|---:|---:|---|
| **H_A — K-space discriminability bottleneck (Exp31)** | per-layer Linear K-adapter trained with InfoNCE | ~40× chance | 0 / 375 | FAIL — shuffled adapter beats trained |
| **H_B — attention is the wrong site (Exp32)** | inject at MLP output, capture K=MLP-input / V=MLP-output, learned softmax gate | **~106× chance** | 0 / 375 | FAIL by **−1.42 logits** |

Both adapters become highly discriminative in their own embedding space
but contribute zero LM-output identity coupling. The failure mode is
**the α-scaled residual readout protocol itself** — `h ← h + α·readout(bank)`
at any sublayer produces detectable activation drift but no fact-identity
binding at the logit head, regardless of read site or routing capacity.

Cross-arch replication of Exp31/Exp32 was *not* run: Qwen3 produced no
LM-output signal, and Gemma/Mistral would only reproduce the same null.
Verdicts: `EXP31_VERDICT.md`, `EXP32_VERDICT.md` under
`experiments/atb_validation_v1/exp31_learned_k_adapter/` and
`experiments/atb_validation_v1/exp32_mlp_side_gated_memory/`.
