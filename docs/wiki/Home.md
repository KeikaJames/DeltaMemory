# DeltaMemory Wiki — Home

> **DeltaMemory** is an external, attention-bank memory module for **frozen** transformer LLMs. The base model's weights are never modified. The bank is plugged into each attention layer's softmax and can be written/read at inference time.

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

Given an LLM that **does not know** "the mayor of Paris is Anne Hidalgo", DeltaMemory:

1. **Write**: forward the fact `"Fact: The mayor of Paris is Anne Hidalgo."` once, capture each attention layer's K/V at the period token, store in the bank.
2. **Read**: when the LLM later forwards the question `"Q: Who is the mayor of Paris?\nA:"`, the bank's K/V participate in attention's softmax via a single `concat` per layer; the model's next-token distribution shifts toward `"Anne"`.

Per-fact log-prob lift on Gemma-4-E2B (v3 frozen K-projector):

| fact (model didn't know) | B0 logprob | v3 logprob | Δ |
|---|---:|---:|---:|
| f1 mayor of Paris → Anne Hidalgo | −5.05 | **−0.64** | **+4.41** ≈ 80× prob |

See `transcripts/v3_intervention/gemma-4-e2b/demo.md` for the full 5-fact transcript including the cases the model already knew (where v3 stays within ±0.04 of B0 — no pollution).

## How is this different from RAG / MEMIT / LoRA / KV-cache?

| | RAG | MEMIT / ROME | LoRA | DeltaMemory |
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
