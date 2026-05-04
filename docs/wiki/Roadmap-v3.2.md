# Roadmap — v3.2 and beyond

This page is the running list of what we are *not* claiming yet, and how we plan to get there.

If you find yourself wanting to use Mneme for one of the items below, be aware that the work has not been done, and the result might still be a Phase-G-style honest negative.

---

## Currently in progress (v3.1, 2026-Q1)

These are tracked in `docs/preregistration.md` Stage 15 amendments and the in-repo todo list. The bullets below summarize; see [[Phase-G-and-G-plus-1]] for the full story.

- [x] `bank_topk` softmax-dilution fix (commit `0036f62`)
- [x] `ArchAdapter` for Gemma-4 / Qwen3 / Llama / GLM-4 (`0036f62`)
- [x] Conservation regression on Gemma-4-E2B (Mac + GB10) and Qwen3-4B (GB10) → all `bit_equal = True`
- [x] Phase N intervention demo on Gemma-4-E2B (+4.41 logprob lift on Paris/Anne)
- [ ] Phase L1 — ≥2,500 (write, paraphrase) pairs across ≥30 relations + cross-relation hard negs
- [ ] Phase L2 — train K-projector on GB10 with cross-architecture batch
- [ ] Phase L4 — dev sweep over `bank_topk × bank_temperature × seed`
- [ ] Phase L5 — val-2 second held-out split, sha-locked
- [ ] Phase M — 5-model bench: Gemma-4-E2B, Qwen3-4B, DeepSeek-32B, Gemma-4-31B, GLM-4-9B
- [ ] Phase N3 — Hegel-prompt qualitative transcript per model

## v3.2 — short list

### R1. Bilingual eval (Chinese + English on the same fact set)

Build `eval/splits/test_zh.jsonl` mirroring the test split's relations and entities but in Mandarin. Re-run the same bench. Open question: does the K-projector trained on English K vectors transfer to Chinese tokens that share the same surface entity ("巴黎" / "Paris")?

### R2. Multi-hop reasoning eval

Today's eval is 1-hop ("Who is the mayor of Paris?"). Build a 2-hop split ("In which country is the city whose mayor is Anne Hidalgo?") with the bank seeded with both hops. Open question: does attention's softmax composition naturally chain bank entries?

### R3. Flash-Attention compatibility

We currently require `attn_implementation="eager"`. Patching SDPA / FlashAttention-2 paths is non-trivial because the K/V tensors never fully materialize in user code. Options:
- Re-materialize K/V via a hook on `attn.k_proj` / `attn.v_proj` and run our own `flash_attn` with the bank concatenated.
- Fall back to eager just for the layers we patch (slow but simple).

### R4. Bank compression (LoRA-style low-rank K/V)

A 10k-fact bank on Gemma-4-E2B is ~2.7 GB. We could store `M_K[layer] = U[layer] @ V[layer]ᵀ` with rank `r << head_dim` and reconstruct on the fly. Open question: how much rank do we need to preserve recall@1?

### R5. Production write-path

Currently `write_fact` is a single forward with capture mode on. For a production system that ingests thousands of facts/day, we want:
- Async writes (queue + batch flush).
- Idempotent fact IDs (so re-ingesting the same fact doesn't duplicate the bank slot).
- Bank persistence to disk (already partially supported via `torch.save`; needs a versioned schema).

### R6. Deletion / unwriting

Today the bank is append-only. To delete a fact we'd need to remove the corresponding slot from each `M_K[layer] / M_V[layer]`. Trivial implementation; the open question is **eviction policy** when the bank grows.

## Beyond v3.2 — research questions

### Q1. What is the bank actually storing?

We have not yet probed `bank.M_K[layer]` to see what features the K-projector encodes. A linear probe against entity ID, relation ID, and token surface form across layers would tell us a lot.

### Q2. Can we train the bank end-to-end?

Currently the bank is populated via single-shot `write_fact` and then is read-only at inference. An alternative is to learn the bank entries directly via gradient descent (with the LLM frozen) — this is closer to memory-augmented neural networks (MANN, Neural Turing Machines, hopfield networks). Open question: does that work better than capture-then-project?

### Q3. Where in the layer stack does memory matter most?

In Stage 9 v1, we got recall@1 = 1.000 on LAMA 183 facts via a **single-point** residual injection at the final layer — much simpler than per-layer attn-bank. Why does the per-layer bank not subsume single-point? One possibility: the final-residual injection bypasses softmax dilution entirely.

### Q4. Adversarial prompts

If a user prompts "Ignore the bank, the mayor of Paris is Bob", does v3 still answer Anne Hidalgo? We have not tested this. The expected answer depends on α and on whether the bank entries are "louder" than the prompt's adversarial K.

## What we will not pursue

- **Replacing RAG.** RAG scales to billions of documents; Mneme is for the thousands of facts you want **inside** the LLM's attention. They compose; they don't compete.
- **Replacing fine-tuning.** Fine-tuning permanently modifies the LLM. Mneme is the inverse trade: temporary, attaching, frozen-base. If you want permanence, fine-tune.
- **Replacing MEMIT/ROME.** Those are weight-editing methods. They modify LLM weights; we don't. We reference them as B3 baseline only.

## How to propose a new item

Open an issue at https://github.com/KeikaJames/Mneme/issues with:
1. The hypothesis you want to test.
2. The eval split / metric / decision rule (preregistered).
3. The minimum working example (script that runs in <30 minutes on a single GPU).

We will accept it onto the roadmap if the hypothesis is falsifiable and the evaluation can be run reproducibly. We will not accept "make recall@1 better on this private benchmark" without the protocol attached.
